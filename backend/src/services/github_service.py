"""
GitHub Service for AI Code Mentor
Handles GitHub repository analysis, cloning, and code extraction.

Features:
- Repository cloning and validation
- Multi-language code parsing with Tree-sitter
- Code structure analysis and metadata extraction
- Security scanning for repositories
- Rate limiting and authentication handling
"""

import os
import shutil
import tempfile
import asyncio
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from urllib.parse import urlparse
import json
import hashlib
from datetime import datetime, timezone

import structlog
import aiofiles
import aiohttp
from git import Repo, GitCommandError, InvalidGitRepositoryError
from github import Github, RateLimitExceededException, UnknownObjectException
import tree_sitter
import tree_sitter_python
import tree_sitter_javascript
import tree_sitter_typescript

from src.config.settings import get_settings
from src.services.redis_service import RedisService

logger = structlog.get_logger(__name__)
settings = get_settings()


class GitHubServiceError(Exception):
    """Base exception for GitHub service errors."""
    pass


class RepositoryNotFoundError(GitHubServiceError):
    """Repository not found or access denied."""
    pass


class RateLimitError(GitHubServiceError):
    """GitHub API rate limit exceeded."""
    pass


class AnalysisError(GitHubServiceError):
    """Error during code analysis."""
    pass


class GitHubService:
    """Service for GitHub repository analysis and code extraction."""
    
    def __init__(self, redis_service: Optional[RedisService] = None):
        """Initialize GitHub service with optional Redis caching."""
        self.github_client = None
        self.redis_service = redis_service
        
        # Initialize Tree-sitter languages
        self.languages = self._init_tree_sitter()
        
        # Supported file extensions for analysis
        self.supported_extensions = {
            '.py': 'python',
            '.js': 'javascript', 
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.cs': 'csharp',
            '.php': 'php',
            '.rb': 'ruby',
            '.go': 'go',
            '.rs': 'rust',
            '.swift': 'swift',
            '.kt': 'kotlin'
        }
        
        # Binary file extensions to skip
        self.binary_extensions = {
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.ico', '.svg',
            '.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm',
            '.mp3', '.wav', '.flac', '.ogg', '.wma',
            '.zip', '.rar', '.7z', '.tar', '.gz', '.bz2',
            '.exe', '.dll', '.so', '.dylib', '.bin',
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx'
        }
    
    def _init_tree_sitter(self) -> Dict[str, tree_sitter.Language]:
        """Initialize Tree-sitter language parsers."""
        languages = {}
        
        try:
            # Python
            languages['python'] = tree_sitter.Language(tree_sitter_python.language())
            
            # JavaScript
            languages['javascript'] = tree_sitter.Language(tree_sitter_javascript.language())
            
            # TypeScript
            languages['typescript'] = tree_sitter.Language(tree_sitter_typescript.language_typescript())
            
            logger.info("Tree-sitter languages initialized", languages=list(languages.keys()))
            
        except Exception as e:
            logger.error("Failed to initialize Tree-sitter languages", error=str(e))
        
        return languages
    
    def authenticate(self, github_token: Optional[str] = None) -> None:
        """Authenticate with GitHub API."""
        token = github_token or settings.GITHUB_TOKEN
        if token:
            self.github_client = Github(token)
            logger.info("GitHub client authenticated")
        else:
            self.github_client = Github()  # Unauthenticated (lower rate limits)
            logger.warning("GitHub client initialized without authentication")
    
    async def validate_repository_url(self, repo_url: str) -> Tuple[str, str]:
        """
        Validate and parse GitHub repository URL.
        
        Returns:
            Tuple of (owner, repo_name)
        """
        try:
            parsed = urlparse(repo_url)
            
            # Handle different GitHub URL formats
            if parsed.netloc == 'github.com':
                path_parts = parsed.path.strip('/').split('/')
                if len(path_parts) >= 2:
                    owner = path_parts[0]
                    repo_name = path_parts[1].replace('.git', '')
                    return owner, repo_name
            
            raise ValueError("Invalid GitHub URL format")
            
        except Exception as e:
            logger.error("Invalid repository URL", url=repo_url, error=str(e))
            raise GitHubServiceError(f"Invalid repository URL: {str(e)}")
    
    async def get_repository_info(self, owner: str, repo_name: str) -> Dict[str, Any]:
        """Get repository information from GitHub API."""
        if not self.github_client:
            self.authenticate()
        
        try:
            repo = self.github_client.get_repo(f"{owner}/{repo_name}")
            
            # Check rate limits
            rate_limit = self.github_client.get_rate_limit()
            if rate_limit.core.remaining < 10:
                logger.warning("GitHub API rate limit low", remaining=rate_limit.core.remaining)
            
            repo_info = {
                'id': repo.id,
                'name': repo.name,
                'full_name': repo.full_name,
                'description': repo.description,
                'url': repo.html_url,
                'clone_url': repo.clone_url,
                'default_branch': repo.default_branch,
                'language': repo.language,
                'languages': dict(repo.get_languages()) if repo.get_languages() else {},
                'size': repo.size,  # KB
                'stargazers_count': repo.stargazers_count,
                'forks_count': repo.forks_count,
                'open_issues_count': repo.open_issues_count,
                'created_at': repo.created_at.isoformat() if repo.created_at else None,
                'updated_at': repo.updated_at.isoformat() if repo.updated_at else None,
                'pushed_at': repo.pushed_at.isoformat() if repo.pushed_at else None,
                'is_private': repo.private,
                'is_fork': repo.fork,
                'is_archived': repo.archived,
                'license': repo.license.name if repo.license else None,
                'topics': repo.get_topics() if repo.get_topics() else []
            }
            
            logger.info("Repository info retrieved", repo=repo.full_name, size=repo.size)
            return repo_info
            
        except UnknownObjectException:
            logger.error("Repository not found", owner=owner, repo=repo_name)
            raise RepositoryNotFoundError(f"Repository {owner}/{repo_name} not found")
        
        except RateLimitExceededException:
            logger.error("GitHub API rate limit exceeded")
            raise RateLimitError("GitHub API rate limit exceeded")
        
        except Exception as e:
            logger.error("Failed to get repository info", error=str(e))
            raise GitHubServiceError(f"Failed to get repository info: {str(e)}")
    
    async def clone_repository(self, clone_url: str, target_dir: str) -> str:
        """Clone repository to local directory."""
        try:
            # Create target directory
            os.makedirs(target_dir, exist_ok=True)
            
            # Clone repository
            logger.info("Cloning repository", url=clone_url, target=target_dir)
            
            # Use asyncio to run git clone in thread pool
            loop = asyncio.get_event_loop()
            repo = await loop.run_in_executor(
                None, 
                lambda: Repo.clone_from(clone_url, target_dir, depth=1)
            )
            
            logger.info("Repository cloned successfully", target=target_dir)
            return target_dir
            
        except GitCommandError as e:
            logger.error("Git clone failed", error=str(e))
            raise GitHubServiceError(f"Failed to clone repository: {str(e)}")
        
        except Exception as e:
            logger.error("Repository cloning failed", error=str(e))
            raise GitHubServiceError(f"Repository cloning failed: {str(e)}")
    
    async def analyze_code_structure(self, repo_path: str) -> Dict[str, Any]:
        """Analyze repository code structure and extract metadata."""
        try:
            repo_path = Path(repo_path)
            
            analysis = {
                'total_files': 0,
                'code_files': 0,
                'lines_of_code': 0,
                'languages': {},
                'file_types': {},
                'directory_structure': {},
                'largest_files': [],
                'complexity_metrics': {},
                'dependencies': {},
                'security_issues': []
            }
            
            # Walk through repository files
            for file_path in repo_path.rglob('*'):
                if file_path.is_file() and not self._should_skip_file(file_path):
                    analysis['total_files'] += 1
                    
                    # Get file extension and language
                    ext = file_path.suffix.lower()
                    if ext in self.supported_extensions:
                        language = self.supported_extensions[ext]
                        analysis['code_files'] += 1
                        
                        # Count lines and analyze
                        file_stats = await self._analyze_file(file_path, language)
                        
                        # Update language statistics
                        if language not in analysis['languages']:
                            analysis['languages'][language] = {
                                'files': 0,
                                'lines': 0,
                                'functions': 0,
                                'classes': 0
                            }
                        
                        analysis['languages'][language]['files'] += 1
                        analysis['languages'][language]['lines'] += file_stats['lines']
                        analysis['languages'][language]['functions'] += file_stats.get('functions', 0)
                        analysis['languages'][language]['classes'] += file_stats.get('classes', 0)
                        
                        analysis['lines_of_code'] += file_stats['lines']
                        
                        # Track largest files
                        if file_stats['size'] > 0:
                            analysis['largest_files'].append({
                                'path': str(file_path.relative_to(repo_path)),
                                'size': file_stats['size'],
                                'lines': file_stats['lines'],
                                'language': language
                            })
                    
                    # Count file types
                    if ext not in analysis['file_types']:
                        analysis['file_types'][ext] = 0
                    analysis['file_types'][ext] += 1
            
            # Sort largest files
            analysis['largest_files'].sort(key=lambda x: x['size'], reverse=True)
            analysis['largest_files'] = analysis['largest_files'][:10]
            
            # Analyze dependencies
            analysis['dependencies'] = await self._analyze_dependencies(repo_path)
            
            # Basic security analysis
            analysis['security_issues'] = await self._basic_security_scan(repo_path)
            
            # Generate directory structure
            analysis['directory_structure'] = await self._generate_directory_tree(repo_path)
            
            logger.info("Code structure analysis completed", 
                       total_files=analysis['total_files'],
                       code_files=analysis['code_files'],
                       lines=analysis['lines_of_code'])
            
            return analysis
            
        except Exception as e:
            logger.error("Code structure analysis failed", error=str(e))
            raise AnalysisError(f"Code structure analysis failed: {str(e)}")
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped during analysis."""
        # Skip hidden files and directories
        if any(part.startswith('.') for part in file_path.parts):
            return True
        
        # Skip common non-code directories
        skip_dirs = {
            'node_modules', '__pycache__', '.git', 'build', 'dist', 
            'target', 'bin', 'obj', 'vendor', '.env', 'venv'
        }
        
        if any(part in skip_dirs for part in file_path.parts):
            return True
        
        # Skip binary files
        if file_path.suffix.lower() in self.binary_extensions:
            return True
        
        # Skip very large files (>10MB)
        try:
            if file_path.stat().st_size > 10 * 1024 * 1024:
                return True
        except (OSError, PermissionError):
            return True
        
        return False
    
    async def _analyze_file(self, file_path: Path, language: str) -> Dict[str, Any]:
        """Analyze individual file and extract metrics."""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = await f.read()
            
            stats = {
                'size': len(content),
                'lines': len(content.splitlines()),
                'functions': 0,
                'classes': 0,
                'imports': 0
            }
            
            # Use Tree-sitter for advanced analysis if available
            if language in self.languages:
                parser = tree_sitter.Parser()
                parser.set_language(self.languages[language])
                
                tree = parser.parse(content.encode('utf-8'))
                stats.update(self._extract_ast_metrics(tree, language))
            
            return stats
            
        except Exception as e:
            logger.warning("Failed to analyze file", file=str(file_path), error=str(e))
            return {'size': 0, 'lines': 0, 'functions': 0, 'classes': 0, 'imports': 0}
    
    def _extract_ast_metrics(self, tree: tree_sitter.Tree, language: str) -> Dict[str, int]:
        """Extract metrics from AST using Tree-sitter."""
        metrics = {'functions': 0, 'classes': 0, 'imports': 0}
        
        def traverse(node):
            if language == 'python':
                if node.type == 'function_definition':
                    metrics['functions'] += 1
                elif node.type == 'class_definition':
                    metrics['classes'] += 1
                elif node.type in ['import_statement', 'import_from_statement']:
                    metrics['imports'] += 1
            
            elif language in ['javascript', 'typescript']:
                if node.type in ['function_declaration', 'method_definition', 'arrow_function']:
                    metrics['functions'] += 1
                elif node.type == 'class_declaration':
                    metrics['classes'] += 1
                elif node.type == 'import_statement':
                    metrics['imports'] += 1
            
            for child in node.children:
                traverse(child)
        
        traverse(tree.root_node)
        return metrics
    
    async def _analyze_dependencies(self, repo_path: Path) -> Dict[str, List[str]]:
        """Analyze project dependencies from various package files."""
        dependencies = {}
        
        # Python dependencies
        for dep_file in ['requirements.txt', 'setup.py', 'pyproject.toml']:
            dep_path = repo_path / dep_file
            if dep_path.exists():
                dependencies['python'] = await self._parse_python_deps(dep_path)
                break
        
        # JavaScript/Node.js dependencies
        package_json = repo_path / 'package.json'
        if package_json.exists():
            dependencies['javascript'] = await self._parse_package_json(package_json)
        
        # Java dependencies
        pom_xml = repo_path / 'pom.xml'
        if pom_xml.exists():
            dependencies['java'] = ['Maven project detected']
        
        gradle_build = repo_path / 'build.gradle'
        if gradle_build.exists():
            dependencies['java'] = ['Gradle project detected']
        
        return dependencies
    
    async def _parse_python_deps(self, dep_file: Path) -> List[str]:
        """Parse Python dependencies from requirements file."""
        try:
            async with aiofiles.open(dep_file, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            deps = []
            for line in content.splitlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    # Extract package name (before ==, >=, etc.)
                    dep_name = line.split('==')[0].split('>=')[0].split('<=')[0].split('!=')[0].strip()
                    if dep_name:
                        deps.append(dep_name)
            
            return deps[:20]  # Limit to first 20 dependencies
            
        except Exception as e:
            logger.warning("Failed to parse Python dependencies", file=str(dep_file), error=str(e))
            return []
    
    async def _parse_package_json(self, package_file: Path) -> List[str]:
        """Parse JavaScript dependencies from package.json."""
        try:
            async with aiofiles.open(package_file, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            package_data = json.loads(content)
            deps = []
            
            # Get dependencies and devDependencies
            for dep_type in ['dependencies', 'devDependencies']:
                if dep_type in package_data:
                    deps.extend(list(package_data[dep_type].keys()))
            
            return deps[:20]  # Limit to first 20 dependencies
            
        except Exception as e:
            logger.warning("Failed to parse package.json", file=str(package_file), error=str(e))
            return []
    
    async def _basic_security_scan(self, repo_path: Path) -> List[Dict[str, str]]:
        """Perform basic security scanning for common issues."""
        issues = []
        
        # Check for common security issues
        security_patterns = {
            'hardcoded_secrets': [
                r'password\s*=\s*["\'][^"\']+["\']',
                r'api_key\s*=\s*["\'][^"\']+["\']',
                r'secret\s*=\s*["\'][^"\']+["\']',
                r'token\s*=\s*["\'][^"\']+["\']'
            ],
            'sql_injection': [
                r'SELECT.*FROM.*WHERE.*\+',
                r'INSERT.*VALUES.*\+',
                r'UPDATE.*SET.*\+'
            ]
        }
        
        # Scan code files for patterns
        for file_path in repo_path.rglob('*.py'):
            if not self._should_skip_file(file_path):
                try:
                    async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = await f.read()
                    
                    for issue_type, patterns in security_patterns.items():
                        for pattern in patterns:
                            import re
                            if re.search(pattern, content, re.IGNORECASE):
                                issues.append({
                                    'type': issue_type,
                                    'file': str(file_path.relative_to(repo_path)),
                                    'severity': 'medium',
                                    'description': f'Potential {issue_type.replace("_", " ")} detected'
                                })
                                break
                
                except Exception:
                    continue
        
        return issues[:10]  # Limit to first 10 issues
    
    async def _generate_directory_tree(self, repo_path: Path, max_depth: int = 3) -> Dict[str, Any]:
        """Generate directory structure tree."""
        def build_tree(path: Path, current_depth: int = 0) -> Dict[str, Any]:
            if current_depth > max_depth:
                return {'type': 'directory', 'truncated': True}
            
            tree = {'type': 'directory', 'children': {}}
            
            try:
                for item in sorted(path.iterdir()):
                    if item.name.startswith('.'):
                        continue
                    
                    if item.is_dir():
                        tree['children'][item.name] = build_tree(item, current_depth + 1)
                    else:
                        tree['children'][item.name] = {
                            'type': 'file',
                            'size': item.stat().st_size if item.exists() else 0
                        }
            except (PermissionError, OSError):
                tree['error'] = 'Access denied'
            
            return tree
        
        return build_tree(repo_path)
    
    async def cleanup_repository(self, repo_path: str) -> None:
        """Clean up cloned repository."""
        try:
            if os.path.exists(repo_path):
                shutil.rmtree(repo_path)
                logger.info("Repository cleaned up", path=repo_path)
        except Exception as e:
            logger.error("Failed to cleanup repository", path=repo_path, error=str(e))
    
    async def get_cache_key(self, repo_url: str) -> str:
        """Generate cache key for repository analysis."""
        url_hash = hashlib.md5(repo_url.encode()).hexdigest()
        return f"github_analysis:{url_hash}"
    
    async def cache_analysis(self, cache_key: str, analysis_data: Dict[str, Any], ttl: int = 3600) -> None:
        """Cache analysis results in Redis."""
        if self.redis_service:
            try:
                await self.redis_service.set_json(cache_key, analysis_data, ttl)
                logger.info("Analysis cached", cache_key=cache_key)
            except Exception as e:
                logger.warning("Failed to cache analysis", error=str(e))
    
    async def get_cached_analysis(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached analysis results from Redis."""
        if self.redis_service:
            try:
                return await self.redis_service.get_json(cache_key)
            except Exception as e:
                logger.warning("Failed to get cached analysis", error=str(e))
        return None