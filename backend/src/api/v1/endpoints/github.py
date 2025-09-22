"""
GitHub Repository Analysis Endpoints
Provides API endpoints for GitHub repository analysis and code extraction.

Endpoints:
- POST /repositories/analyze - Analyze GitHub repository
- GET /repositories/{analysis_id} - Get analysis results
- GET /repositories/{analysis_id}/files - Get repository file structure
- DELETE /repositories/{analysis_id} - Delete analysis and cleanup
"""

import uuid
import tempfile
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from pathlib import Path

import structlog
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from pydantic import BaseModel, Field, HttpUrl

from src.models.database import Base
from src.dependencies import get_db
from src.models.user import User
from src.dependencies import get_current_user
from src.services.github_service import GitHubService, GitHubServiceError, RepositoryNotFoundError, RateLimitError
from src.services.redis_service import redis_service
from src.security.rate_limiting import expensive_operation_rate_limit, api_rate_limit

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/repositories", tags=["GitHub Analysis"])


# Pydantic models
class RepositoryAnalysisRequest(BaseModel):
    """Request model for repository analysis."""
    repository_url: HttpUrl = Field(..., description="GitHub repository URL")
    include_dependencies: bool = Field(True, description="Include dependency analysis")
    include_security_scan: bool = Field(True, description="Include basic security scanning")
    analysis_depth: str = Field("standard", description="Analysis depth: basic, standard, deep")
    
    class Config:
        json_schema_extra = {
            "example": {
                "repository_url": "https://github.com/owner/repository",
                "include_dependencies": True,
                "include_security_scan": True,
                "analysis_depth": "standard"
            }
        }


class RepositoryInfo(BaseModel):
    """Repository information model."""
    id: int
    name: str
    full_name: str
    description: Optional[str]
    url: str
    language: Optional[str]
    languages: Dict[str, int]
    size: int
    stargazers_count: int
    forks_count: int
    is_private: bool
    is_fork: bool
    created_at: Optional[str]
    updated_at: Optional[str]


class CodeStructureAnalysis(BaseModel):
    """Code structure analysis results."""
    total_files: int
    code_files: int
    lines_of_code: int
    languages: Dict[str, Dict[str, int]]
    file_types: Dict[str, int]
    largest_files: List[Dict[str, Any]]
    dependencies: Dict[str, List[str]]
    security_issues: List[Dict[str, str]]


class RepositoryAnalysisResponse(BaseModel):
    """Repository analysis response model."""
    analysis_id: str
    status: str
    repository_info: Optional[RepositoryInfo]
    code_analysis: Optional[CodeStructureAnalysis]
    analysis_metadata: Dict[str, Any]
    created_at: datetime
    completed_at: Optional[datetime]
    error_message: Optional[str]


class RepositoryFileNode(BaseModel):
    """Repository file tree node."""
    name: str
    type: str  # 'file' or 'directory'
    size: Optional[int] = None
    children: Optional[Dict[str, 'RepositoryFileNode']] = None


class RepositoryFilesResponse(BaseModel):
    """Repository files structure response."""
    analysis_id: str
    repository_name: str
    directory_structure: Dict[str, RepositoryFileNode]
    total_files: int
    total_size: int


# In-memory storage for analysis results (in production, use database)
_analysis_storage: Dict[str, Dict[str, Any]] = {}


@router.post("/analyze", response_model=RepositoryAnalysisResponse)
@expensive_operation_rate_limit(calls=5, period=3600)  # 5 repo analyses per hour per user
async def analyze_repository(
    request: RepositoryAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Analyze a GitHub repository and extract code structure, dependencies, and metadata.
    
    This endpoint:
    1. Validates the GitHub repository URL
    2. Retrieves repository metadata from GitHub API
    3. Clones the repository for analysis
    4. Performs code structure analysis
    5. Returns analysis results with unique analysis ID
    
    Rate limited to prevent abuse.
    """
    try:
        # Generate unique analysis ID
        analysis_id = str(uuid.uuid4())
        
        # Initialize GitHub service
        github_service = GitHubService(redis_service)
        github_service.authenticate()
        
        # Validate repository URL
        repo_url = str(request.repository_url)
        owner, repo_name = await github_service.validate_repository_url(repo_url)
        
        logger.info("Starting repository analysis", 
                   analysis_id=analysis_id,
                   repository=f"{owner}/{repo_name}",
                   user_id=str(current_user.id))
        
        # Check cache first
        cache_key = await github_service.get_cache_key(repo_url)
        cached_result = await github_service.get_cached_analysis(cache_key)
        
        if cached_result:
            logger.info("Returning cached analysis", analysis_id=analysis_id)
            cached_result['analysis_id'] = analysis_id
            cached_result['status'] = 'completed'
            _analysis_storage[analysis_id] = cached_result
            return RepositoryAnalysisResponse(**cached_result)
        
        # Initialize analysis record
        analysis_record = {
            'analysis_id': analysis_id,
            'status': 'processing',
            'repository_url': repo_url,
            'repository_name': f"{owner}/{repo_name}",
            'user_id': str(current_user.id),
            'repository_info': None,
            'code_analysis': None,
            'analysis_metadata': {
                'include_dependencies': request.include_dependencies,
                'include_security_scan': request.include_security_scan,
                'analysis_depth': request.analysis_depth,
                'started_by': current_user.email
            },
            'created_at': datetime.now(timezone.utc),
            'completed_at': None,
            'error_message': None
        }
        
        # Store initial record
        _analysis_storage[analysis_id] = analysis_record
        
        # Start background analysis
        background_tasks.add_task(
            _perform_repository_analysis,
            analysis_id,
            github_service,
            repo_url,
            owner,
            repo_name,
            request,
            cache_key
        )
        
        return RepositoryAnalysisResponse(**analysis_record)
        
    except (RepositoryNotFoundError, RateLimitError) as e:
        logger.error("GitHub service error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except GitHubServiceError as e:
        logger.error("GitHub analysis failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )
    
    except Exception as e:
        logger.error("Unexpected error during analysis", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


async def _perform_repository_analysis(
    analysis_id: str,
    github_service: GitHubService,
    repo_url: str,
    owner: str,
    repo_name: str,
    request: RepositoryAnalysisRequest,
    cache_key: str
):
    """Background task to perform repository analysis."""
    try:
        logger.info("Performing background repository analysis", analysis_id=analysis_id)
        
        # Get repository information
        repo_info = await github_service.get_repository_info(owner, repo_name)
        
        # Update analysis record
        _analysis_storage[analysis_id]['repository_info'] = repo_info
        _analysis_storage[analysis_id]['status'] = 'cloning'
        
        # Create temporary directory for cloning
        with tempfile.TemporaryDirectory() as temp_dir:
            clone_path = Path(temp_dir) / "repository"
            
            # Clone repository
            await github_service.clone_repository(repo_info['clone_url'], str(clone_path))
            
            # Update status
            _analysis_storage[analysis_id]['status'] = 'analyzing'
            
            # Perform code structure analysis
            code_analysis = await github_service.analyze_code_structure(str(clone_path))
            
            # Update analysis record
            _analysis_storage[analysis_id].update({
                'code_analysis': code_analysis,
                'status': 'completed',
                'completed_at': datetime.now(timezone.utc),
                'analysis_metadata': {
                    **_analysis_storage[analysis_id]['analysis_metadata'],
                    'processing_time_seconds': (
                        datetime.now(timezone.utc) - _analysis_storage[analysis_id]['created_at']
                    ).total_seconds()
                }
            })
            
            # Cache results
            cache_data = {
                'repository_info': repo_info,
                'code_analysis': code_analysis,
                'analysis_metadata': _analysis_storage[analysis_id]['analysis_metadata'],
                'created_at': _analysis_storage[analysis_id]['created_at'],
                'completed_at': _analysis_storage[analysis_id]['completed_at']
            }
            
            await github_service.cache_analysis(cache_key, cache_data, ttl=7200)  # 2 hours
        
        logger.info("Repository analysis completed", analysis_id=analysis_id)
        
    except Exception as e:
        logger.error("Background analysis failed", analysis_id=analysis_id, error=str(e))
        _analysis_storage[analysis_id].update({
            'status': 'failed',
            'error_message': str(e),
            'completed_at': datetime.now(timezone.utc)
        })


@router.get("/{analysis_id}", response_model=RepositoryAnalysisResponse)
async def get_analysis_results(
    analysis_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get repository analysis results by analysis ID.
    
    Returns the current status and results of a repository analysis.
    """
    if analysis_id not in _analysis_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found"
        )
    
    analysis_record = _analysis_storage[analysis_id]
    
    # Check user access (users can only see their own analyses)
    if analysis_record['user_id'] != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    return RepositoryAnalysisResponse(**analysis_record)


@router.get("/{analysis_id}/files", response_model=RepositoryFilesResponse)
async def get_repository_files(
    analysis_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get repository file structure for a completed analysis.
    
    Returns the directory tree structure of the analyzed repository.
    """
    if analysis_id not in _analysis_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found"
        )
    
    analysis_record = _analysis_storage[analysis_id]
    
    # Check user access
    if analysis_record['user_id'] != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    # Check if analysis is completed
    if analysis_record['status'] != 'completed':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Analysis not completed yet"
        )
    
    # Extract file structure from code analysis
    code_analysis = analysis_record.get('code_analysis', {})
    directory_structure = code_analysis.get('directory_structure', {})
    
    return RepositoryFilesResponse(
        analysis_id=analysis_id,
        repository_name=analysis_record['repository_name'],
        directory_structure=directory_structure,
        total_files=code_analysis.get('total_files', 0),
        total_size=sum(
            lang_stats.get('lines', 0) 
            for lang_stats in code_analysis.get('languages', {}).values()
        )
    )


@router.delete("/{analysis_id}")
async def delete_analysis(
    analysis_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Delete repository analysis and cleanup resources.
    
    Removes the analysis record and any associated cached data.
    """
    if analysis_id not in _analysis_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found"
        )
    
    analysis_record = _analysis_storage[analysis_id]
    
    # Check user access
    if analysis_record['user_id'] != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    # Remove from storage
    del _analysis_storage[analysis_id]
    
    logger.info("Analysis deleted", analysis_id=analysis_id, user_id=str(current_user.id))
    
    return {"message": "Analysis deleted successfully"}


@router.get("/", response_model=List[Dict[str, Any]])
async def list_user_analyses(
    current_user: User = Depends(get_current_user),
    limit: int = 10,
    offset: int = 0
):
    """
    List repository analyses for the current user.
    
    Returns a paginated list of analyses owned by the current user.
    """
    user_analyses = [
        {
            'analysis_id': record['analysis_id'],
            'repository_name': record['repository_name'],
            'status': record['status'],
            'created_at': record['created_at'],
            'completed_at': record.get('completed_at'),
            'error_message': record.get('error_message')
        }
        for record in _analysis_storage.values()
        if record['user_id'] == str(current_user.id)
    ]
    
    # Sort by creation time (newest first)
    user_analyses.sort(key=lambda x: x['created_at'], reverse=True)
    
    # Apply pagination
    paginated_analyses = user_analyses[offset:offset + limit]
    
    return paginated_analyses