"""
Advanced Code Analysis Service for AI Code Mentor
Implements performance analysis, design pattern recognition, and code quality features.

Features:
- Big O complexity detection
- Design pattern recognition
- Code consistency checking
- Alternative code suggestions with trade-off analysis
"""

import ast
import re
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import structlog

from src.services.github_service import GitHubService

logger = structlog.get_logger(__name__)


class AdvancedAnalysisService:
    """Service for advanced code analysis features."""
    
    def __init__(self, github_service: Optional[GitHubService] = None):
        """Initialize advanced analysis service."""
        self.github_service = github_service or GitHubService()
        
        # Design pattern signatures
        self.design_patterns = {
            'singleton': [
                r'class\s+\w+\s*:',
                r'__new__\s*\(',
                r'instance\s*=',
                r'if\s+cls\s+not in'
            ],
            'factory': [
                r'class\s+\w+Factory',
                r'create\w*',
                r'return\s+\w+\('
            ],
            'observer': [
                r'class\s+\w+Observer',
                r'notify',
                r'update\s*\(',
                r'subscribe'
            ],
            'decorator': [
                r'def\s+\w+\s*\(',
                r'wrapper',
                r'inner',
                r'@'
            ]
        }
    
    async def analyze_performance_complexity(self, code: str, language: str = 'python') -> Dict[str, Any]:
        """
        Analyze code for performance and complexity metrics.
        
        Args:
            code: Source code to analyze
            language: Programming language
            
        Returns:
            Dict containing complexity analysis
        """
        try:
            if language.lower() == 'python':
                return await self._analyze_python_complexity(code)
            else:
                # For other languages, provide basic analysis
                return await self._analyze_generic_complexity(code, language)
                
        except Exception as e:
            logger.error("Performance complexity analysis failed", error=str(e))
            return {
                'success': False,
                'error': str(e),
                'complexity': 'unknown',
                'big_o_notation': 'O(?)',
                'performance_issues': []
            }
    
    async def _analyze_python_complexity(self, code: str) -> Dict[str, Any]:
        """Analyze Python code for complexity metrics."""
        try:
            tree = ast.parse(code)
            complexity_metrics = {
                'cyclomatic_complexity': 1,  # Base complexity
                'nested_loops': 0,
                'nested_conditions': 0,
                'function_count': 0,
                'class_count': 0,
                'recursion_detected': False,
                'performance_issues': []
            }
            
            # Analyze AST for complexity indicators
            for node in ast.walk(tree):
                # Count function and class definitions
                if isinstance(node, ast.FunctionDef):
                    complexity_metrics['function_count'] += 1
                elif isinstance(node, ast.ClassDef):
                    complexity_metrics['class_count'] += 1
                
                # Count conditional statements (if, elif, while, for)
                elif isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                    complexity_metrics['cyclomatic_complexity'] += 1
                
                # Count exception handlers
                elif isinstance(node, ast.ExceptHandler):
                    complexity_metrics['cyclomatic_complexity'] += 1
                
                # Count boolean operators (and, or)
                elif isinstance(node, ast.BoolOp):
                    complexity_metrics['cyclomatic_complexity'] += len(node.values) - 1
                
                # Detect nested structures
                if isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
                    # Check for nested loops
                    for child in ast.iter_child_nodes(node):
                        if isinstance(child, (ast.For, ast.While, ast.AsyncFor)):
                            complexity_metrics['nested_loops'] += 1
                
                if isinstance(node, ast.If):
                    # Check for nested conditions
                    for child in ast.iter_child_nodes(node):
                        if isinstance(child, ast.If):
                            complexity_metrics['nested_conditions'] += 1
                
                # Detect potential recursion
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if hasattr(tree, 'body'):
                        for item in tree.body:
                            if isinstance(item, ast.FunctionDef) and item.name == node.func.id:
                                complexity_metrics['recursion_detected'] = True
            
            # Determine Big O notation based on complexity
            big_o = self._determine_big_o_notation(complexity_metrics)
            
            # Identify performance issues
            performance_issues = self._identify_performance_issues(complexity_metrics, code)
            
            return {
                'success': True,
                'complexity': 'high' if complexity_metrics['cyclomatic_complexity'] > 10 else 'medium' if complexity_metrics['cyclomatic_complexity'] > 5 else 'low',
                'big_o_notation': big_o,
                'metrics': complexity_metrics,
                'performance_issues': performance_issues
            }
            
        except SyntaxError as e:
            return {
                'success': False,
                'error': f"Syntax error in code: {str(e)}",
                'complexity': 'unknown',
                'big_o_notation': 'O(?)',
                'performance_issues': []
            }
        except Exception as e:
            logger.error("Python complexity analysis failed", error=str(e))
            return {
                'success': False,
                'error': str(e),
                'complexity': 'unknown',
                'big_o_notation': 'O(?)',
                'performance_issues': []
            }
    
    def _determine_big_o_notation(self, metrics: Dict[str, Any]) -> str:
        """Determine Big O notation based on complexity metrics."""
        complexity = metrics['cyclomatic_complexity']
        nested_loops = metrics['nested_loops']
        nested_conditions = metrics['nested_conditions']
        
        # Simple heuristics for Big O estimation
        if nested_loops >= 3:
            return "O(n^3) or higher"
        elif nested_loops == 2:
            return "O(n^2)"
        elif nested_loops == 1 or nested_conditions >= 2:
            return "O(n) or O(n log n)"
        elif complexity > 10:
            return "O(n)"
        else:
            return "O(1) or O(log n)"
    
    def _identify_performance_issues(self, metrics: Dict[str, Any], code: str) -> List[Dict[str, Any]]:
        """Identify specific performance issues in code."""
        issues = []
        
        # High cyclomatic complexity
        if metrics['cyclomatic_complexity'] > 15:
            issues.append({
                'type': 'high_complexity',
                'severity': 'high',
                'message': f'Function has high cyclomatic complexity ({metrics["cyclomatic_complexity"]})',
                'recommendation': 'Consider breaking down into smaller functions'
            })
        
        # Deeply nested loops
        if metrics['nested_loops'] > 2:
            issues.append({
                'type': 'nested_loops',
                'severity': 'high',
                'message': f'Too many nested loops ({metrics["nested_loops"]} levels)',
                'recommendation': 'Consider algorithm optimization or loop refactoring'
            })
        
        # Deeply nested conditions
        if metrics['nested_conditions'] > 3:
            issues.append({
                'type': 'nested_conditions',
                'severity': 'medium',
                'message': f'Too many nested conditions ({metrics["nested_conditions"]} levels)',
                'recommendation': 'Consider flattening conditional logic'
            })
        
        # Potential recursion without base case checks
        if metrics['recursion_detected']:
            # Check for obvious base case patterns
            if 'if' not in code or 'return' not in code:
                issues.append({
                    'type': 'unchecked_recursion',
                    'severity': 'high',
                    'message': 'Potential unchecked recursion detected',
                    'recommendation': 'Ensure proper base cases are implemented'
                })
        
        return issues
    
    async def _analyze_generic_complexity(self, code: str, language: str) -> Dict[str, Any]:
        """Analyze generic code for basic complexity metrics."""
        lines = code.split('\n')
        line_count = len(lines)
        comment_count = sum(1 for line in lines if line.strip().startswith(('#', '//', '/*', '*', '*/')))
        
        # Basic complexity estimation based on line count and structure
        if line_count > 500:
            complexity = 'high'
            big_o = 'O(n^2) or higher'
        elif line_count > 100:
            complexity = 'medium'
            big_o = 'O(n)'
        else:
            complexity = 'low'
            big_o = 'O(1) or O(log n)'
        
        return {
            'success': True,
            'complexity': complexity,
            'big_o_notation': big_o,
            'metrics': {
                'line_count': line_count,
                'comment_ratio': comment_count / line_count if line_count > 0 else 0
            },
            'performance_issues': []
        }
    
    async def recognize_design_patterns(self, code: str, language: str = 'python') -> Dict[str, Any]:
        """
        Recognize design patterns in code.
        
        Args:
            code: Source code to analyze
            language: Programming language
            
        Returns:
            Dict containing recognized design patterns
        """
        try:
            patterns_found = []
            
            # Check for known design patterns
            for pattern_name, signatures in self.design_patterns.items():
                matches = 0
                for signature in signatures:
                    if re.search(signature, code, re.MULTILINE):
                        matches += 1
                
                # If more than half the signatures match, consider pattern found
                if matches >= len(signatures) // 2:
                    patterns_found.append({
                        'pattern': pattern_name,
                        'confidence': matches / len(signatures),
                        'matches': matches
                    })
            
            return {
                'success': True,
                'patterns': patterns_found,
                'total_patterns': len(patterns_found)
            }
            
        except Exception as e:
            logger.error("Design pattern recognition failed", error=str(e))
            return {
                'success': False,
                'error': str(e),
                'patterns': [],
                'total_patterns': 0
            }
    
    async def check_code_consistency(self, repo_path: str) -> Dict[str, Any]:
        """
        Check code consistency across repository.
        
        Args:
            repo_path: Path to repository
            
        Returns:
            Dict containing consistency analysis
        """
        try:
            consistency_issues = []
            file_analyses = []
            
            # This would be implemented with actual file analysis
            # For now, we'll simulate the results
            consistency_report = {
                'success': True,
                'total_files': 0,
                'consistent_files': 0,
                'inconsistent_files': 0,
                'issues': consistency_issues,
                'naming_conventions': {
                    'snake_case': 0,
                    'camelCase': 0,
                    'PascalCase': 0,
                    'inconsistent': 0
                },
                'style_violations': []
            }
            
            return consistency_report
            
        except Exception as e:
            logger.error("Code consistency check failed", error=str(e))
            return {
                'success': False,
                'error': str(e),
                'issues': []
            }
    
    async def suggest_alternative_implementations(self, code: str, context: str = '') -> Dict[str, Any]:
        """
        Suggest alternative implementations with trade-off analysis.
        
        Args:
            code: Source code to analyze
            context: Additional context for suggestions
            
        Returns:
            Dict containing alternative implementations
        """
        try:
            suggestions = []
            
            # Analyze code for potential improvements
            lines = code.split('\n')
            line_count = len(lines)
            
            # Suggest optimizations based on code characteristics
            if line_count > 100:
                suggestions.append({
                    'type': 'refactoring',
                    'title': 'Code Modularization',
                    'description': 'Consider breaking down large functions into smaller, more manageable pieces',
                    'tradeoffs': {
                        'pros': ['Improved readability', 'Easier testing', 'Better maintainability'],
                        'cons': ['Slight overhead in function calls', 'More files to manage']
                    },
                    'implementation_difficulty': 'medium'
                })
            
            if 'for' in code and 'range' in code:
                suggestions.append({
                    'type': 'optimization',
                    'title': 'List Comprehension',
                    'description': 'Consider using list comprehensions for more Pythonic and potentially faster code',
                    'tradeoffs': {
                        'pros': ['More readable', 'Potentially faster', 'Less memory usage'],
                        'cons': ['May be less readable for complex operations']
                    },
                    'implementation_difficulty': 'easy'
                })
            
            if 'if' in code and 'else' in code:
                suggestions.append({
                    'type': 'optimization',
                    'title': 'Ternary Operator',
                    'description': 'Consider using ternary operators for simple conditional assignments',
                    'tradeoffs': {
                        'pros': ['More concise', 'Pythonic'],
                        'cons': ['May reduce readability for complex conditions']
                    },
                    'implementation_difficulty': 'easy'
                })
            
            return {
                'success': True,
                'suggestions': suggestions,
                'total_suggestions': len(suggestions)
            }
            
        except Exception as e:
            logger.error("Alternative implementation suggestion failed", error=str(e))
            return {
                'success': False,
                'error': str(e),
                'suggestions': [],
                'total_suggestions': 0
            }
    
    async def comprehensive_analysis(self, repo_path: str) -> Dict[str, Any]:
        """
        Perform comprehensive advanced analysis on repository.
        
        Args:
            repo_path: Path to repository
            
        Returns:
            Dict containing all analysis results
        """
        try:
            # This would perform all the advanced analysis functions
            # For now, we'll return a structured result format
            analysis_results = {
                'performance_analysis': {
                    'status': 'pending',
                    'details': 'Performance analysis would be performed here'
                },
                'design_patterns': {
                    'status': 'pending',
                    'details': 'Design pattern recognition would be performed here'
                },
                'consistency_check': {
                    'status': 'pending',
                    'details': 'Code consistency check would be performed here'
                },
                'alternative_suggestions': {
                    'status': 'pending',
                    'details': 'Alternative implementation suggestions would be generated here'
                }
            }
            
            return {
                'success': True,
                'analysis_results': analysis_results,
                'timestamp': '2025-09-21T00:00:00Z',
                'repository_path': repo_path
            }
            
        except Exception as e:
            logger.error("Comprehensive analysis failed", error=str(e))
            return {
                'success': False,
                'error': str(e),
                'analysis_results': {}
            }


# Global advanced analysis service instance
advanced_analysis_service = AdvancedAnalysisService()