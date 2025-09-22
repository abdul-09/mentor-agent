#!/usr/bin/env python3
"""
Test script for GitHub service functionality
Tests the core GitHub repository analysis without requiring full app setup
"""

import sys
import os
import asyncio
import tempfile
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.services.github_service import GitHubService


async def test_github_service():
    """Test GitHub service functionality."""
    print("🚀 Testing GitHub Service...")
    
    # Initialize service
    github_service = GitHubService()
    
    # Test 1: Validate repository URL
    print("\n📝 Test 1: Repository URL validation")
    try:
        repo_url = "https://github.com/octocat/Hello-World"
        owner, repo_name = await github_service.validate_repository_url(repo_url)
        print(f"✅ URL validation successful: {owner}/{repo_name}")
    except Exception as e:
        print(f"❌ URL validation failed: {e}")
        return
    
    # Test 2: Repository info (without API key)
    print("\n🔍 Test 2: Repository info retrieval")
    try:
        github_service.authenticate()  # Will use unauthenticated mode
        repo_info = await github_service.get_repository_info(owner, repo_name)
        print(f"✅ Repository info retrieved: {repo_info['name']}")
        print(f"   Description: {repo_info.get('description', 'No description')}")
        print(f"   Language: {repo_info.get('language', 'Unknown')}")
        print(f"   Stars: {repo_info.get('stargazers_count', 0)}")
    except Exception as e:
        print(f"❌ Repository info failed: {e}")
        return
    
    # Test 3: Clone repository
    print("\n📦 Test 3: Repository cloning")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            clone_path = os.path.join(temp_dir, "test_repo")
            await github_service.clone_repository(repo_info['clone_url'], clone_path)
            print(f"✅ Repository cloned to: {clone_path}")
            
            # Test 4: Code structure analysis
            print("\n🔬 Test 4: Code structure analysis")
            try:
                analysis = await github_service.analyze_code_structure(clone_path)
                print(f"✅ Code analysis completed:")
                print(f"   Total files: {analysis['total_files']}")
                print(f"   Code files: {analysis['code_files']}")
                print(f"   Lines of code: {analysis['lines_of_code']}")
                print(f"   Languages: {list(analysis['languages'].keys())}")
                
                if analysis['largest_files']:
                    print(f"   Largest file: {analysis['largest_files'][0]['path']}")
                
                # Test 5: Cache functionality
                print("\n💾 Test 5: Cache functionality")
                cache_key = await github_service.get_cache_key(repo_url)
                print(f"✅ Cache key generated: {cache_key}")
                
                print("\n🎉 All tests passed! GitHub service is working correctly.")
                
            except Exception as e:
                print(f"❌ Code analysis failed: {e}")
                
    except Exception as e:
        print(f"❌ Repository cloning failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_github_service())