"""
User Management Endpoints for AI Code Mentor
Handles user profiles, preferences, and account management.

Compliance:
- RULE API-001: RESTful resource naming
- RULE PRIVACY-001: Personal data handling
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import select, func
from sqlalchemy.orm import selectinload

from src.dependencies import get_db, get_current_active_user
from src.models.user import User
from src.models.file import File
from src.models.analysis import AnalysisSession
from src.services.redis_service import redis_service
import structlog

logger = structlog.get_logger(__name__)
router = APIRouter()


# Pydantic Models
class UserProfileResponse(BaseModel):
    """User profile response model."""
    id: str
    email: str
    full_name: str
    is_email_verified: bool
    is_active: bool
    created_at: datetime
    updated_at: datetime
    last_login_at: Optional[datetime]
    
    class Config:
        from_attributes = True


class UserProfileUpdate(BaseModel):
    """User profile update model."""
    full_name: Optional[str] = Field(None, min_length=1, max_length=100)
    current_password: Optional[str] = Field(None, min_length=8)
    new_password: Optional[str] = Field(None, min_length=8)
    

class UserPreferences(BaseModel):
    """User preferences model."""
    ai_provider: Optional[str] = Field(None, description="Preferred AI provider")
    ai_model: Optional[str] = Field(None, description="Preferred AI model")
    language: Optional[str] = Field("en", description="Interface language")
    theme: Optional[str] = Field("light", description="UI theme")
    email_notifications: Optional[bool] = Field(True, description="Email notifications enabled")
    analysis_auto_start: Optional[bool] = Field(False, description="Auto-start analysis on upload")
    

class UserUsageStats(BaseModel):
    """User usage statistics model."""
    total_files_uploaded: int
    total_analyses_completed: int
    total_questions_asked: int
    storage_used_bytes: int
    storage_quota_bytes: int
    api_requests_today: int
    api_quota_daily: int
    last_activity: Optional[datetime]
    

class PreferencesUpdate(BaseModel):
    """Preferences update model."""
    preferences: UserPreferences


@router.get("/profile", response_model=UserProfileResponse, tags=["users"])
async def get_current_user_profile(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get current user profile with detailed information.
    
    Returns user profile including account status, verification state,
    and last activity information.
    """
    try:
        # Refresh user data from database
        result = await db.execute(
            select(User).where(User.id == current_user.id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        logger.info(
            "User profile retrieved",
            user_id=str(user.id),
            email=user.email
        )
        
        return UserProfileResponse(
            id=str(user.id),
            email=user.email,
            full_name=user.full_name,
            is_email_verified=user.is_email_verified,
            is_active=user.is_active,
            created_at=user.created_at,
            updated_at=user.updated_at,
            last_login_at=user.last_login_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to retrieve user profile",
            user_id=str(current_user.id),
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user profile"
        )


@router.put("/profile", response_model=UserProfileResponse, tags=["users"])
async def update_current_user_profile(
    update_data: UserProfileUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Update current user profile.
    
    Allows updating full name and password. Password change requires
    current password verification for security.
    """
    try:
        # Get fresh user data
        result = await db.execute(
            select(User).where(User.id == current_user.id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        updates_made = []
        
        # Update full name if provided
        if update_data.full_name is not None:
            user.full_name = update_data.full_name
            updates_made.append("full_name")
        
        # Handle password change
        if update_data.new_password is not None:
            if not update_data.current_password:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Current password required for password change"
                )
            
            # Verify current password
            from src.services.auth import auth_service
            if not auth_service.verify_password(update_data.current_password, user.password_hash):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Current password is incorrect"
                )
            
            # Hash and set new password
            user.password_hash = auth_service.hash_password(update_data.new_password)
            updates_made.append("password")
        
        # Update timestamp
        user.updated_at = datetime.now(timezone.utc)
        
        await db.commit()
        await db.refresh(user)
        
        logger.info(
            "User profile updated",
            user_id=str(user.id),
            updates=updates_made
        )
        
        return UserProfileResponse(
            id=str(user.id),
            email=user.email,
            full_name=user.full_name,
            is_email_verified=user.is_email_verified,
            is_active=user.is_active,
            created_at=user.created_at,
            updated_at=user.updated_at,
            last_login_at=user.last_login_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to update user profile",
            user_id=str(current_user.id),
            error=str(e)
        )
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user profile"
        )


@router.get("/usage", response_model=UserUsageStats, tags=["users"])
async def get_usage_statistics(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get user's usage statistics and quotas.
    
    Returns comprehensive usage data including file uploads,
    analyses completed, API usage, and storage consumption.
    """
    try:
        # Get file statistics
        file_stats = await db.execute(
            select(
                func.count(File.id).label('total_files'),
                func.coalesce(func.sum(File.file_size), 0).label('storage_used')
            )
            .where(
                File.user_id == current_user.id,
                File.status != 'deleted'
            )
        )
        file_data = file_stats.first()
        
        # Get analysis statistics
        analysis_stats = await db.execute(
            select(
                func.count(AnalysisSession.id).label('total_analyses')
            )
            .where(
                AnalysisSession.user_id == current_user.id,
                AnalysisSession.status == 'completed'
            )
        )
        analysis_data = analysis_stats.first()
        
        # Get Q&A statistics from Redis if available
        qa_count = 0
        api_requests_today = 0
        
        if redis_service.is_connected:
            try:
                # Get Q&A count from cache
                qa_count = await redis_service.cache_get(
                    f"user_qa_count:{current_user.id}", 
                    default=0
                )
                
                # Get today's API requests
                today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
                api_requests_today = await redis_service.cache_get(
                    f"user_api_requests:{current_user.id}:{today}",
                    default=0
                )
                
            except Exception as e:
                logger.warning("Failed to get cached usage stats", error=str(e))
        
        # Define quotas (these could be moved to user model or config)
        storage_quota = 1024 * 1024 * 1024  # 1GB default
        api_quota_daily = 1000  # 1000 requests per day default
        
        usage_stats = UserUsageStats(
            total_files_uploaded=file_data.total_files or 0,
            total_analyses_completed=analysis_data.total_analyses or 0,
            total_questions_asked=qa_count,
            storage_used_bytes=file_data.storage_used or 0,
            storage_quota_bytes=storage_quota,
            api_requests_today=api_requests_today,
            api_quota_daily=api_quota_daily,
            last_activity=current_user.last_login_at
        )
        
        logger.info(
            "Usage statistics retrieved",
            user_id=str(current_user.id),
            files=usage_stats.total_files_uploaded,
            analyses=usage_stats.total_analyses_completed
        )
        
        return usage_stats
        
    except Exception as e:
        logger.error(
            "Failed to retrieve usage statistics",
            user_id=str(current_user.id),
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve usage statistics"
        )


@router.get("/preferences", response_model=UserPreferences, tags=["users"])
async def get_user_preferences(
    current_user: User = Depends(get_current_active_user)
):
    """
    Get user preferences.
    
    Returns user's saved preferences for AI models, UI settings,
    and notification preferences.
    """
    try:
        # Get preferences from Redis cache or use defaults
        preferences_key = f"user_preferences:{current_user.id}"
        
        if redis_service.is_connected:
            cached_prefs = await redis_service.cache_get(preferences_key)
            if cached_prefs:
                return UserPreferences(**cached_prefs)
        
        # Return default preferences
        default_preferences = UserPreferences(
            ai_provider="gemini",
            ai_model="gemini-1.5-flash",
            language="en",
            theme="light",
            email_notifications=True,
            analysis_auto_start=False
        )
        
        logger.info(
            "User preferences retrieved (defaults)",
            user_id=str(current_user.id)
        )
        
        return default_preferences
        
    except Exception as e:
        logger.error(
            "Failed to retrieve user preferences",
            user_id=str(current_user.id),
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve preferences"
        )


@router.put("/preferences", response_model=UserPreferences, tags=["users"])
async def update_user_preferences(
    preferences_update: PreferencesUpdate,
    current_user: User = Depends(get_current_active_user)
):
    """
    Update user preferences.
    
    Allows updating AI provider, model selection, UI preferences,
    and notification settings.
    """
    try:
        preferences_key = f"user_preferences:{current_user.id}"
        
        # Get current preferences
        current_prefs = {}
        if redis_service.is_connected:
            current_prefs = await redis_service.cache_get(preferences_key, default={})
        
        # Update with new values
        updated_prefs = {**current_prefs}
        
        if preferences_update.preferences.ai_provider is not None:
            updated_prefs['ai_provider'] = preferences_update.preferences.ai_provider
        if preferences_update.preferences.ai_model is not None:
            updated_prefs['ai_model'] = preferences_update.preferences.ai_model
        if preferences_update.preferences.language is not None:
            updated_prefs['language'] = preferences_update.preferences.language
        if preferences_update.preferences.theme is not None:
            updated_prefs['theme'] = preferences_update.preferences.theme
        if preferences_update.preferences.email_notifications is not None:
            updated_prefs['email_notifications'] = preferences_update.preferences.email_notifications
        if preferences_update.preferences.analysis_auto_start is not None:
            updated_prefs['analysis_auto_start'] = preferences_update.preferences.analysis_auto_start
        
        # Save to Redis cache
        if redis_service.is_connected:
            await redis_service.cache_set(
                preferences_key, 
                updated_prefs, 
                ttl=86400 * 30  # Cache for 30 days
            )
        
        logger.info(
            "User preferences updated",
            user_id=str(current_user.id),
            updated_fields=list(updated_prefs.keys())
        )
        
        return UserPreferences(**updated_prefs)
        
    except Exception as e:
        logger.error(
            "Failed to update user preferences",
            user_id=str(current_user.id),
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update preferences"
        )