"""
Dependency Injection for AI Code Mentor
Provides database sessions and authentication dependencies.

Compliance:
- RULE AUTH-002: JWT token validation
- RULE DB-003: Database session management
"""

from typing import Generator, Optional
from uuid import UUID

import structlog
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.database import get_async_session
from src.models.user import User
from src.models.session import UserSession
from src.services.auth import auth_service, AuthenticationError

logger = structlog.get_logger(__name__)
security = HTTPBearer(auto_error=False)


async def get_db() -> AsyncSession:
    """
    Get database session dependency.
    
    Yields:
        AsyncSession: Database session
    """
    async for session in get_async_session():
        yield session


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Get current authenticated user from JWT token.
    
    Args:
        request: FastAPI request object
        credentials: Bearer token credentials
        db: Database session
        
    Returns:
        User: Current authenticated user
        
    Raises:
        HTTPException: If authentication fails
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        user, session = await auth_service.verify_token(
            token=credentials.credentials,
            db=db,
            token_type="access"
        )
        
        # Add user and session to request state for logging
        request.state.user = user
        request.state.session = session
        
        return user
        
    except AuthenticationError as e:
        logger.warning(
            "Authentication failed",
            error=str(e),
            ip_address=request.client.host,
            trace_id=getattr(request.state, "trace_id", None)
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get current active user (additional check for active status).
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User: Current active user
        
    Raises:
        HTTPException: If user is inactive
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user account"
        )
    
    return current_user


async def get_current_verified_user(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """
    Get current verified user (requires email verification).
    
    Args:
        current_user: Current active user
        
    Returns:
        User: Current verified user
        
    Raises:
        HTTPException: If user email is not verified
    """
    if not current_user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email verification required"
        )
    
    return current_user


async def get_current_admin_user(
    current_user: User = Depends(get_current_verified_user)
) -> User:
    """
    Get current admin user (requires admin privileges).
    
    Args:
        current_user: Current verified user
        
    Returns:
        User: Current admin user
        
    Raises:
        HTTPException: If user is not an admin
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    
    return current_user


async def get_current_session(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> UserSession:
    """
    Get current user session from JWT token.
    
    Args:
        request: FastAPI request object
        credentials: Bearer token credentials
        db: Database session
        
    Returns:
        UserSession: Current user session
        
    Raises:
        HTTPException: If authentication fails
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        user, session = await auth_service.verify_token(
            token=credentials.credentials,
            db=db,
            token_type="access"
        )
        
        return session
        
    except AuthenticationError as e:
        logger.warning(
            "Session authentication failed",
            error=str(e),
            ip_address=request.client.host,
            trace_id=getattr(request.state, "trace_id", None)
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_user_agent(request: Request) -> str:
    """
    Extract user agent from request headers.
    
    Args:
        request: FastAPI request object
        
    Returns:
        str: User agent string
    """
    return request.headers.get("user-agent", "Unknown")


def get_client_ip(request: Request) -> str:
    """
    Extract client IP address from request.
    
    Args:
        request: FastAPI request object
        
    Returns:
        str: Client IP address
    """
    # Check for forwarded headers (behind proxy)
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip
    
    return request.client.host if request.client else "unknown"


async def validate_user_id(user_id: str) -> UUID:
    """
    Validate and convert user ID to UUID.
    
    Args:
        user_id: User ID string
        
    Returns:
        UUID: Validated user ID
        
    Raises:
        HTTPException: If user ID is invalid
    """
    try:
        return UUID(user_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user ID format"
        )