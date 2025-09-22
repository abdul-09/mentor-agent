"""
Authentication Endpoints for AI Code Mentor
Implements production-grade authentication with security compliance.

Compliance:
- RULE AUTH-001: bcrypt password hashing (cost factor 12+)
- RULE AUTH-002: JWT tokens (15min access, 7day refresh)
- RULE AUTH-003: MFA for sensitive operations
- RULE API-002: Correct HTTP status codes
"""

import structlog
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from src.config.settings import get_settings
from src.dependencies import get_db, get_current_user, get_current_session, get_client_ip, get_user_agent, get_current_active_user
from src.models.user import UserCreate, User
from src.services.auth import auth_service, AuthenticationError
from src.services.mfa_service import mfa_service
from src.services.email_service import email_service
from src.security.rate_limiting import auth_rate_limit, api_rate_limit

logger = structlog.get_logger(__name__)
settings = get_settings()
security = HTTPBearer()

router = APIRouter()


# Request/Response Models
class LoginRequest(BaseModel):
    """User login request model."""
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=8, description="User password")
    remember_me: bool = Field(default=False, description="Extended session duration")


class RegisterRequest(BaseModel):
    """User registration request model."""
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(
        ..., 
        min_length=8, 
        description="Password (min 8 characters, must contain uppercase, lowercase, number)"
    )
    full_name: str = Field(..., min_length=2, max_length=100, description="Full name")
    accept_terms: bool = Field(..., description="Accept terms and conditions")


class TokenResponse(BaseModel):
    """Authentication token response."""
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Access token expiration in seconds")
    user_id: str = Field(..., description="User ID")


class RefreshTokenRequest(BaseModel):
    """Token refresh request."""
    refresh_token: str = Field(..., description="Valid refresh token")


class PasswordResetRequest(BaseModel):
    """Password reset request."""
    email: EmailStr = Field(..., description="User email address")


class PasswordResetConfirm(BaseModel):
    """Password reset confirmation."""
    token: str = Field(..., description="Password reset token")
    new_password: str = Field(..., min_length=8, description="New password")


class MFASetupResponse(BaseModel):
    """MFA setup response."""
    qr_code_url: str = Field(..., description="QR code URL for authenticator app")
    secret_key: str = Field(..., description="Secret key for manual entry")
    backup_codes: list[str] = Field(..., description="Backup codes")


class MFAVerifyRequest(BaseModel):
    """MFA verification request."""
    code: str = Field(..., min_length=6, max_length=6, description="6-digit MFA code")


# Authentication endpoints
@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
@auth_rate_limit(calls=3, period=3600)  # 3 registrations per hour per IP
async def register(
    request: RegisterRequest, 
    http_request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Register a new user account.
    
    Implements RULE AUTH-001 with bcrypt password hashing.
    """
    if not request.accept_terms:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Terms and conditions must be accepted"
        )
    
    try:
        # Create user data
        user_data = UserCreate(
            email=request.email,
            password=request.password,
            full_name=request.full_name
        )
        
        # Register user
        user = await auth_service.register_user(
            user_data=user_data,
            db=db,
            ip_address=get_client_ip(http_request),
            user_agent=get_user_agent(http_request)
        )
        
        # Authenticate and create session
        user, session = await auth_service.authenticate_user(
            email=request.email,
            password=request.password,
            db=db,
            ip_address=get_client_ip(http_request),
            user_agent=get_user_agent(http_request)
        )
        
        # Create tokens
        tokens = await auth_service.create_tokens(user, session)
        
        return TokenResponse(
            access_token=tokens["access_token"],
            refresh_token=tokens["refresh_token"],
            token_type=tokens["token_type"],
            expires_in=tokens["expires_in"],
            user_id=tokens["user_id"]
        )
        
    except AuthenticationError as e:
        logger.error("Registration failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Registration failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/login", response_model=TokenResponse)
@auth_rate_limit(calls=5, period=900)  # 5 login attempts per 15 minutes per IP
async def login(
    request: LoginRequest, 
    http_request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Authenticate user and return JWT tokens.
    
    Implements RULE AUTH-002 with JWT token management.
    """
    try:
        # Authenticate user
        user, session = await auth_service.authenticate_user(
            email=request.email,
            password=request.password,
            db=db,
            ip_address=get_client_ip(http_request),
            user_agent=get_user_agent(http_request)
        )
        
        # Create tokens
        tokens = await auth_service.create_tokens(user, session)
        
        return TokenResponse(
            access_token=tokens["access_token"],
            refresh_token=tokens["refresh_token"],
            token_type=tokens["token_type"],
            expires_in=tokens["expires_in"],
            user_id=tokens["user_id"]
        )
        
    except AuthenticationError as e:
        logger.error("Login failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    except Exception as e:
        logger.error("Login failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication failed"
        )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: RefreshTokenRequest, 
    http_request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Refresh access token using refresh token.
    
    Implements RULE AUTH-002 with token rotation.
    """
    try:
        # Refresh tokens
        tokens = await auth_service.refresh_token(
            refresh_token=request.refresh_token,
            db=db,
            ip_address=get_client_ip(http_request)
        )
        
        return TokenResponse(
            access_token=tokens["access_token"],
            refresh_token=tokens["refresh_token"],
            token_type=tokens["token_type"],
            expires_in=tokens["expires_in"],
            user_id=tokens["user_id"]
        )
        
    except AuthenticationError as e:
        logger.error("Token refresh failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    except Exception as e:
        logger.error("Token refresh failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    http_request: Request, 
    db: AsyncSession = Depends(get_db),
    current_session = Depends(get_current_session)
):
    """
    Logout user and invalidate tokens.
    
    Implements RULE AUTH-004 with proper session invalidation.
    """
    try:
        # Logout user
        await auth_service.logout_user(
            session_id=str(current_session.id),
            db=db,
            revoke_all=False
        )
        
        logger.info(
            "User logged out successfully",
            user_id=str(current_session.user_id),
            session_id=str(current_session.id),
            ip_address=get_client_ip(http_request)
        )
        
    except AuthenticationError as e:
        logger.error("Logout failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Logout failed"
        )
    except Exception as e:
        logger.error("Logout failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )


@router.post("/mfa/setup", response_model=MFASetupResponse)
@api_rate_limit(calls=2, period=3600)  # 2 MFA setups per hour
async def setup_mfa(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Set up Multi-Factor Authentication.
    
    Implements RULE AUTH-003 for sensitive operations.
    """
    try:
        # Set up MFA for the current user
        setup_result = await mfa_service.setup_mfa(current_user, db)
        
        logger.info(
            "MFA setup completed",
            user_id=str(current_user.id),
            user_email=current_user.email
        )
        
        return MFASetupResponse(
            qr_code_url=setup_result["qr_code_url"],
            secret_key=setup_result["secret_key"],
            backup_codes=setup_result["backup_codes"]
        )
        
    except Exception as e:
        logger.error(
            "MFA setup failed",
            user_id=str(current_user.id),
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="MFA setup failed"
        )


@router.post("/mfa/verify", status_code=status.HTTP_200_OK)
@auth_rate_limit(calls=10, period=900)  # 10 MFA attempts per 15 minutes
async def verify_mfa(
    request: MFAVerifyRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Verify MFA code and complete authentication.
    
    Implements RULE AUTH-003 MFA verification.
    """
    try:
        # Verify the MFA code
        verification_result = await mfa_service.verify_mfa_code(
            user=current_user,
            provided_code=request.code,
            db=db
        )
        
        if verification_result.get("verified"):
            logger.info(
                "MFA verification successful",
                user_id=str(current_user.id),
                method=verification_result.get("method"),
                mfa_enabled=verification_result.get("mfa_enabled")
            )
            
            response_data = {"message": "MFA verified successfully"}
            
            # Add additional info for backup code usage
            if verification_result.get("method") == "backup_code":
                remaining_codes = verification_result.get("remaining_backup_codes", 0)
                response_data["remaining_backup_codes"] = remaining_codes
                
                if remaining_codes <= 2:
                    response_data["warning"] = "Low backup codes remaining. Consider regenerating."
            
            return response_data
        else:
            logger.warning(
                "MFA verification failed",
                user_id=str(current_user.id),
                error=verification_result.get("error")
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=verification_result.get("error", "Invalid MFA code")
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "MFA verification failed",
            user_id=str(current_user.id),
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid MFA code"
        )


@router.post("/mfa/disable", status_code=status.HTTP_200_OK)
async def disable_mfa(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Disable Multi-Factor Authentication for the current user.
    """
    try:
        success = await mfa_service.disable_mfa(current_user, db)
        
        if success:
            logger.info(
                "MFA disabled by user",
                user_id=str(current_user.id)
            )
            return {"message": "MFA disabled successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to disable MFA"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "MFA disable failed",
            user_id=str(current_user.id),
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to disable MFA"
        )


@router.post("/mfa/regenerate-backup-codes", status_code=status.HTTP_200_OK)
async def regenerate_backup_codes(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Regenerate backup codes for MFA.
    """
    try:
        backup_codes = await mfa_service.regenerate_backup_codes(current_user, db)
        
        if backup_codes:
            logger.info(
                "MFA backup codes regenerated",
                user_id=str(current_user.id),
                new_code_count=len(backup_codes)
            )
            return {
                "message": "Backup codes regenerated successfully",
                "backup_codes": backup_codes
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to regenerate backup codes. MFA must be enabled first."
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Backup codes regeneration failed",
            user_id=str(current_user.id),
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to regenerate backup codes"
        )


@router.post("/password-reset", status_code=status.HTTP_200_OK)
@auth_rate_limit(calls=3, period=3600)  # 3 password reset requests per hour per IP
async def request_password_reset(request: PasswordResetRequest, http_request: Request, db: AsyncSession = Depends(get_db)):
    """
    Request password reset email.
    
    Implements secure password reset with rate limiting.
    """
    try:
        # Validate email exists in database
        stmt = select(User).where(User.email == request.email, User.is_active == True)
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()
        
        if user:
            # Generate secure reset token (valid for 1 hour)
            reset_token = auth_service.generate_password_reset_token(user)
            
            # Store token in database with expiration
            user.password_reset_token = reset_token
            user.password_reset_expires = datetime.now(timezone.utc) + timedelta(hours=1)
            await db.commit()
            
            # Send reset email
            reset_link = f"{settings.FRONTEND_URL}/reset-password?token={reset_token}"
            email_result = await email_service.send_password_reset_email(
                to_email=user.email,
                reset_token=reset_token,
                user_name=user.full_name,
                reset_link=reset_link
            )
            
            if not email_result["success"]:
                logger.error(
                    "Failed to send password reset email",
                    user_id=str(user.id),
                    email=user.email,
                    error=email_result.get("error")
                )
            else:
                logger.info(
                    "Password reset email sent",
                    user_id=str(user.id),
                    email=user.email
                )
        
        # Always return success to prevent email enumeration
        return {"message": "If the email exists, a reset link has been sent"}
        
    except Exception as e:
        logger.error("Password reset request failed", error=str(e))
        return {"message": "If the email exists, a reset link has been sent"}


@router.post("/password-reset/confirm", status_code=status.HTTP_200_OK)
async def confirm_password_reset(request: PasswordResetConfirm, http_request: Request, db: AsyncSession = Depends(get_db)):
    """
    Confirm password reset with new password.
    
    Implements secure password reset confirmation.
    """
    try:
        # Validate reset token
        stmt = select(User).where(
            and_(
                User.password_reset_token == request.token,
                User.password_reset_expires > datetime.now(timezone.utc),
                User.is_active == True
            )
        )
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired reset token"
            )
        
        # Hash new password with bcrypt
        user.set_password(request.new_password)
        
        # Clear reset token
        user.password_reset_token = None
        user.password_reset_expires = None
        
        # Invalidate all existing tokens by updating the password change timestamp
        user.last_password_change = datetime.now(timezone.utc)
        
        await db.commit()
        
        logger.info(
            "Password reset successfully",
            user_id=str(user.id),
            ip_address=get_client_ip(http_request)
        )
        
        return {"message": "Password reset successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Password reset confirmation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token"
        )