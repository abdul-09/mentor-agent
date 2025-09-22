"""
Authentication Service for AI Code Mentor
Implements production-grade authentication with security compliance.

Compliance:
- RULE AUTH-001: bcrypt password hashing (cost factor 12+)
- RULE AUTH-002: JWT tokens (15min access, 7day refresh)
- RULE AUTH-003: MFA for sensitive operations
- RULE AUTH-004: Session management security
"""

import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, Tuple

import jwt
import structlog
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from src.models.user import User, UserCreate
from src.models.session import UserSession, SessionStatus
from src.config.settings import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class AuthenticationError(Exception):
    """Authentication-related errors."""
    pass


class AuthorizationError(Exception):
    """Authorization-related errors."""
    pass


class AuthService:
    """
    Authentication service with production-grade security features.
    
    Implements all authentication rules and security best practices.
    """
    
    def __init__(self):
        self.secret_key = settings.JWT_SECRET_KEY
        self.algorithm = settings.JWT_ALGORITHM
        self.access_token_expire_minutes = settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
        self.refresh_token_expire_days = settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS
    
    def generate_password_reset_token(self, user: User) -> str:
        """
        Generate a secure password reset token.
        
        Args:
            user: User object
            
        Returns:
            str: Secure password reset token
        """
        # Create a unique token using user ID, timestamp, and random component
        token_data = f"{user.id}{datetime.now(timezone.utc).timestamp()}{uuid.uuid4()}"
        import hashlib
        token = hashlib.sha256(token_data.encode()).hexdigest()
        return token
    
    async def register_user(
        self,
        user_data: UserCreate,
        db: AsyncSession,
        ip_address: str = None,
        user_agent: str = None
    ) -> User:
        """
        Register a new user with secure password hashing.
        
        Implements RULE AUTH-001 with bcrypt password hashing.
        
        Args:
            user_data: User registration data
            db: Database session
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            User: Created user object
            
        Raises:
            AuthenticationError: If email already exists or validation fails
        """
        try:
            # Check if email already exists
            existing_user = await self._get_user_by_email(user_data.email, db)
            if existing_user:
                raise AuthenticationError("Email already registered")
            
            # Create new user
            user = User(
                email=user_data.email,
                full_name=user_data.full_name,
                is_active=True,
                is_verified=False  # Require email verification
            )
            
            # Set password with bcrypt (RULE AUTH-001)
            user.set_password(user_data.password)
            
            # Save to database
            db.add(user)
            await db.commit()
            await db.refresh(user)
            
            logger.info(
                "User registered successfully",
                user_id=str(user.id),
                email=user_data.email,
                ip_address=ip_address
            )
            
            return user
            
        except Exception as e:
            await db.rollback()
            if isinstance(e, AuthenticationError):
                raise
            
            logger.error("User registration failed", error=str(e))
            raise AuthenticationError("Registration failed")
    
    async def authenticate_user(
        self,
        email: str,
        password: str,
        db: AsyncSession,
        ip_address: str = None,
        user_agent: str = None,
        device_info: Dict[str, Any] = None
    ) -> Tuple[User, UserSession]:
        """
        Authenticate user and create session.
        
        Implements RULE AUTH-002 with JWT token management.
        
        Args:
            email: User email
            password: User password
            db: Database session
            ip_address: Client IP address
            user_agent: Client user agent
            device_info: Device information
            
        Returns:
            Tuple[User, UserSession]: User and session objects
            
        Raises:
            AuthenticationError: If authentication fails
        """
        user = await self._get_user_by_email(email, db)
        
        if not user:
            raise AuthenticationError("Invalid credentials")
        
        # Check if account is locked
        if user.is_account_locked():
            raise AuthenticationError("Account is temporarily locked")
        
        # Check if account is active
        if not user.is_active:
            raise AuthenticationError("Account is inactive")
        
        # Verify password
        if not user.verify_password(password):
            user.increment_failed_login()
            await db.commit()
            raise AuthenticationError("Invalid credentials")
        
        # Reset failed login attempts
        user.reset_failed_login_attempts()
        
        # Create session (RULE AUTH-002)
        session = await self._create_user_session(
            user=user,
            db=db,
            ip_address=ip_address,
            user_agent=user_agent,
            device_info=device_info
        )
        
        await db.commit()
        
        logger.info(
            "User authenticated successfully",
            user_id=str(user.id),
            session_id=str(session.id),
            ip_address=ip_address
        )
        
        return user, session
    
    async def create_tokens(self, user: User, session: UserSession) -> Dict[str, Any]:
        """
        Create JWT access and refresh tokens.
        
        Implements RULE AUTH-002 with proper token structure.
        
        Args:
            user: User object
            session: Session object
            
        Returns:
            Dict containing tokens and metadata
        """
        now = datetime.now(timezone.utc)
        
        # Access token payload
        access_payload = {
            "sub": str(user.id),
            "email": user.email,
            "session_id": str(session.id),
            "jti": session.access_token_jti,
            "type": "access",
            "iat": now,
            "exp": session.access_token_expires,
            "aud": "ai-code-mentor",
            "iss": "ai-code-mentor-auth"
        }
        
        # Refresh token payload
        refresh_payload = {
            "sub": str(user.id),
            "session_id": str(session.id),
            "jti": session.refresh_token_jti,
            "type": "refresh",
            "iat": now,
            "exp": session.refresh_token_expires,
            "aud": "ai-code-mentor",
            "iss": "ai-code-mentor-auth"
        }
        
        # Generate tokens
        access_token = jwt.encode(access_payload, self.secret_key, algorithm=self.algorithm)
        refresh_token = jwt.encode(refresh_payload, self.secret_key, algorithm=self.algorithm)
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": self.access_token_expire_minutes * 60,
            "session_id": str(session.id),
            "user_id": str(user.id)
        }
    
    async def refresh_token(
        self,
        refresh_token: str,
        db: AsyncSession,
        ip_address: str = None
    ) -> Dict[str, Any]:
        """
        Refresh access token using refresh token.
        
        Implements RULE AUTH-002 with token rotation.
        
        Args:
            refresh_token: Refresh token
            db: Database session
            ip_address: Client IP address
            
        Returns:
            Dict containing new tokens
            
        Raises:
            AuthenticationError: If refresh fails
        """
        try:
            # Decode refresh token
            payload = jwt.decode(
                refresh_token,
                self.secret_key,
                algorithms=[self.algorithm],
                audience="ai-code-mentor"
            )
            
            if payload.get("type") != "refresh":
                raise AuthenticationError("Invalid token type")
            
            session_id = payload.get("session_id")
            jti = payload.get("jti")
            
            # Get session
            session = await self._get_session_by_id(session_id, db)
            
            if not session or session.refresh_token_jti != jti:
                raise AuthenticationError("Invalid refresh token")
            
            if not session.is_active():
                raise AuthenticationError("Session expired or revoked")
            
            # Get user
            user = await self._get_user_by_id(session.user_id, db)
            if not user or not user.is_active:
                raise AuthenticationError("User not found or inactive")
            
            # Refresh access token
            session.refresh_access_token()
            await db.commit()
            
            # Create new tokens
            tokens = await self.create_tokens(user, session)
            
            logger.info(
                "Tokens refreshed successfully",
                user_id=str(user.id),
                session_id=str(session.id),
                ip_address=ip_address
            )
            
            return tokens
            
        except jwt.InvalidTokenError as e:
            logger.warning("Invalid refresh token", error=str(e), ip_address=ip_address)
            raise AuthenticationError("Invalid refresh token")
        except Exception as e:
            logger.error("Token refresh failed", error=str(e))
            raise AuthenticationError("Token refresh failed")
    
    async def logout_user(
        self,
        session_id: str,
        db: AsyncSession,
        revoke_all: bool = False
    ) -> None:
        """
        Logout user and revoke session.
        
        Implements RULE AUTH-004 with proper session invalidation.
        
        Args:
            session_id: Session ID to revoke
            db: Database session
            revoke_all: Revoke all user sessions
        """
        session = await self._get_session_by_id(session_id, db)
        
        if not session:
            raise AuthenticationError("Session not found")
        
        if revoke_all:
            # Revoke all user sessions
            await self._revoke_all_user_sessions(session.user_id, db)
        else:
            # Revoke specific session
            session.revoke(reason="user_logout")
        
        await db.commit()
        
        logger.info(
            "User logged out",
            user_id=str(session.user_id),
            session_id=session_id,
            revoke_all=revoke_all
        )
    
    async def verify_token(
        self,
        token: str,
        db: AsyncSession,
        token_type: str = "access"
    ) -> Tuple[User, UserSession]:
        """
        Verify JWT token and return user and session.
        
        Args:
            token: JWT token to verify
            db: Database session
            token_type: Type of token (access, refresh)
            
        Returns:
            Tuple[User, UserSession]: User and session objects
            
        Raises:
            AuthenticationError: If token is invalid
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                audience="ai-code-mentor"
            )
            
            if payload.get("type") != token_type:
                raise AuthenticationError(f"Invalid token type, expected {token_type}")
            
            session_id = payload.get("session_id")
            jti = payload.get("jti")
            user_id = payload.get("sub")
            
            # Get session
            session = await self._get_session_by_id(session_id, db)
            
            if not session:
                raise AuthenticationError("Session not found")
            
            # Verify JTI
            if token_type == "access" and session.access_token_jti != jti:
                raise AuthenticationError("Invalid token JTI")
            elif token_type == "refresh" and session.refresh_token_jti != jti:
                raise AuthenticationError("Invalid token JTI")
            
            # Check session validity
            if token_type == "access" and not session.is_access_token_valid():
                raise AuthenticationError("Access token expired")
            elif token_type == "refresh" and not session.is_active():
                raise AuthenticationError("Refresh token expired")
            
            # Get user
            user = await self._get_user_by_id(user_id, db)
            if not user or not user.is_active:
                raise AuthenticationError("User not found or inactive")
            
            # Update session activity
            session.update_activity()
            await db.commit()
            
            return user, session
            
        except jwt.InvalidTokenError as e:
            logger.warning("Invalid token", error=str(e))
            raise AuthenticationError("Invalid token")
    
    # Private helper methods
    async def _get_user_by_email(self, email: str, db: AsyncSession) -> Optional[User]:
        """Get user by email address."""
        result = await db.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()
    
    async def _get_user_by_id(self, user_id: str, db: AsyncSession) -> Optional[User]:
        """Get user by ID."""
        result = await db.execute(
            select(User).where(User.id == uuid.UUID(user_id))
        )
        return result.scalar_one_or_none()
    
    async def _get_session_by_id(self, session_id: str, db: AsyncSession) -> Optional[UserSession]:
        """Get session by ID."""
        result = await db.execute(
            select(UserSession).where(UserSession.id == uuid.UUID(session_id))
        )
        return result.scalar_one_or_none()
    
    async def _create_user_session(
        self,
        user: User,
        db: AsyncSession,
        ip_address: str = None,
        user_agent: str = None,
        device_info: Dict[str, Any] = None
    ) -> UserSession:
        """Create new user session."""
        session = UserSession(
            user_id=user.id,
            access_token_jti=str(uuid.uuid4()),
            refresh_token_jti=str(uuid.uuid4()),
            ip_address=ip_address,
            user_agent=user_agent,
            device_info=device_info or {}
        )
        
        db.add(session)
        await db.flush()  # Get session ID
        
        return session
    
    async def _revoke_all_user_sessions(self, user_id: uuid.UUID, db: AsyncSession) -> None:
        """Revoke all sessions for a user."""
        result = await db.execute(
            select(UserSession).where(
                and_(
                    UserSession.user_id == user_id,
                    UserSession.status == SessionStatus.ACTIVE
                )
            )
        )
        sessions = result.scalars().all()
        
        for session in sessions:
            session.revoke(reason="logout_all")


# Global auth service instance
auth_service = AuthService()