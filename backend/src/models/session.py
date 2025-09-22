"""
Session Model for AI Code Mentor
Handles user sessions, JWT tokens, and authentication state management.

Compliance:
- RULE AUTH-002: JWT token management (15min access, 7day refresh)
- RULE AUTH-004: Session management security
- RULE DB-002: UUID primary keys with proper indexing
- RULE PRIVACY-002: Audit trails for authentication events
"""

import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any
from enum import Enum

import structlog
from sqlalchemy import (
    Column, String, Boolean, DateTime, Text, Integer,
    ForeignKey, Enum as SQLEnum,
    Index, CheckConstraint
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from pydantic import BaseModel

from src.models.database import Base
from src.config.settings import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class SessionStatus(str, Enum):
    """Session status enumeration."""
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    INVALIDATED = "invalidated"


class UserSession(Base):
    """
    User session model for JWT token management.
    
    Implements RULE AUTH-002 with proper token lifecycle management
    and RULE AUTH-004 with session security features.
    """
    __tablename__ = "user_sessions"
    
    # Primary key
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
        comment="Unique session identifier"
    )
    
    # User relationship
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Session owner user ID"
    )
    
    # Token information
    access_token_jti = Column(
        String(36),
        unique=True,
        nullable=False,
        comment="Access token unique identifier (JTI)"
    )
    
    refresh_token_jti = Column(
        String(36),
        unique=True,
        nullable=False,
        comment="Refresh token unique identifier (JTI)"
    )
    
    # Token expiration (RULE AUTH-002)
    access_token_expires = Column(
        DateTime(timezone=True),
        nullable=False,
        comment="Access token expiration (15 minutes max)"
    )
    
    refresh_token_expires = Column(
        DateTime(timezone=True),
        nullable=False,
        comment="Refresh token expiration (7 days max)"
    )
    
    # Session metadata
    status = Column(
        SQLEnum(SessionStatus),
        default=SessionStatus.ACTIVE,
        nullable=False,
        index=True,
        comment="Session status"
    )
    
    device_info = Column(
        JSONB,
        nullable=True,
        comment="Device and browser information"
    )
    
    ip_address = Column(
        String(45),  # IPv6 support
        nullable=True,
        comment="Client IP address"
    )
    
    user_agent = Column(
        Text,
        nullable=True,
        comment="Client user agent string"
    )
    
    location = Column(
        JSONB,
        nullable=True,
        comment="Approximate geographic location"
    )
    
    # Security features
    is_trusted_device = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="Trusted device status"
    )
    
    mfa_verified = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="MFA verification status for this session"
    )
    
    security_level = Column(
        String(20),
        default="standard",
        nullable=False,
        comment="Session security level: standard, elevated, admin"
    )
    
    # Activity tracking
    last_activity_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        comment="Last session activity timestamp"
    )
    
    activity_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Number of requests in this session"
    )
    
    # Revocation information
    revoked_reason = Column(
        String(100),
        nullable=True,
        comment="Reason for session revocation"
    )
    
    revoked_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Session revocation timestamp"
    )
    
    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        comment="Session creation timestamp"
    )
    
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
        comment="Last update timestamp"
    )
    
    # Relationships
    user = relationship("User", back_populates="user_sessions")
    
    # Indexes for performance
    __table_args__ = (
        Index("ix_sessions_user_status", "user_id", "status"),
        Index("ix_sessions_user_created", "user_id", "created_at"),
        Index("ix_sessions_access_expires", "access_token_expires"),
        Index("ix_sessions_refresh_expires", "refresh_token_expires"),
        Index("ix_sessions_ip_created", "ip_address", "created_at"),
        Index("ix_sessions_last_activity", "last_activity_at"),
        CheckConstraint("activity_count >= 0", name="ck_sessions_activity_count"),
        {"comment": "User authentication sessions with security tracking"}
    )
    
    def __init__(self, **kwargs):
        """Initialize session with token expiration times."""
        super().__init__(**kwargs)
        
        now = datetime.now(timezone.utc)
        
        # Set token expiration times (RULE AUTH-002)
        if not self.access_token_expires:
            self.access_token_expires = now + timedelta(
                minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
            )
        
        if not self.refresh_token_expires:
            self.refresh_token_expires = now + timedelta(
                days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS
            )
    
    def is_active(self) -> bool:
        """
        Check if session is active and not expired.
        
        Returns:
            bool: True if session is active
        """
        if self.status != SessionStatus.ACTIVE:
            return False
        
        now = datetime.now(timezone.utc)
        return now < self.refresh_token_expires
    
    def is_access_token_valid(self) -> bool:
        """
        Check if access token is still valid.
        
        Returns:
            bool: True if access token is valid
        """
        if not self.is_active():
            return False
        
        now = datetime.now(timezone.utc)
        return now < self.access_token_expires
    
    def refresh_access_token(self) -> None:
        """Refresh the access token with new expiration and JTI."""
        now = datetime.now(timezone.utc)
        
        # Generate new access token JTI
        self.access_token_jti = str(uuid.uuid4())
        
        # Set new expiration
        self.access_token_expires = now + timedelta(
            minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
        )
        
        # Update activity
        self.update_activity()
        
        logger.info(
            "Access token refreshed",
            session_id=str(self.id),
            user_id=str(self.user_id),
            new_expiration=self.access_token_expires.isoformat()
        )
    
    def update_activity(self) -> None:
        """Update session activity tracking."""
        self.last_activity_at = datetime.now(timezone.utc)
        self.activity_count += 1
    
    def revoke(self, reason: str = "user_logout") -> None:
        """
        Revoke the session.
        
        Args:
            reason: Reason for revocation
        """
        self.status = SessionStatus.REVOKED
        self.revoked_reason = reason
        self.revoked_at = datetime.now(timezone.utc)
        
        logger.info(
            "Session revoked",
            session_id=str(self.id),
            user_id=str(self.user_id),
            reason=reason
        )
    
    def mark_expired(self) -> None:
        """Mark session as expired."""
        self.status = SessionStatus.EXPIRED
        
        logger.info(
            "Session marked as expired",
            session_id=str(self.id),
            user_id=str(self.user_id)
        )
    
    def set_mfa_verified(self) -> None:
        """Mark session as MFA verified."""
        self.mfa_verified = True
        self.update_activity()
        
        logger.info(
            "Session MFA verified",
            session_id=str(self.id),
            user_id=str(self.user_id)
        )
    
    def elevate_security_level(self, level: str = "elevated") -> None:
        """
        Elevate session security level.
        
        Args:
            level: Security level (elevated, admin)
        """
        valid_levels = ["standard", "elevated", "admin"]
        if level not in valid_levels:
            raise ValueError(f"Invalid security level. Must be one of: {valid_levels}")
        
        self.security_level = level
        self.update_activity()
        
        logger.info(
            "Session security level elevated",
            session_id=str(self.id),
            user_id=str(self.user_id),
            new_level=level
        )
    
    def __repr__(self) -> str:
        return f"<UserSession(id={self.id}, user_id={self.user_id}, status={self.status})>"


# Pydantic models for API
class SessionResponse(BaseModel):
    """Session response model."""
    id: str
    status: str
    ip_address: Optional[str]
    user_agent: Optional[str]
    is_trusted_device: bool
    mfa_verified: bool
    security_level: str
    last_activity_at: datetime
    created_at: datetime
    
    class Config:
        from_attributes = True


class SessionSummary(BaseModel):
    """Session summary for listings."""
    id: str
    ip_address: Optional[str]
    last_activity_at: datetime
    is_current: bool = False
    
    class Config:
        from_attributes = True