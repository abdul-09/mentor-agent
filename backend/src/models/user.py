"""
User Model for AI Code Mentor
Implements production-grade user management with security compliance.

Compliance:
- RULE AUTH-001: bcrypt password hashing (cost factor 12+)
- RULE AUTH-003: MFA support for sensitive operations
- RULE DB-001: 3NF normalization
- RULE DB-002: UUID primary keys with proper indexing
- RULE PRIVACY-001: Personal data handling with GDPR compliance
"""

import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional

import bcrypt
import structlog
from sqlalchemy import (
    Column, String, Boolean, DateTime, Text, Integer,
    Index, CheckConstraint
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from pydantic import BaseModel, EmailStr, Field

from src.models.database import Base
from src.config.settings import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class User(Base):
    """
    User model with production-grade security features.
    
    Implements RULE AUTH-001 with secure password hashing and
    RULE AUTH-003 with MFA support.
    """
    __tablename__ = "users"
    
    # Primary key (RULE DB-002: UUID primary keys)
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
        comment="Unique user identifier"
    )
    
    # Authentication fields
    email = Column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
        comment="User email address (unique)"
    )
    
    password_hash = Column(
        String(60),  # bcrypt hash length
        nullable=False,
        comment="bcrypt password hash (cost factor 12+)"
    )
    
    # User profile
    full_name = Column(
        String(100),
        nullable=False,
        comment="User's full name"
    )
    
    # Account status
    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
        comment="Account active status"
    )
    
    is_verified = Column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        comment="Email verification status"
    )
    
    is_admin = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="Admin privileges"
    )
    
    # MFA fields (RULE AUTH-003)
    mfa_enabled = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="Multi-factor authentication enabled"
    )
    
    mfa_secret = Column(
        String(32),
        nullable=True,
        comment="TOTP secret key (encrypted)"
    )
    
    mfa_backup_codes = Column(
        JSONB,
        nullable=True,
        comment="Encrypted MFA backup codes"
    )
    
    # Security tracking
    failed_login_attempts = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Failed login attempt counter"
    )
    
    account_locked_until = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Account lockout expiration"
    )
    
    last_login_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Last successful login timestamp"
    )
    
    last_password_change = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Last password change timestamp"
    )
    
    # Password reset fields
    password_reset_token = Column(
        String(255),
        nullable=True,
        comment="Password reset token"
    )
    
    password_reset_expires = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Password reset token expiration"
    )
    
    # Timestamps (RULE PRIVACY-002: Audit trail)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        comment="Account creation timestamp"
    )
    
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
        comment="Last update timestamp"
    )
    
    # Relationships
    user_sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    files = relationship("File", back_populates="user", cascade="all, delete-orphan")
    analyses = relationship("AnalysisSession", back_populates="user", cascade="all, delete-orphan")
    
    # Indexes for performance
    __table_args__ = (
        Index("ix_users_email_active", "email", "is_active"),
        Index("ix_users_created_verified", "created_at", "is_verified"),
        CheckConstraint("failed_login_attempts >= 0", name="ck_users_failed_attempts"),
        CheckConstraint("length(email) > 0", name="ck_users_email_not_empty"),
        CheckConstraint("length(full_name) > 0", name="ck_users_name_not_empty"),
        {"comment": "Users table with security and audit features"}
    )
    
    def set_password(self, password: str) -> None:
        """
        Set user password with bcrypt hashing.
        
        Implements RULE AUTH-001 with cost factor 12+.
        
        Args:
            password: Plain text password to hash
        """
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters long")
        
        # Use cost factor 12+ for production security (RULE AUTH-001)
        cost_factor = max(12, settings.BCRYPT_ROUNDS if hasattr(settings, 'BCRYPT_ROUNDS') else 12)
        
        salt = bcrypt.gensalt(rounds=cost_factor)
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        self.password_hash = hashed.decode('utf-8')
        self.last_password_change = datetime.now(timezone.utc)
        
        logger.info(
            "Password updated",
            user_id=str(self.id),
            cost_factor=cost_factor
        )
    
    def verify_password(self, password: str) -> bool:
        """
        Verify password against stored hash.
        
        Args:
            password: Plain text password to verify
            
        Returns:
            bool: True if password matches
        """
        if not self.password_hash:
            return False
        
        return bcrypt.checkpw(
            password.encode('utf-8'),
            self.password_hash.encode('utf-8')
        )
    
    def increment_failed_login(self) -> None:
        """Increment failed login attempts and lock account if needed."""
        self.failed_login_attempts += 1
        
        # Lock account after 5 failed attempts for 15 minutes
        if self.failed_login_attempts >= 5:
            self.account_locked_until = datetime.now(timezone.utc) + timedelta(minutes=15)
            
            logger.warning(
                "Account locked due to failed login attempts",
                user_id=str(self.id),
                failed_attempts=self.failed_login_attempts
            )
    
    def reset_failed_login_attempts(self) -> None:
        """Reset failed login attempts and unlock account."""
        self.failed_login_attempts = 0
        self.account_locked_until = None
        self.last_login_at = datetime.now(timezone.utc)
    
    def is_account_locked(self) -> bool:
        """
        Check if account is currently locked.
        
        Returns:
            bool: True if account is locked
        """
        if not self.account_locked_until:
            return False
        
        # Check if lock has expired
        if datetime.now(timezone.utc) > self.account_locked_until:
            self.account_locked_until = None
            self.failed_login_attempts = 0
            return False
        
        return True
    
    def can_enable_mfa(self) -> bool:
        """Check if user can enable MFA (verified email required)."""
        return self.is_verified and self.is_active
    
    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email}, active={self.is_active})>"


# Pydantic models for API
class UserCreate(BaseModel):
    """User creation request model."""
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=8, description="Password (min 8 characters)")
    full_name: str = Field(..., min_length=2, max_length=100, description="Full name")


class UserUpdate(BaseModel):
    """User update request model."""
    full_name: Optional[str] = Field(None, min_length=2, max_length=100)
    email: Optional[EmailStr] = None


class UserResponse(BaseModel):
    """User response model."""
    id: str
    email: str
    full_name: str
    is_active: bool
    is_verified: bool
    is_admin: bool
    mfa_enabled: bool
    created_at: datetime
    last_login_at: Optional[datetime]
    
    class Config:
        from_attributes = True


class UserSummary(BaseModel):
    """Minimal user information for listings."""
    id: str
    email: str
    full_name: str
    is_active: bool
    
    class Config:
        from_attributes = True