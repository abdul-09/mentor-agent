"""
File Model for AI Code Mentor
Handles PDF uploads and file metadata with security and compliance.

Compliance:
- RULE SEC-001: File validation and security scanning
- RULE DB-001: 3NF normalization
- RULE DB-002: UUID primary keys with proper indexing
- RULE PRIVACY-001: Data retention and GDPR compliance
"""

import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from enum import Enum

import structlog
from sqlalchemy import (
    Column, String, Boolean, DateTime, Text, Integer, 
    BigInteger, ForeignKey, Enum as SQLEnum,
    Index, CheckConstraint
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from pydantic import BaseModel, Field

from src.models.database import Base
from src.config.settings import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class FileStatus(str, Enum):
    """File processing status enumeration."""
    UPLOADING = "uploading"
    UPLOADED = "uploaded" 
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    QUARANTINED = "quarantined"  # Security scan failed
    DELETED = "deleted"


class FileType(str, Enum):
    """Supported file types."""
    PDF = "pdf"
    DOCUMENT = "document"
    CODE_ARCHIVE = "code_archive"


class SecurityScanStatus(str, Enum):
    """Security scan status enumeration."""
    PENDING = "pending"
    CLEAN = "clean"
    INFECTED = "infected"
    ERROR = "error"
    SKIPPED = "skipped"


class File(Base):
    """
    File model for document and PDF management.
    
    Implements secure file handling with metadata tracking.
    """
    __tablename__ = "files"
    
    # Primary key (RULE DB-002: UUID primary keys)
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
        comment="Unique file identifier"
    )
    
    # Owner relationship
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="File owner user ID"
    )
    
    # File metadata
    original_filename = Column(
        String(255),
        nullable=False,
        comment="Original filename as uploaded"
    )
    
    stored_filename = Column(
        String(255),
        nullable=False,
        unique=True,
        comment="Internal storage filename (UUID-based)"
    )
    
    file_type = Column(
        SQLEnum(FileType),
        nullable=False,
        index=True,
        comment="File type classification"
    )
    
    mime_type = Column(
        String(100),
        nullable=False,
        comment="MIME type from file headers"
    )
    
    file_size = Column(
        BigInteger,
        nullable=False,
        comment="File size in bytes"
    )
    
    file_hash = Column(
        String(64),
        nullable=False,
        index=True,
        comment="SHA-256 hash for deduplication and integrity"
    )
    
    # Processing status
    status = Column(
        SQLEnum(FileStatus),
        default=FileStatus.UPLOADING,
        nullable=False,
        index=True,
        comment="File processing status"
    )
    
    # Security scanning (RULE SEC-001)
    security_scan_status = Column(
        SQLEnum(SecurityScanStatus),
        default=SecurityScanStatus.PENDING,
        nullable=False,
        comment="Security scan result"
    )
    
    security_scan_details = Column(
        JSONB,
        nullable=True,
        comment="Detailed security scan results"
    )
    
    # Content processing
    extracted_text = Column(
        Text,
        nullable=True,
        comment="Extracted text content (searchable)"
    )
    
    text_extraction_status = Column(
        String(20),
        default="pending",
        nullable=False,
        comment="Text extraction status"
    )
    
    page_count = Column(
        Integer,
        nullable=True,
        comment="Number of pages (for PDFs)"
    )
    
    # AI processing metadata
    embedding_status = Column(
        String(20),
        default="pending",
        nullable=False,
        comment="Embedding generation status"
    )
    
    chunk_count = Column(
        Integer,
        nullable=True,
        comment="Number of text chunks created"
    )
    
    processing_metadata = Column(
        JSONB,
        nullable=True,
        comment="Processing configuration and results"
    )
    
    # Storage information
    storage_path = Column(
        String(500),
        nullable=False,
        comment="Storage path or cloud storage key"
    )
    
    storage_backend = Column(
        String(20),
        default="local",
        nullable=False,
        comment="Storage backend: local, s3, gcs"
    )
    
    # Access control
    is_public = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="Public access allowed"
    )
    
    access_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Number of times file was accessed"
    )
    
    # Data retention (RULE PRIVACY-001)
    retention_policy = Column(
        String(20),
        default="standard",
        nullable=False,
        comment="Data retention policy: standard, extended, permanent"
    )
    
    expires_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="File expiration timestamp"
    )
    
    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        comment="File upload timestamp"
    )
    
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
        comment="Last update timestamp"
    )
    
    processed_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Processing completion timestamp"
    )
    
    last_accessed_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Last access timestamp"
    )
    
    # Relationships
    user = relationship("User", back_populates="files")
    analyses = relationship("AnalysisSession", back_populates="file", cascade="all, delete-orphan")
    
    # Indexes for performance
    __table_args__ = (
        Index("ix_files_user_status", "user_id", "status"),
        Index("ix_files_user_created", "user_id", "created_at"),
        Index("ix_files_type_status", "file_type", "status"),
        Index("ix_files_hash_size", "file_hash", "file_size"),
        Index("ix_files_expires", "expires_at"),
        CheckConstraint("file_size > 0", name="ck_files_size_positive"),
        CheckConstraint("access_count >= 0", name="ck_files_access_count"),
        CheckConstraint("page_count >= 0", name="ck_files_page_count"),
        CheckConstraint("chunk_count >= 0", name="ck_files_chunk_count"),
        {"comment": "File storage and metadata tracking"}
    )
    
    def mark_uploaded(self) -> None:
        """Mark file as successfully uploaded."""
        self.status = FileStatus.UPLOADED
        logger.info("File marked as uploaded", file_id=str(self.id))
    
    def start_processing(self) -> None:
        """Mark file as processing started."""
        self.status = FileStatus.PROCESSING
        logger.info("File processing started", file_id=str(self.id))
    
    def mark_processed(self) -> None:
        """Mark file as successfully processed."""
        self.status = FileStatus.PROCESSED
        self.processed_at = datetime.now(timezone.utc)
        logger.info("File processing completed", file_id=str(self.id))
    
    def mark_failed(self, error_message: str = None) -> None:
        """Mark file processing as failed."""
        self.status = FileStatus.FAILED
        if error_message and self.processing_metadata:
            self.processing_metadata.setdefault("errors", []).append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": error_message
            })
        logger.error("File processing failed", file_id=str(self.id), error=error_message)
    
    def quarantine(self, reason: str) -> None:
        """Quarantine file due to security issues."""
        self.status = FileStatus.QUARANTINED
        self.security_scan_status = SecurityScanStatus.INFECTED
        if not self.security_scan_details:
            self.security_scan_details = {}
        self.security_scan_details["quarantine_reason"] = reason
        self.security_scan_details["quarantined_at"] = datetime.now(timezone.utc).isoformat()
        
        logger.warning("File quarantined", file_id=str(self.id), reason=reason)
    
    def mark_clean(self) -> None:
        """Mark file as clean after security scan."""
        self.security_scan_status = SecurityScanStatus.CLEAN
        logger.info("File marked as clean", file_id=str(self.id))
    
    def record_access(self) -> None:
        """Record file access for analytics."""
        self.access_count += 1
        self.last_accessed_at = datetime.now(timezone.utc)
    
    def is_expired(self) -> bool:
        """Check if file has expired based on retention policy."""
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) > self.expires_at
    
    def can_be_deleted(self) -> bool:
        """Check if file can be safely deleted."""
        return (
            self.status in [FileStatus.FAILED, FileStatus.QUARANTINED] or
            self.is_expired()
        )
    
    def get_file_size_mb(self) -> float:
        """Get file size in megabytes."""
        return round(self.file_size / (1024 * 1024), 2)
    
    def __repr__(self) -> str:
        return f"<File(id={self.id}, filename={self.original_filename}, status={self.status})>"


# Pydantic models for API
class FileUploadResponse(BaseModel):
    """File upload response model."""
    id: str
    filename: str
    file_type: str
    file_size: int
    status: str
    
    class Config:
        from_attributes = True


class FileResponse(BaseModel):
    """File information response model."""
    id: str
    original_filename: str
    file_type: str
    mime_type: str
    file_size: int
    status: str
    security_scan_status: str
    is_public: bool
    page_count: Optional[int]
    chunk_count: Optional[int]
    access_count: int
    created_at: datetime
    processed_at: Optional[datetime]
    
    class Config:
        from_attributes = True


class FileSummary(BaseModel):
    """File summary for listings."""
    id: str
    original_filename: str
    file_type: str
    file_size: int
    status: str
    created_at: datetime
    
    class Config:
        from_attributes = True