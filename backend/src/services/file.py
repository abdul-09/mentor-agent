"""
File Management Service for AI Code Mentor
Handles PDF upload, storage, validation, and management with security compliance.

Compliance:
- RULE SEC-001: Input validation and file security
- RULE PERF-004: File upload performance optimization
- RULE LOG-001: Structured logging with trace IDs
"""

import asyncio
import hashlib
import mimetypes
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Tuple

import aiofiles
import structlog
from fastapi import HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy.sql import select

from src.config.settings import get_settings
from src.models.file import File, FileType, FileStatus
from src.models.user import User

logger = structlog.get_logger(__name__)
settings = get_settings()


class FileValidationError(Exception):
    """Custom exception for file validation errors."""
    pass


class FileStorageError(Exception):
    """Custom exception for file storage errors."""
    pass


class FileService:
    """Service class for file management operations."""

    def __init__(self):
        self.max_file_size = settings.max_file_size_bytes
        self.allowed_types = settings.ALLOWED_FILE_TYPES
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def _detect_mime_type(self, content: bytes, filename: str) -> str:
        """
        Detect MIME type using file content and filename.
        
        Args:
            content: File content bytes
            filename: Original filename
            
        Returns:
            MIME type string
        """
        # First, try to detect by file extension
        guessed_type, _ = mimetypes.guess_type(filename)
        
        # For PDF files, check the magic bytes
        if content.startswith(b"%PDF-"):
            return "application/pdf"
        
        # Check other common file signatures
        if content.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        elif content.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        elif content.startswith(b"GIF87a") or content.startswith(b"GIF89a"):
            return "image/gif"
        elif content.startswith(b"PK\x03\x04") or content.startswith(b"PK\x05\x06") or content.startswith(b"PK\x07\x08"):
            # ZIP-based formats
            if filename.lower().endswith(('.docx', '.xlsx', '.pptx')):
                return "application/vnd.openxmlformats-officedocument"
            return "application/zip"
        elif content.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
            # Microsoft Office legacy formats
            return "application/msword"
        
        # Fall back to guessed type from filename
        if guessed_type:
            return guessed_type
        
        # Default to binary if we can't determine
        return "application/octet-stream"

    async def validate_file(self, file: UploadFile) -> Dict[str, any]:
        """
        Validate uploaded file for security and compliance.
        
        Args:
            file: The uploaded file object
            
        Returns:
            Dict containing validation results
            
        Raises:
            FileValidationError: If file validation fails
        """
        try:
            # Check file size
            file.file.seek(0, 2)  # Seek to end
            file_size = file.file.tell()
            file.file.seek(0)  # Reset position
            
            if file_size > self.max_file_size:
                raise FileValidationError(
                    f"File size {file_size} bytes exceeds maximum allowed size "
                    f"{self.max_file_size} bytes"
                )
            
            if file_size == 0:
                raise FileValidationError("File is empty")
            
            # Read file content for validation
            content = await file.read()
            file.file.seek(0)  # Reset position
            
            # Detect MIME type using content analysis
            mime_type = self._detect_mime_type(content, file.filename)
            
            # Validate against allowed types
            if mime_type not in self.allowed_types:
                # Also check by file extension as fallback
                guessed_type, _ = mimetypes.guess_type(file.filename)
                if guessed_type not in self.allowed_types:
                    raise FileValidationError(
                        f"File type '{mime_type}' not allowed. "
                        f"Allowed types: {', '.join(self.allowed_types)}"
                    )
            
            # Additional security checks for PDF files
            if mime_type == "application/pdf":
                if not content.startswith(b"%PDF-"):
                    raise FileValidationError("Invalid PDF file format")
            
            # Calculate file hash for deduplication
            file_hash = hashlib.sha256(content).hexdigest()
            
            logger.info(
                "File validation completed",
                filename=file.filename,
                size=file_size,
                mime_type=mime_type,
                hash=file_hash[:16] + "..."
            )
            
            return {
                "filename": file.filename,
                "size": file_size,
                "mime_type": mime_type,
                "content": content,
                "hash": file_hash
            }
            
        except Exception as e:
            logger.error(
                "File validation failed",
                filename=getattr(file, 'filename', 'unknown'),
                error=str(e)
            )
            if isinstance(e, FileValidationError):
                raise
            raise FileValidationError(f"File validation failed: {str(e)}")

    async def store_file(
        self, 
        file_data: Dict[str, any], 
        user_id: uuid.UUID
    ) -> Tuple[str, str]:
        """
        Store file to disk with secure naming.
        
        Args:
            file_data: Validated file data
            user_id: ID of the user uploading the file
            
        Returns:
            Tuple of (file_id, storage_path, secure_filename)
            
        Raises:
            FileStorageError: If file storage fails
        """
        try:
            # Generate unique file ID and secure filename
            file_id = str(uuid.uuid4())
            file_extension = Path(file_data["filename"]).suffix.lower()
            secure_filename = f"{file_id}{file_extension}"
            
            # Create user-specific directory
            user_dir = self.upload_dir / str(user_id)
            user_dir.mkdir(parents=True, exist_ok=True)
            
            # Full storage path
            storage_path = user_dir / secure_filename
            
            # Write file to disk asynchronously
            async with aiofiles.open(storage_path, 'wb') as f:
                await f.write(file_data["content"])
            
            # Verify file was written correctly
            if not storage_path.exists():
                raise FileStorageError("File was not written to disk")
            
            stored_size = storage_path.stat().st_size
            if stored_size != file_data["size"]:
                storage_path.unlink()  # Remove corrupted file
                raise FileStorageError(
                    f"File size mismatch: expected {file_data['size']}, "
                    f"got {stored_size}"
                )
            
            logger.info(
                "File stored successfully",
                file_id=file_id,
                storage_path=str(storage_path),
                user_id=str(user_id)
            )
            
            return file_id, str(storage_path), secure_filename
            
        except Exception as e:
            logger.error(
                "File storage failed",
                user_id=str(user_id),
                error=str(e)
            )
            if isinstance(e, FileStorageError):
                raise
            raise FileStorageError(f"File storage failed: {str(e)}")

    async def upload_file(
        self, 
        file: UploadFile, 
        user: User, 
        db: AsyncSession
    ) -> File:
        """
        Complete file upload process with validation and storage.
        
        Args:
            file: The uploaded file
            user: The user uploading the file
            db: Database session
            
        Returns:
            File: Created file record
            
        Raises:
            HTTPException: If upload fails
        """
        try:
            # Validate file
            file_data = await self.validate_file(file)
            
            # Check for duplicate files (optional deduplication)
            existing_file = await self.find_duplicate_file(
                file_data["hash"], user.id, db
            )
            if existing_file:
                logger.info(
                    "Duplicate file detected",
                    existing_file_id=str(existing_file.id),
                    user_id=str(user.id)
                )
                return existing_file
            
            # Store file to disk
            file_id, storage_path, secure_filename = await self.store_file(file_data, user.id)
            
            # Create database record
            file_record = File(
                id=uuid.UUID(file_id),
                user_id=user.id,
                original_filename=file_data["filename"],
                stored_filename=secure_filename,
                file_type=FileType.PDF,  # Currently only supporting PDF
                file_size=file_data["size"],
                mime_type=file_data["mime_type"],
                file_hash=file_data["hash"],
                storage_path=storage_path,
                status=FileStatus.UPLOADED,
                processing_metadata={
                    "upload_timestamp": datetime.now(timezone.utc).isoformat(),
                    "upload_ip": "127.0.0.1",  # TODO: Get from request
                    "user_agent": "Unknown"    # TODO: Get from request
                }
            )
            
            db.add(file_record)
            await db.commit()
            await db.refresh(file_record)
            
            logger.info(
                "File upload completed",
                file_id=file_id,
                user_id=str(user.id),
                filename=file_data["filename"]
            )
            
            return file_record
            
        except (FileValidationError, FileStorageError) as e:
            logger.error(
                "File upload failed",
                user_id=str(user.id),
                filename=getattr(file, 'filename', 'unknown'),
                error=str(e)
            )
            await db.rollback()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        except Exception as e:
            logger.error(
                "Unexpected error during file upload",
                user_id=str(user.id),
                error=str(e)
            )
            await db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="File upload failed"
            )

    async def find_duplicate_file(
        self, 
        file_hash: str, 
        user_id: uuid.UUID, 
        db: AsyncSession
    ) -> Optional[File]:
        """
        Find duplicate file by hash for the same user.
        
        Args:
            file_hash: SHA256 hash of the file
            user_id: User ID
            db: Database session
            
        Returns:
            File record if duplicate found, None otherwise
        """
        try:
            result = await db.execute(
                select(File)
                .where(
                    File.file_hash == file_hash,
                    File.user_id == user_id,
                    File.status != FileStatus.DELETED
                )
                .limit(1)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(
                "Error checking for duplicate files",
                file_hash=file_hash[:16] + "...",
                user_id=str(user_id),
                error=str(e)
            )
            return None

    async def list_user_files(
        self, 
        user_id: uuid.UUID, 
        db: AsyncSession,
        limit: int = 50,
        offset: int = 0
    ) -> List[File]:
        """
        List files for a specific user.
        
        Args:
            user_id: User ID
            db: Database session
            limit: Maximum number of files to return
            offset: Number of files to skip
            
        Returns:
            List of file records
        """
        try:
            result = await db.execute(
                select(File)
                .where(
                    File.user_id == user_id,
                    File.status != FileStatus.DELETED
                )
                .order_by(File.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            return result.scalars().all()
        except Exception as e:
            logger.error(
                "Error listing user files",
                user_id=str(user_id),
                error=str(e)
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to list files"
            )

    async def get_file(
        self, 
        file_id: uuid.UUID, 
        user_id: uuid.UUID, 
        db: AsyncSession
    ) -> Optional[File]:
        """
        Get a specific file by ID for a user.
        
        Args:
            file_id: File ID
            user_id: User ID
            db: Database session
            
        Returns:
            File record if found and accessible, None otherwise
        """
        try:
            result = await db.execute(
                select(File)
                .where(
                    File.id == file_id,
                    File.user_id == user_id,
                    File.status != FileStatus.DELETED
                )
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(
                "Error retrieving file",
                file_id=str(file_id),
                user_id=str(user_id),
                error=str(e)
            )
            return None

    async def delete_file(
        self, 
        file_id: uuid.UUID, 
        user_id: uuid.UUID, 
        db: AsyncSession
    ) -> bool:
        """
        Delete a file (soft delete in database, physical removal from disk).
        
        Args:
            file_id: File ID
            user_id: User ID
            db: Database session
            
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            # Get file record
            file_record = await self.get_file(file_id, user_id, db)
            if not file_record:
                return False
            
            # Update database record
            file_record.status = FileStatus.DELETED
            file_record.updated_at = datetime.now(timezone.utc)
            
            await db.commit()
            
            # Remove physical file asynchronously
            try:
                storage_path = Path(file_record.storage_path)
                if storage_path.exists():
                    storage_path.unlink()
                    logger.info(
                        "Physical file deleted",
                        file_id=str(file_id),
                        storage_path=str(storage_path)
                    )
            except Exception as e:
                logger.warning(
                    "Failed to delete physical file",
                    file_id=str(file_id),
                    storage_path=file_record.storage_path,
                    error=str(e)
                )
            
            logger.info(
                "File deleted successfully",
                file_id=str(file_id),
                user_id=str(user_id)
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Error deleting file",
                file_id=str(file_id),
                user_id=str(user_id),
                error=str(e)
            )
            await db.rollback()
            return False


# Global service instance
file_service = FileService()