"""
File Management Endpoints for AI Code Mentor
Handles PDF uploads and file operations with security controls.

Compliance:
- RULE SEC-001: Input validation and file security
- RULE PERF-004: File upload performance
- RULE API-001: RESTful resource naming
"""

import os
from pathlib import Path as PathLib
from fastapi.responses import FileResponse, StreamingResponse
import aiofiles
import structlog

import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, File, HTTPException, Path, Query, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.dependencies import get_current_active_user, get_db
from src.models.file import File as FileModel
from src.models.user import User
from src.services.file import file_service
from src.security.rate_limiting import file_upload_rate_limit, api_rate_limit

logger = structlog.get_logger(__name__)
router = APIRouter()


# Response Models
class FileResponse(BaseModel):
    """File information response model."""
    id: str = Field(..., description="File ID")
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="File type")
    file_size: int = Field(..., description="File size in bytes")
    mime_type: str = Field(..., description="MIME type")
    status: str = Field(..., description="File status")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class FileListResponse(BaseModel):
    """File list response model."""
    files: List[FileResponse] = Field(..., description="List of files")
    total: int = Field(..., description="Total number of files")
    limit: int = Field(..., description="Query limit")
    offset: int = Field(..., description="Query offset")


class FileUploadResponse(BaseModel):
    """File upload response model."""
    file: FileResponse = Field(..., description="Uploaded file information")
    message: str = Field(..., description="Success message")


# Helper functions
def file_to_response(file: FileModel) -> FileResponse:
    """Convert File model to response format."""
    return FileResponse(
        id=str(file.id),
        filename=file.original_filename,
        file_type=file.file_type.value,
        file_size=file.file_size,
        mime_type=file.mime_type,
        status=file.status.value,
        created_at=file.created_at.isoformat(),
        updated_at=file.updated_at.isoformat(),
        metadata=file.processing_metadata or {}
    )


# Endpoints
@router.get("/", response_model=FileListResponse, tags=["files"])
async def list_files(
    limit: int = Query(default=50, ge=1, le=100, description="Maximum number of files to return"),
    offset: int = Query(default=0, ge=0, description="Number of files to skip"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    List user's uploaded files with pagination.
    
    Returns paginated list of files uploaded by the current user.
    Only shows files that are not deleted.
    """
    try:
        files = await file_service.list_user_files(
            user_id=current_user.id,
            db=db,
            limit=limit,
            offset=offset
        )
        
        file_responses = [file_to_response(file) for file in files]
        
        # TODO: Add actual total count query for better pagination
        total = len(file_responses) + offset  # Simplified for now
        
        logger.info(
            "Files listed successfully",
            user_id=str(current_user.id),
            count=len(files),
            limit=limit,
            offset=offset
        )
        
        return FileListResponse(
            files=file_responses,
            total=total,
            limit=limit,
            offset=offset
        )
        
    except Exception as e:
        logger.error(
            "Failed to list files",
            user_id=str(current_user.id),
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve file list"
        )


@router.post("/upload", response_model=FileUploadResponse, status_code=status.HTTP_201_CREATED, tags=["files"])
@file_upload_rate_limit(calls=20, period=3600)  # 20 file uploads per hour per IP
async def upload_file(
    file: UploadFile = File(..., description="PDF file to upload"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload PDF file with comprehensive security validation.
    
    Features:
    - File type validation (PDF only)
    - Size limits enforcement 
    - MIME type verification
    - Virus scanning (TODO)
    - Duplicate detection
    - Secure storage with unique naming
    """
    try:
        # Upload file using service
        uploaded_file = await file_service.upload_file(
            file=file,
            user=current_user,
            db=db
        )
        
        file_response = file_to_response(uploaded_file)
        
        return FileUploadResponse(
            file=file_response,
            message="File uploaded successfully"
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions from service
        raise
    except Exception as e:
        logger.error(
            "Unexpected error during file upload",
            user_id=str(current_user.id),
            filename=getattr(file, 'filename', 'unknown'),
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="File upload failed due to server error"
        )


@router.get("/{file_id}", response_model=FileResponse, tags=["files"])
async def get_file(
    file_id: str = Path(..., description="File ID"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get detailed information about a specific file.
    
    Returns file metadata and information. Users can only access their own files.
    """
    try:
        # Validate file ID format
        try:
            file_uuid = uuid.UUID(file_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file ID format"
            )
        
        # Get file from service
        file = await file_service.get_file(
            file_id=file_uuid,
            user_id=current_user.id,
            db=db
        )
        
        if not file:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found or access denied"
            )
        
        logger.info(
            "File retrieved successfully",
            file_id=file_id,
            user_id=str(current_user.id)
        )
        
        return file_to_response(file)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to retrieve file",
            file_id=file_id,
            user_id=str(current_user.id),
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve file information"
        )


@router.delete("/{file_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["files"])
async def delete_file(
    file_id: str = Path(..., description="File ID"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a file (soft delete in database, physical removal from storage).
    
    Users can only delete their own files. This operation cannot be undone.
    """
    try:
        # Validate file ID format
        try:
            file_uuid = uuid.UUID(file_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file ID format"
            )
        
        # Delete file using service
        success = await file_service.delete_file(
            file_id=file_uuid,
            user_id=current_user.id,
            db=db
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found or access denied"
            )
        
        logger.info(
            "File deleted successfully",
            file_id=file_id,
            user_id=str(current_user.id)
        )
        
        # Return 204 No Content on successful deletion
        return
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to delete file",
            file_id=file_id,
            user_id=str(current_user.id),
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete file"
        )


@router.get("/{file_id}/download", tags=["files"])
async def download_file(
    file_id: str = Path(..., description="File ID"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Generate secure download URL or stream file content.
    
    TODO: Implement file streaming or signed URL generation.
    """
    try:
        # Validate file ID format
        try:
            file_uuid = uuid.UUID(file_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file ID format"
            )
        
        # Get file from service
        file = await file_service.get_file(
            file_id=file_uuid,
            user_id=current_user.id,
            db=db
        )
        
        if not file:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found or access denied"
            )
        
        # Validate file path exists and is accessible
        file_path = PathLib(file.storage_path)
        if not file_path.exists() or not file_path.is_file():
            logger.error(
                "File not found on disk",
                file_id=file_id,
                storage_path=file.storage_path
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found on storage"
            )
        
        # Record file access
        file.record_access()
        await db.commit()
        
        # Return file using FileResponse for efficient streaming
        logger.info(
            "File download initiated",
            file_id=file_id,
            user_id=str(current_user.id),
            filename=file.original_filename,
            size=file.file_size
        )
        
        return FileResponse(
            path=file_path,
            filename=file.original_filename,
            media_type=file.mime_type,
            headers={
                "Content-Disposition": f"attachment; filename={file.original_filename}",
                "X-File-ID": file_id,
                "X-Content-Type-Options": "nosniff"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to generate download URL",
            file_id=file_id,
            user_id=str(current_user.id),
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate download URL"
        )