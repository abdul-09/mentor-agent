"""
Models package for AI Code Mentor
Exports all database models for easy importing.
"""

from src.models.database import Base, get_async_session, init_database, close_database
from src.models.user import User, UserCreate, UserUpdate, UserResponse, UserSummary
from src.models.session import UserSession, SessionStatus, SessionResponse, SessionSummary
from src.models.file import (
    File, FileStatus, FileType, SecurityScanStatus,
    FileUploadResponse, FileResponse, FileSummary
)
from src.models.analysis import (
    AnalysisSession, QAInteraction, AnalysisType, AnalysisStatus,
    AnalysisResponse, QARequest, QAResponse
)

__all__ = [
    # Database
    "Base",
    "get_async_session",
    "init_database", 
    "close_database",
    
    # User models
    "User",
    "UserCreate",
    "UserUpdate", 
    "UserResponse",
    "UserSummary",
    
    # Session models
    "UserSession",
    "SessionStatus",
    "SessionResponse",
    "SessionSummary",
    
    # File models
    "File",
    "FileStatus",
    "FileType",
    "SecurityScanStatus",
    "FileUploadResponse",
    "FileResponse",
    "FileSummary",
    
    # Analysis models
    "AnalysisSession",
    "QAInteraction",
    "AnalysisType",
    "AnalysisStatus",
    "AnalysisResponse",
    "QARequest",
    "QAResponse",
]