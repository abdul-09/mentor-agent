"""
Analysis Models for AI Code Mentor
Handles analysis sessions, Q&A interactions, and AI processing results.

Compliance:
- RULE DB-001: 3NF normalization
- RULE DB-002: UUID primary keys with proper indexing
- RULE PRIVACY-002: Audit trails for all interactions
"""

import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from enum import Enum

import structlog
from sqlalchemy import (
    Column, String, Boolean, DateTime, Text, Integer,
    Float, ForeignKey, Enum as SQLEnum,
    Index, CheckConstraint
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from pydantic import BaseModel, Field

from src.models.database import Base

logger = structlog.get_logger(__name__)


class AnalysisType(str, Enum):
    """Analysis type enumeration."""
    PDF_ANALYSIS = "pdf_analysis"
    GITHUB_ANALYSIS = "github_analysis"
    CODE_REVIEW = "code_review"
    QA_SESSION = "qa_session"


class AnalysisStatus(str, Enum):
    """Analysis processing status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AnalysisSession(Base):
    """
    Analysis session model for tracking AI analysis requests.
    
    Stores metadata about analysis sessions including performance metrics.
    """
    __tablename__ = "analyses"
    
    # Primary key
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
        comment="Unique analysis session identifier"
    )
    
    # Relationships
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Session owner user ID"
    )
    
    file_id = Column(
        UUID(as_uuid=True),
        ForeignKey("files.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
        comment="Associated file ID (for PDF analysis)"
    )
    
    # Session metadata
    session_name = Column(
        String(200),
        nullable=False,
        comment="Human-readable session name"
    )
    
    analysis_type = Column(
        SQLEnum(AnalysisType),
        nullable=False,
        index=True,
        comment="Type of analysis being performed"
    )
    
    status = Column(
        SQLEnum(AnalysisStatus),
        default=AnalysisStatus.PENDING,
        nullable=False,
        index=True,
        comment="Analysis processing status"
    )
    
    # Configuration
    configuration = Column(
        JSONB,
        nullable=True,
        comment="Analysis configuration parameters"
    )
    
    # GitHub repository info (for repo analysis)
    repository_url = Column(
        String(500),
        nullable=True,
        comment="GitHub repository URL"
    )
    
    repository_branch = Column(
        String(100),
        nullable=True,
        comment="Git branch analyzed"
    )
    
    repository_commit = Column(
        String(40),
        nullable=True,
        comment="Git commit hash analyzed"
    )
    
    # Processing results
    results = Column(
        JSONB,
        nullable=True,
        comment="Analysis results and findings"
    )
    
    summary = Column(
        Text,
        nullable=True,
        comment="Human-readable analysis summary"
    )
    
    # Performance metrics
    processing_time_seconds = Column(
        Float,
        nullable=True,
        comment="Total processing time in seconds"
    )
    
    tokens_used = Column(
        Integer,
        nullable=True,
        comment="AI tokens consumed"
    )
    
    api_calls_made = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Number of API calls made"
    )
    
    # Cost tracking
    estimated_cost_usd = Column(
        Float,
        nullable=True,
        comment="Estimated cost in USD"
    )
    
    # Quality metrics
    confidence_score = Column(
        Float,
        nullable=True,
        comment="Analysis confidence score (0-1)"
    )
    
    complexity_score = Column(
        Float,
        nullable=True,
        comment="Code complexity score (0-10)"
    )
    
    quality_score = Column(
        Float,
        nullable=True,
        comment="Code quality score (0-10)"
    )
    
    security_score = Column(
        Float,
        nullable=True,
        comment="Security assessment score (0-10)"
    )
    
    performance_score = Column(
        Float,
        nullable=True,
        comment="Performance assessment score (0-10)"
    )
    
    maintainability_score = Column(
        Float,
        nullable=True,
        comment="Maintainability score (0-10)"
    )
    
    # Error handling
    error_message = Column(
        Text,
        nullable=True,
        comment="Error message if analysis failed"
    )
    
    progress_percentage = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Processing progress (0-100)"
    )
    
    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        comment="Session creation timestamp"
    )
    
    started_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Processing start timestamp"
    )
    
    completed_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Processing completion timestamp"
    )
    
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
        comment="Last update timestamp"
    )
    
    # Relationships
    user = relationship("User", back_populates="analyses")
    file = relationship("File", back_populates="analyses")
    qa_interactions = relationship("QAInteraction", back_populates="analysis", cascade="all, delete-orphan")
    
    # Indexes for performance
    __table_args__ = (
        Index("ix_analyses_user_type", "user_id", "analysis_type"),
        Index("ix_analyses_user_created", "user_id", "created_at"),
        Index("ix_analyses_status_created", "status", "created_at"),
        Index("ix_analyses_file_status", "file_id", "status"),
        CheckConstraint("progress_percentage >= 0 AND progress_percentage <= 100", name="ck_analyses_progress"),
        CheckConstraint("processing_time_seconds >= 0", name="ck_analyses_processing_time"),
        CheckConstraint("tokens_used >= 0", name="ck_analyses_tokens"),
        CheckConstraint("api_calls_made >= 0", name="ck_analyses_api_calls"),
        {"comment": "Analysis sessions and AI processing results"}
    )
    
    def start_processing(self) -> None:
        """Mark analysis as started."""
        self.status = AnalysisStatus.IN_PROGRESS
        self.started_at = datetime.now(timezone.utc)
        self.progress_percentage = 10
        
        logger.info("Analysis processing started", analysis_id=str(self.id))
    
    def update_progress(self, percentage: int, message: str = None) -> None:
        """Update processing progress."""
        self.progress_percentage = max(0, min(100, percentage))
        if message:
            logger.info("Analysis progress updated", 
                       analysis_id=str(self.id), 
                       progress=percentage, 
                       message=message)
    
    def mark_completed(self, results: Dict[str, Any] = None) -> None:
        """Mark analysis as completed."""
        self.status = AnalysisStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)
        self.progress_percentage = 100
        
        if results:
            self.results = results
        
        # Calculate processing time
        if self.started_at:
            duration = self.completed_at - self.started_at
            self.processing_time_seconds = duration.total_seconds()
        
        logger.info("Analysis completed", 
                   analysis_id=str(self.id), 
                   processing_time=self.processing_time_seconds)
    
    def mark_failed(self, error_message: str) -> None:
        """Mark analysis as failed."""
        self.status = AnalysisStatus.FAILED
        self.error_message = error_message
        self.completed_at = datetime.now(timezone.utc)
        
        if self.started_at:
            duration = self.completed_at - self.started_at
            self.processing_time_seconds = duration.total_seconds()
        
        logger.error("Analysis failed", 
                    analysis_id=str(self.id), 
                    error=error_message)
    
    def cancel(self) -> None:
        """Cancel the analysis."""
        self.status = AnalysisStatus.CANCELLED
        self.completed_at = datetime.now(timezone.utc)
        
        logger.info("Analysis cancelled", analysis_id=str(self.id))
    
    def __repr__(self) -> str:
        return f"<AnalysisSession(id={self.id}, type={self.analysis_type}, status={self.status})>"


class QAInteraction(Base):
    """
    Q&A interaction model for storing questions and AI responses.
    
    Tracks individual questions within analysis sessions.
    """
    __tablename__ = "qa_interactions"
    
    # Primary key
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
        comment="Unique interaction identifier"
    )
    
    # Relationships
    analysis_id = Column(
        UUID(as_uuid=True),
        ForeignKey("analyses.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Parent analysis session"
    )
    
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="User who asked the question"
    )
    
    # Question and answer
    question = Column(
        Text,
        nullable=False,
        comment="User's question"
    )
    
    answer = Column(
        Text,
        nullable=True,
        comment="AI-generated answer"
    )
    
    # Context and metadata
    context_chunks = Column(
        JSONB,
        nullable=True,
        comment="Relevant text chunks used for context"
    )
    
    confidence_score = Column(
        Float,
        nullable=True,
        comment="Answer confidence score (0-1)"
    )
    
    # Performance metrics
    response_time_ms = Column(
        Integer,
        nullable=True,
        comment="Response generation time in milliseconds"
    )
    
    tokens_used = Column(
        Integer,
        nullable=True,
        comment="Tokens consumed for this interaction"
    )
    
    model_used = Column(
        String(50),
        nullable=True,
        comment="AI model used for response"
    )
    
    # User feedback
    feedback_rating = Column(
        Integer,
        nullable=True,
        comment="User rating (1-5 stars)"
    )
    
    feedback_comment = Column(
        Text,
        nullable=True,
        comment="User feedback comment"
    )
    
    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        comment="Question timestamp"
    )
    
    answered_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Answer timestamp"
    )
    
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
        comment="Last update timestamp"
    )
    
    # Relationships
    analysis = relationship("AnalysisSession", back_populates="qa_interactions")
    user = relationship("User")
    
    # Indexes for performance
    __table_args__ = (
        Index("ix_qa_analysis_created", "analysis_id", "created_at"),
        Index("ix_qa_user_created", "user_id", "created_at"),
        CheckConstraint("feedback_rating >= 1 AND feedback_rating <= 5", name="ck_qa_rating"),
        CheckConstraint("confidence_score >= 0 AND confidence_score <= 1", name="ck_qa_confidence"),
        CheckConstraint("response_time_ms >= 0", name="ck_qa_response_time"),
        CheckConstraint("tokens_used >= 0", name="ck_qa_tokens"),
        {"comment": "Q&A interactions and AI responses"}
    )
    
    def set_answer(self, answer: str, confidence: float = None, 
                   tokens_used: int = None, model: str = None) -> None:
        """Set the AI answer for this interaction."""
        self.answer = answer
        self.answered_at = datetime.now(timezone.utc)
        self.confidence_score = confidence
        self.tokens_used = tokens_used
        self.model_used = model
        
        # Calculate response time
        if self.created_at:
            duration = self.answered_at - self.created_at
            self.response_time_ms = int(duration.total_seconds() * 1000)
        
        logger.info("Q&A interaction answered", 
                   interaction_id=str(self.id),
                   response_time_ms=self.response_time_ms)
    
    def set_feedback(self, rating: int, comment: str = None) -> None:
        """Set user feedback for this interaction."""
        if not 1 <= rating <= 5:
            raise ValueError("Rating must be between 1 and 5")
        
        self.feedback_rating = rating
        self.feedback_comment = comment
        
        logger.info("Q&A feedback received", 
                   interaction_id=str(self.id),
                   rating=rating)
    
    def __repr__(self) -> str:
        return f"<QAInteraction(id={self.id}, analysis_id={self.analysis_id})>"


# Pydantic models for API
class AnalysisResponse(BaseModel):
    """Analysis session response model."""
    id: str
    session_name: str
    analysis_type: str
    status: str
    progress_percentage: int
    repository_url: Optional[str]
    processing_time_seconds: Optional[float]
    confidence_score: Optional[float]
    created_at: datetime
    completed_at: Optional[datetime]
    
    class Config:
        from_attributes = True


class QARequest(BaseModel):
    """Q&A question request model."""
    question: str = Field(..., min_length=1, max_length=1000)


class QAResponse(BaseModel):
    """Q&A interaction response model."""
    id: str
    question: str
    answer: Optional[str]
    confidence_score: Optional[float]
    response_time_ms: Optional[int]
    created_at: datetime
    answered_at: Optional[datetime]
    
    class Config:
        from_attributes = True


# Alias for backwards compatibility
Analysis = AnalysisSession