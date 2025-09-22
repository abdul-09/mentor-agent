"""
Code Analysis Endpoints for AI Code Mentor
Handles PDF analysis, GitHub repository analysis, and Q&A.

Compliance:
- RULE PERF-004: Analysis response times
- RULE API-001: RESTful resource naming
"""

import uuid
from typing import Dict, List, Optional, Any

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import select

from src.dependencies import get_current_active_user, get_db
from src.models.analysis import Analysis, AnalysisStatus, AnalysisType
from src.models.file import File
from src.models.user import User
from src.services.ai_service import ai_service
from src.services.pdf_processor import pdf_processor
from src.services.redis_service import redis_service
from src.services.vector_search_service import vector_search_service
from src.security.rate_limiting import expensive_operation_rate_limit, api_rate_limit, user_rate_limit

logger = structlog.get_logger(__name__)
router = APIRouter()


# Request/Response Models
class PDFAnalysisRequest(BaseModel):
    """PDF analysis request model."""
    file_id: str = Field(..., description="ID of uploaded PDF file")
    analysis_type: str = Field(default="comprehensive", description="Type of analysis to perform")
    include_embeddings: bool = Field(default=True, description="Whether to generate embeddings")


class GitHubAnalysisRequest(BaseModel):
    """GitHub repository analysis request model."""
    repository_url: str = Field(..., description="GitHub repository URL")
    branch: str = Field(default="main", description="Branch to analyze")
    include_dependencies: bool = Field(default=True, description="Include dependency analysis")
    language_filter: Optional[List[str]] = Field(default=None, description="Filter by programming languages")


class QuestionRequest(BaseModel):
    """Q&A question request model."""
    question: str = Field(..., min_length=5, max_length=1000, description="Question to ask")
    analysis_id: Optional[str] = Field(default=None, description="Related analysis ID for context")
    conversation_id: Optional[str] = Field(default=None, description="Conversation ID for context")


class AnalysisResponse(BaseModel):
    """Analysis response model."""
    id: str = Field(..., description="Analysis ID")
    analysis_type: str = Field(..., description="Type of analysis")
    status: str = Field(..., description="Analysis status")
    result: Optional[Dict] = Field(default=None, description="Analysis results")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    metadata: Dict = Field(default_factory=dict, description="Additional metadata")


class AnalysisListResponse(BaseModel):
    """Analysis list response model."""
    analyses: List[AnalysisResponse] = Field(..., description="List of analyses")
    total: int = Field(..., description="Total number of analyses")
    limit: int = Field(..., description="Query limit")
    offset: int = Field(..., description="Query offset")


class QuestionResponse(BaseModel):
    """Q&A response model."""
    answer: str = Field(..., description="AI-generated answer")
    confidence: Optional[float] = Field(default=None, description="Confidence score")
    sources: List[str] = Field(default_factory=list, description="Source references")
    conversation_id: str = Field(..., description="Conversation ID")
    timestamp: str = Field(..., description="Response timestamp")


class ConversationMessage(BaseModel):
    """Conversation message model."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: str = Field(..., description="Message timestamp")
    analysis_id: Optional[str] = Field(default=None, description="Related analysis ID")
    sources: Optional[List[str]] = Field(default=None, description="Source references for assistant messages")


class ConversationHistory(BaseModel):
    """Conversation history response model."""
    conversation_id: str = Field(..., description="Conversation ID")
    messages: List[ConversationMessage] = Field(..., description="Conversation messages")
    total_messages: int = Field(..., description="Total number of messages")
    created_at: Optional[str] = Field(default=None, description="Conversation start time")
    last_activity: Optional[str] = Field(default=None, description="Last activity time")


class ConversationSummary(BaseModel):
    """Conversation summary model."""
    conversation_id: str = Field(..., description="Conversation ID")
    title: Optional[str] = Field(default=None, description="Conversation title")
    message_count: int = Field(..., description="Number of messages")
    last_activity: str = Field(..., description="Last activity timestamp")
    related_analysis_ids: List[str] = Field(default_factory=list, description="Related analysis IDs")


class ConversationListResponse(BaseModel):
    """Conversation list response model."""
    conversations: List[ConversationSummary] = Field(..., description="List of conversations")
    total: int = Field(..., description="Total number of conversations")
    limit: int = Field(..., description="Query limit")
    offset: int = Field(..., description="Query offset")


# In-memory conversation storage for when Redis is not available
_conversation_storage: Dict[str, List[Dict[str, Any]]] = {}


def _save_conversation_message_memory(conversation_id: str, message: Dict[str, Any]) -> bool:
    """Save conversation message to memory when Redis is unavailable."""
    try:
        if conversation_id not in _conversation_storage:
            _conversation_storage[conversation_id] = []
        
        message['timestamp'] = datetime.now(timezone.utc).isoformat()
        _conversation_storage[conversation_id].append(message)
        
        # Keep only last 100 messages
        _conversation_storage[conversation_id] = _conversation_storage[conversation_id][-100:]
        
        return True
    except Exception as e:
        logger.error("Failed to save message to memory", error=str(e))
        return False


def _get_conversation_history_memory(conversation_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Get conversation history from memory when Redis is unavailable."""
    try:
        messages = _conversation_storage.get(conversation_id, [])
        return messages[-limit:] if messages else []
    except Exception as e:
        logger.error("Failed to get conversation history from memory", error=str(e))
        return []
def analysis_to_response(analysis: Analysis) -> AnalysisResponse:
    """Convert Analysis model to response format."""
    return AnalysisResponse(
        id=str(analysis.id),
        analysis_type=analysis.analysis_type.value,
        status=analysis.status.value,
        result=analysis.results,
        error_message=analysis.error_message,
        created_at=analysis.created_at.isoformat(),
        updated_at=analysis.updated_at.isoformat(),
        metadata=analysis.configuration or {}
    )


async def process_pdf_analysis_background(
    file_id: uuid.UUID,
    user_id: uuid.UUID,
    analysis_id: uuid.UUID
):
    """Background task for PDF analysis processing."""
    from src.models.database import get_async_session
    
    async for db in get_async_session():
        try:
            # Get file and user records
            file_result = await db.execute(
                select(File).where(File.id == file_id, File.user_id == user_id)
            )
            file_record = file_result.scalar_one_or_none()
            
            user_result = await db.execute(
                select(User).where(User.id == user_id)
            )
            user = user_result.scalar_one_or_none()
            
            if not file_record or not user:
                logger.error(
                    "File or user not found for background processing",
                    file_id=str(file_id),
                    user_id=str(user_id)
                )
                return
            
            # Process PDF
            await pdf_processor.process_pdf_file(file_record, user, db)
            
        except Exception as e:
            logger.error(
                "Background PDF processing failed",
                file_id=str(file_id),
                user_id=str(user_id),
                error=str(e)
            )
        finally:
            await db.close()


# Endpoints
@router.post("/pdf", response_model=AnalysisResponse, status_code=status.HTTP_202_ACCEPTED, tags=["analysis"])
async def analyze_pdf(
    request: PDFAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Analyze uploaded PDF and extract content for Q&A.
    
    Starts background processing of PDF file including:
    - Text extraction using PyMuPDF/pdfplumber
    - Content chunking for optimal processing
    - Embedding generation for semantic search
    - Metadata extraction and storage
    """
    try:
        # Validate file ID format
        try:
            file_uuid = uuid.UUID(request.file_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file ID format"
            )
        
        # Get and validate file
        result = await db.execute(
            select(File).where(
                File.id == file_uuid,
                File.user_id == current_user.id
            )
        )
        file_record = result.scalar_one_or_none()
        
        if not file_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found or access denied"
            )
        
        # Check if file is already being processed
        existing_analysis = await db.execute(
            select(Analysis).where(
                Analysis.file_id == file_uuid,
                Analysis.analysis_type == AnalysisType.PDF_ANALYSIS,
                Analysis.status.in_([AnalysisStatus.IN_PROGRESS, AnalysisStatus.COMPLETED])
            )
        )
        existing = existing_analysis.scalar_one_or_none()
        
        if existing and existing.status == AnalysisStatus.IN_PROGRESS:
            return analysis_to_response(existing)
        elif existing and existing.status == AnalysisStatus.COMPLETED:
            logger.info(
                "Returning existing completed analysis",
                analysis_id=str(existing.id),
                file_id=request.file_id
            )
            return analysis_to_response(existing)
        
        # Create new analysis record
        analysis = Analysis(
            id=uuid.uuid4(),
            user_id=current_user.id,
            file_id=file_uuid,
            session_name=f"PDF Analysis - {file_record.original_filename}",
            analysis_type=AnalysisType.PDF_ANALYSIS,
            status=AnalysisStatus.PENDING,
            configuration={
                "analysis_type_detail": request.analysis_type,
                "include_embeddings": request.include_embeddings,
                "requested_at": str(uuid.uuid4())  # Use as timestamp placeholder
            }
        )
        
        db.add(analysis)
        await db.commit()
        await db.refresh(analysis)
        
        # Start background processing
        background_tasks.add_task(
            process_pdf_analysis_background,
            file_uuid,
            current_user.id,
            analysis.id
        )
        
        logger.info(
            "PDF analysis started",
            analysis_id=str(analysis.id),
            file_id=request.file_id,
            user_id=str(current_user.id)
        )
        
        return analysis_to_response(analysis)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to start PDF analysis",
            file_id=request.file_id,
            user_id=str(current_user.id),
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start PDF analysis"
        )


@router.post("/github", response_model=AnalysisResponse, status_code=status.HTTP_202_ACCEPTED, tags=["analysis"])
async def analyze_github_repo(
    request: GitHubAnalysisRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Analyze GitHub repository structure and code.
    
    TODO: Implement GitHub repository analysis including:
    - Repository cloning and code extraction
    - Multi-language code parsing
    - Architecture analysis and dependency mapping
    - Performance and security analysis
    """
    try:
        # Create placeholder analysis record
        analysis = Analysis(
            id=uuid.uuid4(),
            user_id=current_user.id,
            session_name=f"GitHub Analysis - {request.repository_url.split('/')[-1]}",
            analysis_type=AnalysisType.GITHUB_ANALYSIS,
            status=AnalysisStatus.PENDING,
            configuration={
                "repository_url": request.repository_url,
                "branch": request.branch,
                "include_dependencies": request.include_dependencies,
                "language_filter": request.language_filter
            }
        )
        
        db.add(analysis)
        await db.commit()
        await db.refresh(analysis)
        
        # TODO: Implement actual GitHub analysis
        analysis.status = AnalysisStatus.FAILED
        analysis.error_message = "GitHub analysis not yet implemented"
        await db.commit()
        
        logger.info(
            "GitHub analysis placeholder created",
            analysis_id=str(analysis.id),
            repository_url=request.repository_url
        )
        
        return analysis_to_response(analysis)
        
    except Exception as e:
        logger.error(
            "Failed to start GitHub analysis",
            repository_url=request.repository_url,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start GitHub analysis"
        )


@router.post("/qa", response_model=QuestionResponse, tags=["analysis"])
@user_rate_limit(calls=100, period=3600)  # 100 questions per hour per user
async def ask_question(
    request: QuestionRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Ask questions about analyzed content using AI.
    
    Provides intelligent Q&A based on:
    - Previously analyzed PDF content
    - Code analysis results
    - Conversation context
    """
    try:
        context = ""
        sources = []
        
        # Use vector search to get relevant context
        vector_context = await vector_search_service.get_context_for_question(
            question=request.question,
            user_id=str(current_user.id),
            analysis_id=request.analysis_id,
            max_context_length=3000  # Leave room for conversation history
        )
        
        if vector_context.get('context'):
            context = vector_context['context']
            # Add source information from vector search
            for source in vector_context.get('sources', []):
                source_info = f"Document {source.get('document_id')}"
                if source.get('page_number'):
                    source_info += f" (Page {source['page_number']})"
                if source.get('section'):
                    source_info += f" - {source['section']}"
                sources.append(source_info)
            
            logger.info(
                "Vector search context retrieved",
                context_length=len(context),
                source_count=len(sources),
                question=request.question[:100]
            )
        
        # Fallback: Get context from analysis if vector search didn't provide context
        if not context and request.analysis_id:
            try:
                analysis_uuid = uuid.UUID(request.analysis_id)
                result = await db.execute(
                    select(Analysis).where(
                        Analysis.id == analysis_uuid,
                        Analysis.user_id == current_user.id,
                        Analysis.status == AnalysisStatus.COMPLETED
                    )
                )
                analysis = result.scalar_one_or_none()
                
                if analysis and analysis.results:
                    # Extract relevant context from analysis results
                    if "chunking" in analysis.results:
                        chunks = analysis.results["chunking"].get("chunks", [])
                        # Use first few chunks as context
                        for chunk in chunks[:3]:
                            if "preview" in chunk:
                                context += chunk["preview"] + "\n\n"
                    
                    sources.append(f"Analysis {request.analysis_id}")
                    
            except ValueError:
                logger.warning(
                    "Invalid analysis ID format",
                    analysis_id=request.analysis_id
                )
        
        # If no specific context, provide general guidance
        if not context:
            context = """
            This is an AI Code Mentor platform that helps with code analysis and learning.
            You can upload PDF files containing code documentation, technical papers, or code snippets,
            and ask questions about the content.
            """
        
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # Get conversation history for context
        conversation_history = []
        if request.conversation_id:
            if redis_service.is_connected:
                try:
                    history = await redis_service.get_conversation_history(
                        conversation_id=request.conversation_id,
                        limit=10  # Last 10 messages for context
                    )
                    # Convert to format expected by AI service
                    for msg in reversed(history):  # Reverse to get chronological order
                        if msg.get('role') and msg.get('content'):
                            conversation_history.append({
                                'role': msg['role'],
                                'content': msg['content']
                            })
                            
                    logger.info(
                        "Retrieved conversation history from Redis",
                        conversation_id=conversation_id,
                        history_length=len(conversation_history)
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to retrieve conversation history from Redis",
                        conversation_id=request.conversation_id,
                        error=str(e)
                    )
            else:
                # Use memory fallback when Redis is not available
                try:
                    history = _get_conversation_history_memory(
                        conversation_id=request.conversation_id,
                        limit=10
                    )
                    for msg in history:
                        if msg.get('role') and msg.get('content'):
                            conversation_history.append({
                                'role': msg['role'],
                                'content': msg['content']
                            })
                    
                    logger.info(
                        "Retrieved conversation history from memory",
                        conversation_id=conversation_id,
                        history_length=len(conversation_history)
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to retrieve conversation history from memory",
                        conversation_id=request.conversation_id,
                        error=str(e)
                    )
        
        # Generate AI response
        answer = await ai_service.generate_qa_response(
            question=request.question,
            context=context,
            conversation_history=conversation_history
        )
        
        # Save conversation messages
        if redis_service.is_connected:
            try:
                # Save user question
                await redis_service.save_conversation_message(
                    conversation_id=conversation_id,
                    message={
                        'role': 'user',
                        'content': request.question,
                        'analysis_id': request.analysis_id,
                        'user_id': str(current_user.id)
                    }
                )
                
                # Save AI response
                await redis_service.save_conversation_message(
                    conversation_id=conversation_id,
                    message={
                        'role': 'assistant',
                        'content': answer,
                        'sources': sources,
                        'context_length': len(context)
                    }
                )
                
                logger.info(
                    "Conversation messages saved to Redis",
                    conversation_id=conversation_id
                )
            except Exception as e:
                logger.warning(
                    "Failed to save conversation messages to Redis",
                    conversation_id=conversation_id,
                    error=str(e)
                )
        else:
            # Use memory fallback when Redis is not available
            try:
                # Save user question
                _save_conversation_message_memory(
                    conversation_id=conversation_id,
                    message={
                        'role': 'user',
                        'content': request.question,
                        'analysis_id': request.analysis_id,
                        'user_id': str(current_user.id)
                    }
                )
                
                # Save AI response
                _save_conversation_message_memory(
                    conversation_id=conversation_id,
                    message={
                        'role': 'assistant',
                        'content': answer,
                        'sources': sources,
                        'context_length': len(context)
                    }
                )
                
                logger.info(
                    "Conversation messages saved to memory",
                    conversation_id=conversation_id
                )
            except Exception as e:
                logger.warning(
                    "Failed to save conversation messages to memory",
                    conversation_id=conversation_id,
                    error=str(e)
                )
        
        logger.info(
            "Q&A response generated",
            question_length=len(request.question),
            answer_length=len(answer),
            user_id=str(current_user.id),
            analysis_id=request.analysis_id
        )
        
        return QuestionResponse(
            answer=answer,
            confidence=None,  # TODO: Implement confidence scoring
            sources=sources,
            conversation_id=conversation_id,
            timestamp=str(uuid.uuid4())  # Placeholder timestamp
        )
        
    except Exception as e:
        logger.error(
            "Failed to generate Q&A response",
            question=request.question[:100],
            user_id=str(current_user.id),
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate response"
        )


@router.post("/chat", response_model=QuestionResponse, tags=["analysis"])
@user_rate_limit(calls=100, period=3600)  # 100 questions per hour per user
async def chat_with_context(
    request: QuestionRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Chat with AI about specific files or repositories with context.
    
    Provides intelligent conversation based on:
    - Specific file content
    - Repository analysis results
    - Conversation history
    """
    try:
        context = ""
        sources = []
        
        # Add context based on the type of content being discussed
        if request.analysis_id:
            # Get context from specific analysis
            try:
                analysis_uuid = uuid.UUID(request.analysis_id)
                result = await db.execute(
                    select(Analysis).where(
                        Analysis.id == analysis_uuid,
                        Analysis.user_id == current_user.id,
                        Analysis.status == AnalysisStatus.COMPLETED
                    )
                )
                analysis = result.scalar_one_or_none()
                
                if analysis and analysis.results:
                    # Extract relevant context from analysis results
                    if "chunking" in analysis.results:
                        chunks = analysis.results["chunking"].get("chunks", [])
                        # Use first few chunks as context
                        for chunk in chunks[:3]:
                            if "preview" in chunk:
                                context += chunk["preview"] + "\n\n"
                    
                    sources.append(f"Analysis {request.analysis_id}")
                    
            except ValueError:
                logger.warning(
                    "Invalid analysis ID format",
                    analysis_id=request.analysis_id
                )
        
        # Use vector search to get relevant context if we don't have specific analysis context
        if not context:
            vector_context = await vector_search_service.get_context_for_question(
                question=request.question,
                user_id=str(current_user.id),
                analysis_id=request.analysis_id,
                max_context_length=3000  # Leave room for conversation history
            )
            
            if vector_context.get('context'):
                context = vector_context['context']
                # Add source information from vector search
                for source in vector_context.get('sources', []):
                    source_info = f"Document {source.get('document_id')}"
                    if source.get('page_number'):
                        source_info += f" (Page {source['page_number']})"
                    if source.get('section'):
                        source_info += f" - {source['section']}"
                    sources.append(source_info)
                
                logger.info(
                    "Vector search context retrieved",
                    context_length=len(context),
                    source_count=len(sources),
                    question=request.question[:100]
                )
        
        # If no specific context, provide general guidance
        if not context:
            context = """
            This is an AI Code Mentor platform that helps with code analysis and learning.
            You can upload PDF files containing code documentation, technical papers, or code snippets,
            and ask questions about the content.
            """
        
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # Get conversation history for context
        conversation_history = []
        if request.conversation_id:
            if redis_service.is_connected:
                try:
                    history = await redis_service.get_conversation_history(
                        conversation_id=request.conversation_id,
                        limit=10  # Last 10 messages for context
                    )
                    # Convert to format expected by AI service
                    for msg in reversed(history):  # Reverse to get chronological order
                        if msg.get('role') and msg.get('content'):
                            conversation_history.append({
                                'role': msg['role'],
                                'content': msg['content']
                            })
                            
                    logger.info(
                        "Retrieved conversation history from Redis",
                        conversation_id=conversation_id,
                        history_length=len(conversation_history)
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to retrieve conversation history from Redis",
                        conversation_id=request.conversation_id,
                        error=str(e)
                    )
            else:
                # Use memory fallback when Redis is not available
                try:
                    history = _get_conversation_history_memory(
                        conversation_id=request.conversation_id,
                        limit=10
                    )
                    for msg in history:
                        if msg.get('role') and msg.get('content'):
                            conversation_history.append({
                                'role': msg['role'],
                                'content': msg['content']
                            })
                    
                    logger.info(
                        "Retrieved conversation history from memory",
                        conversation_id=conversation_id,
                        history_length=len(conversation_history)
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to retrieve conversation history from memory",
                        conversation_id=request.conversation_id,
                        error=str(e)
                    )
        
        # Generate AI response
        answer = await ai_service.generate_qa_response(
            question=request.question,
            context=context,
            conversation_history=conversation_history
        )
        
        # Save conversation messages
        if redis_service.is_connected:
            try:
                # Save user question
                await redis_service.save_conversation_message(
                    conversation_id=conversation_id,
                    message={
                        'role': 'user',
                        'content': request.question,
                        'analysis_id': request.analysis_id,
                        'user_id': str(current_user.id)
                    }
                )
                
                # Save AI response
                await redis_service.save_conversation_message(
                    conversation_id=conversation_id,
                    message={
                        'role': 'assistant',
                        'content': answer,
                        'sources': sources,
                        'context_length': len(context)
                    }
                )
                
                # Add conversation to user's conversation list
                await redis_service.add_user_conversation(
                    user_id=str(current_user.id),
                    conversation_id=conversation_id
                )
                
                logger.info(
                    "Conversation messages saved to Redis",
                    conversation_id=conversation_id
                )
            except Exception as e:
                logger.warning(
                    "Failed to save conversation messages to Redis",
                    conversation_id=conversation_id,
                    error=str(e)
                )
        else:
            # Use memory fallback when Redis is not available
            try:
                # Save user question
                _save_conversation_message_memory(
                    conversation_id=conversation_id,
                    message={
                        'role': 'user',
                        'content': request.question,
                        'analysis_id': request.analysis_id,
                        'user_id': str(current_user.id)
                    }
                )
                
                # Save AI response
                _save_conversation_message_memory(
                    conversation_id=conversation_id,
                    message={
                        'role': 'assistant',
                        'content': answer,
                        'sources': sources,
                        'context_length': len(context)
                    }
                )
                
                logger.info(
                    "Conversation messages saved to memory",
                    conversation_id=conversation_id
                )
            except Exception as e:
                logger.warning(
                    "Failed to save conversation messages to memory",
                    conversation_id=conversation_id,
                    error=str(e)
                )
        
        logger.info(
            "Chat response generated",
            question_length=len(request.question),
            answer_length=len(answer),
            user_id=str(current_user.id),
            analysis_id=request.analysis_id
        )
        
        return QuestionResponse(
            answer=answer,
            confidence=None,  # TODO: Implement confidence scoring
            sources=sources,
            conversation_id=conversation_id,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        logger.error(
            "Failed to generate chat response",
            question=request.question[:100],
            user_id=str(current_user.id),
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate response"
        )


@router.get("/sessions", response_model=AnalysisListResponse, tags=["analysis"])
async def list_analysis_sessions(
    limit: int = Query(default=50, ge=1, le=100, description="Maximum number of analyses to return"),
    offset: int = Query(default=0, ge=0, description="Number of analyses to skip"),
    analysis_type: Optional[str] = Query(default=None, description="Filter by analysis type"),
    status: Optional[str] = Query(default=None, description="Filter by status"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    List user's analysis sessions with optional filtering.
    
    Returns paginated list of analysis sessions with filtering options
    for type and status.
    """
    try:
        # Build query with filters
        query = select(Analysis).where(Analysis.user_id == current_user.id)
        
        if analysis_type:
            try:
                analysis_type_enum = AnalysisType(analysis_type)
                query = query.where(Analysis.analysis_type == analysis_type_enum)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid analysis type: {analysis_type}"
                )
        
        if status:
            try:
                status_enum = AnalysisStatus(status)
                query = query.where(Analysis.status == status_enum)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid status: {status}"
                )
        
        # Add pagination and ordering
        query = query.order_by(Analysis.created_at.desc()).limit(limit).offset(offset)
        
        result = await db.execute(query)
        analyses = result.scalars().all()
        
        analysis_responses = [analysis_to_response(analysis) for analysis in analyses]
        
        # TODO: Add actual total count query
        total = len(analysis_responses) + offset
        
        logger.info(
            "Analysis sessions listed",
            user_id=str(current_user.id),
            count=len(analyses),
            filters={"type": analysis_type, "status": status}
        )
        
        return AnalysisListResponse(
            analyses=analysis_responses,
            total=total,
            limit=limit,
            offset=offset
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to list analysis sessions",
            user_id=str(current_user.id),
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve analysis sessions"
        )


@router.get("/{analysis_id}", response_model=AnalysisResponse, tags=["analysis"])
async def get_analysis(
    analysis_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get detailed information about a specific analysis.
    
    Returns complete analysis results including metadata,
    processing status, and any error information.
    """
    try:
        # Validate analysis ID format
        try:
            analysis_uuid = uuid.UUID(analysis_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid analysis ID format"
            )
        
        # Get analysis
        result = await db.execute(
            select(Analysis).where(
                Analysis.id == analysis_uuid,
                Analysis.user_id == current_user.id
            )
        )
        analysis = result.scalar_one_or_none()
        
        if not analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Analysis not found or access denied"
            )
        
        logger.info(
            "Analysis retrieved",
            analysis_id=analysis_id,
            user_id=str(current_user.id),
            status=analysis.status.value
        )
        
        return analysis_to_response(analysis)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to retrieve analysis",
            analysis_id=analysis_id,
            user_id=str(current_user.id),
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve analysis"
        )


# Conversation Management Endpoints

@router.get("/conversations", response_model=ConversationListResponse, tags=["conversations"])
async def list_conversations(
    limit: int = Query(default=20, ge=1, le=50, description="Maximum number of conversations to return"),
    offset: int = Query(default=0, ge=0, description="Number of conversations to skip"),
    current_user: User = Depends(get_current_active_user)
):
    """
    List user's conversation history.
    
    Returns a paginated list of conversation summaries for the current user,
    ordered by last activity (most recent first).
    """
    try:
        if not redis_service.is_connected:
            return ConversationListResponse(
                conversations=[],
                total=0,
                limit=limit,
                offset=offset
            )
        
        # Get user's conversation IDs
        user_conversations = await redis_service.get_user_conversations(str(current_user.id))
        
        if not user_conversations:
            return ConversationListResponse(
                conversations=[],
                total=0,
                limit=limit,
                offset=offset
            )
        
        # Get conversation summaries with pagination
        conversations = []
        start_idx = offset
        end_idx = min(offset + limit, len(user_conversations))
        
        for conv_id in user_conversations[start_idx:end_idx]:
            try:
                # Get conversation history to create summary
                messages = await redis_service.get_conversation_history(conv_id, limit=1)
                if messages:
                    last_message = messages[0]
                    
                    # Extract related analysis IDs
                    related_analysis_ids = []
                    history = await redis_service.get_conversation_history(conv_id, limit=100)
                    for msg in history:
                        if msg.get('analysis_id') and msg['analysis_id'] not in related_analysis_ids:
                            related_analysis_ids.append(msg['analysis_id'])
                    
                    # Generate title from first user message
                    title = None
                    for msg in reversed(history):
                        if msg.get('role') == 'user':
                            title = msg.get('content', '')[:50] + ('...' if len(msg.get('content', '')) > 50 else '')
                            break
                    
                    conversations.append(ConversationSummary(
                        conversation_id=conv_id,
                        title=title,
                        message_count=len(history),
                        last_activity=last_message.get('timestamp', ''),
                        related_analysis_ids=related_analysis_ids
                    ))
            except Exception as e:
                logger.warning(
                    "Failed to get conversation summary",
                    conversation_id=conv_id,
                    error=str(e)
                )
                continue
        
        logger.info(
            "Conversations listed",
            user_id=str(current_user.id),
            total=len(user_conversations),
            returned=len(conversations)
        )
        
        return ConversationListResponse(
            conversations=conversations,
            total=len(user_conversations),
            limit=limit,
            offset=offset
        )
        
    except Exception as e:
        logger.error(
            "Failed to list conversations",
            user_id=str(current_user.id),
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conversations"
        )


@router.get("/conversations/{conversation_id}", response_model=ConversationHistory, tags=["conversations"])
async def get_conversation_history(
    conversation_id: str,
    limit: int = Query(default=50, ge=1, le=100, description="Maximum number of messages to return"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get detailed conversation history.
    
    Returns the complete message history for a specific conversation,
    including both user questions and AI responses.
    """
    try:
        if not redis_service.is_connected:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Conversation history service unavailable"
            )
        
        # Get conversation messages
        messages_data = await redis_service.get_conversation_history(
            conversation_id=conversation_id,
            limit=limit
        )
        
        if not messages_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
        # Verify user access (check if any message belongs to current user)
        user_has_access = any(
            msg.get('user_id') == str(current_user.id) 
            for msg in messages_data 
            if msg.get('user_id')
        )
        
        if not user_has_access:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this conversation"
            )
        
        # Convert to response format
        messages = []
        created_at = None
        last_activity = None
        
        for msg_data in reversed(messages_data):  # Reverse to get chronological order
            message = ConversationMessage(
                role=msg_data.get('role', 'unknown'),
                content=msg_data.get('content', ''),
                timestamp=msg_data.get('timestamp', ''),
                analysis_id=msg_data.get('analysis_id'),
                sources=msg_data.get('sources')
            )
            messages.append(message)
            
            # Track timestamps
            msg_timestamp = msg_data.get('timestamp', '')
            if not created_at or msg_timestamp < created_at:
                created_at = msg_timestamp
            if not last_activity or msg_timestamp > last_activity:
                last_activity = msg_timestamp
        
        logger.info(
            "Conversation history retrieved",
            conversation_id=conversation_id,
            user_id=str(current_user.id),
            message_count=len(messages)
        )
        
        return ConversationHistory(
            conversation_id=conversation_id,
            messages=messages,
            total_messages=len(messages),
            created_at=created_at,
            last_activity=last_activity
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to retrieve conversation history",
            conversation_id=conversation_id,
            user_id=str(current_user.id),
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conversation history"
        )


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["conversations"])
async def delete_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    Delete a conversation and all its messages.
    
    Permanently removes the conversation history. This action cannot be undone.
    """
    try:
        if not redis_service.is_connected:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Conversation service unavailable"
            )
        
        # Verify user access before deletion
        messages_data = await redis_service.get_conversation_history(
            conversation_id=conversation_id,
            limit=1
        )
        
        if not messages_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
        # Check if user owns the conversation
        user_has_access = any(
            msg.get('user_id') == str(current_user.id) 
            for msg in messages_data 
            if msg.get('user_id')
        )
        
        if not user_has_access:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this conversation"
            )
        
        # Delete the conversation
        conv_key = f"conversation:{conversation_id}"
        
        # Note: Redis service doesn't have a delete conversation method, so we'll use the client directly
        if redis_service.client:
            await redis_service.client.delete(conv_key)
        
        logger.info(
            "Conversation deleted",
            conversation_id=conversation_id,
            user_id=str(current_user.id)
        )
        
        return None  # 204 No Content
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to delete conversation",
            conversation_id=conversation_id,
            user_id=str(current_user.id),
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete conversation"
        )