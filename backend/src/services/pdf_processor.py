"""
PDF Processing Service for AI Code Mentor
Handles PDF text extraction, content analysis, and chunking with security compliance.

Compliance:
- RULE SEC-001: Secure file processing with input validation
- RULE PERF-004: Optimized PDF processing for large files
- RULE LOG-001: Structured logging for processing operations
"""

import asyncio
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import pdfplumber
import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import select

from src.config.settings import get_settings
from src.models.analysis import Analysis, AnalysisStatus, AnalysisType
from src.models.file import File, FileStatus
from src.models.user import User
from src.services.ai_service import ai_service
from src.services.vector_search_service import vector_search_service

logger = structlog.get_logger(__name__)
settings = get_settings()


class PDFProcessingError(Exception):
    """Custom exception for PDF processing errors."""
    pass


class TextChunk:
    """Represents a chunk of extracted text with metadata."""
    
    def __init__(
        self,
        content: str,
        page_number: int,
        chunk_index: int,
        start_char: int,
        end_char: int,
        metadata: Optional[Dict] = None
    ):
        self.content = content
        self.page_number = page_number
        self.chunk_index = chunk_index
        self.start_char = start_char
        self.end_char = end_char
        self.metadata = metadata or {}
        self.word_count = len(content.split())
        self.char_count = len(content)


class PDFProcessor:
    """Service class for PDF processing operations."""

    def __init__(self):
        self.max_chunk_size = 2000  # Characters per chunk
        self.chunk_overlap = 200    # Character overlap between chunks
        self.min_chunk_size = 100   # Minimum viable chunk size

    async def extract_text_pymupdf(self, file_path: str) -> Tuple[str, Dict]:
        """
        Extract text from PDF using PyMuPDF (fast, good for text extraction).
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple of (extracted_text, metadata)
            
        Raises:
            PDFProcessingError: If text extraction fails
        """
        try:
            doc = fitz.open(file_path)
            
            full_text = ""
            page_texts = []
            metadata = {
                "page_count": len(doc),
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "creator": doc.metadata.get("creator", ""),
                "producer": doc.metadata.get("producer", ""),
                "creation_date": doc.metadata.get("creationDate", ""),
                "modification_date": doc.metadata.get("modDate", ""),
                "extraction_method": "pymupdf"
            }
            
            logger.info(
                "Starting PyMuPDF text extraction",
                file_path=file_path,
                page_count=len(doc)
            )
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                
                if page_text.strip():
                    page_texts.append({
                        "page_number": page_num + 1,
                        "text": page_text,
                        "char_count": len(page_text)
                    })
                    full_text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
            
            doc.close()
            
            metadata["pages"] = page_texts
            metadata["total_chars"] = len(full_text)
            metadata["total_words"] = len(full_text.split())
            
            logger.info(
                "PyMuPDF text extraction completed",
                total_chars=metadata["total_chars"],
                total_words=metadata["total_words"],
                pages_with_text=len(page_texts)
            )
            
            return full_text, metadata
            
        except Exception as e:
            logger.error(
                "PyMuPDF text extraction failed",
                file_path=file_path,
                error=str(e)
            )
            raise PDFProcessingError(f"PyMuPDF extraction failed: {str(e)}")

    async def extract_text_pdfplumber(self, file_path: str) -> Tuple[str, Dict]:
        """
        Extract text from PDF using pdfplumber (better for tables and layout).
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple of (extracted_text, metadata)
            
        Raises:
            PDFProcessingError: If text extraction fails
        """
        try:
            with pdfplumber.open(file_path) as pdf:
                full_text = ""
                page_texts = []
                
                metadata = {
                    "page_count": len(pdf.pages),
                    "extraction_method": "pdfplumber"
                }
                
                logger.info(
                    "Starting pdfplumber text extraction",
                    file_path=file_path,
                    page_count=len(pdf.pages)
                )
                
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        
                        if page_text and page_text.strip():
                            page_texts.append({
                                "page_number": page_num + 1,
                                "text": page_text,
                                "char_count": len(page_text)
                            })
                            full_text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                            
                        # Extract tables if present
                        tables = page.extract_tables()
                        if tables:
                            for table_idx, table in enumerate(tables):
                                table_text = "\n".join(["\t".join(row) for row in table if row])
                                if table_text.strip():
                                    full_text += f"\n--- Table {table_idx + 1} on Page {page_num + 1} ---\n{table_text}\n"
                                    
                    except Exception as e:
                        logger.warning(
                            "Failed to extract text from page",
                            page_number=page_num + 1,
                            error=str(e)
                        )
                        continue
                
                metadata["pages"] = page_texts
                metadata["total_chars"] = len(full_text)
                metadata["total_words"] = len(full_text.split())
                
                logger.info(
                    "pdfplumber text extraction completed",
                    total_chars=metadata["total_chars"],
                    total_words=metadata["total_words"],
                    pages_with_text=len(page_texts)
                )
                
                return full_text, metadata
                
        except Exception as e:
            logger.error(
                "pdfplumber text extraction failed",
                file_path=file_path,
                error=str(e)
            )
            raise PDFProcessingError(f"pdfplumber extraction failed: {str(e)}")

    async def extract_text(
        self, 
        file_path: str, 
        method: str = "auto"
    ) -> Tuple[str, Dict]:
        """
        Extract text from PDF using specified or best method.
        
        Args:
            file_path: Path to the PDF file
            method: Extraction method ("pymupdf", "pdfplumber", "auto")
            
        Returns:
            Tuple of (extracted_text, metadata)
        """
        if not Path(file_path).exists():
            raise PDFProcessingError(f"File not found: {file_path}")
        
        if method == "pymupdf":
            return await self.extract_text_pymupdf(file_path)
        elif method == "pdfplumber":
            return await self.extract_text_pdfplumber(file_path)
        elif method == "auto":
            # Try PyMuPDF first (faster), fallback to pdfplumber if needed
            try:
                text, metadata = await self.extract_text_pymupdf(file_path)
                # Check if extraction was successful (enough text extracted)
                if len(text.strip()) > 100:
                    return text, metadata
                else:
                    logger.info("PyMuPDF extracted minimal text, trying pdfplumber")
                    return await self.extract_text_pdfplumber(file_path)
            except PDFProcessingError:
                logger.info("PyMuPDF failed, trying pdfplumber")
                return await self.extract_text_pdfplumber(file_path)
        else:
            raise PDFProcessingError(f"Unknown extraction method: {method}")

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Remove common PDF artifacts
        text = re.sub(r'(\x0c|\f)', '\n', text)  # Form feed characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)  # Control characters
        
        # Normalize line breaks
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove page headers/footers (basic pattern)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Skip likely headers/footers (short lines with page numbers, etc.)
            if len(line) < 5 and re.match(r'^\d+$', line):
                continue
            if line:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    def create_chunks(self, text: str, metadata: Dict) -> List[TextChunk]:
        """
        Split text into semantically meaningful chunks.
        
        Args:
            text: Clean text to chunk
            metadata: Text metadata
            
        Returns:
            List of TextChunk objects
        """
        chunks = []
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        current_chunk = ""
        current_char_pos = 0
        chunk_index = 0
        current_page = 1
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check if adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) + 2 > self.max_chunk_size:
                # Save current chunk if it's not empty and meets minimum size
                if len(current_chunk.strip()) >= self.min_chunk_size:
                    chunks.append(TextChunk(
                        content=current_chunk.strip(),
                        page_number=current_page,
                        chunk_index=chunk_index,
                        start_char=current_char_pos - len(current_chunk),
                        end_char=current_char_pos,
                        metadata={"source": "pdf_extraction"}
                    ))
                    chunk_index += 1
                
                # Start new chunk with overlap if needed
                if self.chunk_overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_text + "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            
            current_char_pos += len(paragraph) + 2
            
            # Update page number based on page markers
            if "--- Page" in paragraph:
                page_match = re.search(r'--- Page (\d+) ---', paragraph)
                if page_match:
                    current_page = int(page_match.group(1))
        
        # Add final chunk
        if len(current_chunk.strip()) >= self.min_chunk_size:
            chunks.append(TextChunk(
                content=current_chunk.strip(),
                page_number=current_page,
                chunk_index=chunk_index,
                start_char=current_char_pos - len(current_chunk),
                end_char=current_char_pos,
                metadata={"source": "pdf_extraction"}
            ))
        
        logger.info(
            "Text chunking completed",
            total_chunks=len(chunks),
            avg_chunk_size=sum(c.char_count for c in chunks) // len(chunks) if chunks else 0,
            total_chars=sum(c.char_count for c in chunks)
        )
        
        return chunks

    async def process_pdf_file(
        self, 
        file_record: File, 
        user: User, 
        db: AsyncSession
    ) -> Analysis:
        """
        Complete PDF processing pipeline: extract, clean, chunk, and analyze.
        
        Args:
            file_record: File database record
            user: User who owns the file
            db: Database session
            
        Returns:
            Analysis record with processing results
            
        Raises:
            PDFProcessingError: If processing fails
        """
        try:
            # Create analysis record
            analysis = Analysis(
                id=uuid.uuid4(),
                user_id=user.id,
                file_id=file_record.id,
                session_name=f"PDF Processing - {file_record.original_filename}",
                analysis_type=AnalysisType.PDF_ANALYSIS,
                status=AnalysisStatus.IN_PROGRESS,
                configuration={"processing_started": datetime.now(timezone.utc).isoformat()}
            )
            
            db.add(analysis)
            await db.commit()
            await db.refresh(analysis)
            
            logger.info(
                "Starting PDF processing",
                file_id=str(file_record.id),
                analysis_id=str(analysis.id),
                user_id=str(user.id)
            )
            
            # Extract text
            text, extraction_metadata = await self.extract_text(file_record.storage_path)
            
            if not text.strip():
                raise PDFProcessingError("No text could be extracted from PDF")
            
            # Clean text
            cleaned_text = self.clean_text(text)
            
            # Create chunks
            chunks = self.create_chunks(cleaned_text, extraction_metadata)
            
            if not chunks:
                raise PDFProcessingError("No meaningful text chunks could be created")
            
            # Generate embeddings for chunks (first few chunks to start)
            chunk_embeddings = []
            for i, chunk in enumerate(chunks[:5]):  # Limit to first 5 chunks for now
                try:
                    embeddings = await ai_service.create_embeddings(chunk.content)
                    chunk_embeddings.append({
                        "chunk_index": i,
                        "embedding": embeddings[0] if embeddings else None,
                        "content_preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
                    })
                except Exception as e:
                    logger.warning(
                        "Failed to create embedding for chunk",
                        chunk_index=i,
                        error=str(e)
                    )
            
            # Update analysis with results
            analysis.status = AnalysisStatus.COMPLETED
            analysis.results = {
                "text_extraction": {
                    "method": extraction_metadata.get("extraction_method"),
                    "total_chars": len(cleaned_text),
                    "total_words": len(cleaned_text.split()),
                    "page_count": extraction_metadata.get("page_count", 0)
                },
                "chunking": {
                    "total_chunks": len(chunks),
                    "avg_chunk_size": sum(c.char_count for c in chunks) // len(chunks),
                    "chunks": [
                        {
                            "index": c.chunk_index,
                            "page": c.page_number,
                            "char_count": c.char_count,
                            "word_count": c.word_count,
                            "preview": c.content[:100] + "..." if len(c.content) > 100 else c.content
                        } for c in chunks
                    ]
                },
                "embeddings": {
                    "generated_count": len(chunk_embeddings),
                    "embedding_dimension": len(chunk_embeddings[0]["embedding"]) if chunk_embeddings else 0,
                    "chunks_with_embeddings": chunk_embeddings
                },
                "metadata": extraction_metadata
            }
            # Store vectors in Pinecone for semantic search
            try:
                if chunk_embeddings:
                    # Prepare chunks for vector storage
                    vector_chunks = []
                    for i, chunk in enumerate(chunks):
                        if i < len(chunk_embeddings):
                            vector_chunks.append({
                                'text': chunk.content,
                                'page_number': chunk.page_number,
                                'metadata': {
                                    'chunk_index': i,
                                    'char_count': chunk.char_count,
                                    'word_count': chunk.word_count,
                                    'start_char': chunk.start_char,
                                    'end_char': chunk.end_char
                                }
                            })
                    
                    # Store in vector database
                    vector_result = await vector_search_service.store_document_chunks(
                        document_id=str(analysis.id),
                        chunks=vector_chunks,
                        user_id=str(user.id),
                        document_type="pdf",
                        namespace=""  # Use default namespace
                    )
                    
                    # Add vector storage info to results
                    analysis.results["vector_storage"] = vector_result
                    
                    logger.info(
                        "Vectors stored for PDF analysis",
                        analysis_id=str(analysis.id),
                        stored_count=vector_result.get('stored_count', 0),
                        vector_storage_success=vector_result.get('stored', False)
                    )
                
            except Exception as e:
                logger.warning(
                    "Failed to store vectors for PDF analysis",
                    analysis_id=str(analysis.id),
                    error=str(e)
                )
                # Don't fail the entire process if vector storage fails
                analysis.results["vector_storage"] = {
                    "stored": False,
                    "error": str(e)
                }
            
            analysis.configuration["processing_completed"] = datetime.now(timezone.utc).isoformat()
            analysis.configuration["processing_duration"] = (
                datetime.now(timezone.utc) - 
                datetime.fromisoformat(analysis.configuration["processing_started"].replace("Z", "+00:00"))
            ).total_seconds()
            
            # Update file status
            file_record.status = FileStatus.PROCESSED
            file_record.processing_metadata["analysis_id"] = str(analysis.id)
            file_record.processing_metadata["processed_at"] = datetime.now(timezone.utc).isoformat()
            
            await db.commit()
            
            logger.info(
                "PDF processing completed successfully",
                analysis_id=str(analysis.id),
                total_chunks=len(chunks),
                total_chars=len(cleaned_text),
                embeddings_generated=len(chunk_embeddings)
            )
            
            return analysis
            
        except Exception as e:
            logger.error(
                "PDF processing failed",
                file_id=str(file_record.id),
                error=str(e)
            )
            
            # Update analysis status to failed
            if 'analysis' in locals():
                analysis.status = AnalysisStatus.FAILED
                analysis.error_message = str(e)
                analysis.configuration["processing_failed"] = datetime.now(timezone.utc).isoformat()
                await db.commit()
            
            raise PDFProcessingError(f"PDF processing failed: {str(e)}")


# Global service instance
pdf_processor = PDFProcessor()