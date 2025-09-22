"""
Vector Search Service for AI Code Mentor
Implements semantic search using Pinecone and OpenAI embeddings.

Compliance:
- RULE PERF-003: Optimized vector operations
- RULE SEC-005: Secure API handling
- RULE AI-001: Context-aware responses
"""

import uuid
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone

import structlog
from openai import AsyncOpenAI

from src.config.settings import get_settings
from src.services.pinecone_service import pinecone_service
from src.services.ai_service import ai_service

logger = structlog.get_logger(__name__)
settings = get_settings()


class VectorSearchService:
    """
    Production-grade vector search service combining OpenAI embeddings with Pinecone.
    
    Provides semantic search, context retrieval, and intelligent document matching
    for enhanced Q&A and analysis capabilities.
    """
    
    def __init__(self):
        self.embedding_model = settings.OPENAI_EMBEDDING_MODEL
        self.embedding_dimension = 1536  # OpenAI text-embedding-ada-002
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text using OpenAI.
        
        Args:
            text: Text to embed
            
        Returns:
            List[float]: Embedding vector
        """
        try:
            # Use the AI service to generate embeddings
            client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            
            response = await client.embeddings.create(
                model=self.embedding_model,
                input=text.strip()
            )
            
            embedding = response.data[0].embedding
            
            logger.debug(
                "Embedding generated successfully",
                text_length=len(text),
                embedding_dimension=len(embedding)
            )
            
            return embedding
            
        except Exception as e:
            logger.error(
                "Failed to generate embedding",
                error=str(e),
                text_length=len(text)
            )
            raise
    
    async def store_document_chunks(
        self,
        document_id: str,
        chunks: List[Dict[str, Any]],
        user_id: str,
        document_type: str = "pdf",
        namespace: str = ""
    ) -> Dict[str, Any]:
        """
        Store document chunks as vectors in Pinecone.
        
        Args:
            document_id: Unique document identifier
            chunks: List of text chunks with metadata
            user_id: User ID for access control
            document_type: Type of document (pdf, github, etc.)
            namespace: Optional namespace for organization
            
        Returns:
            Dict containing storage results
        """
        if not pinecone_service.is_connected:
            logger.warning("Pinecone not available, skipping vector storage")
            return {"stored": False, "reason": "Pinecone not available"}
        
        try:
            vectors = []
            
            for i, chunk in enumerate(chunks):
                text = chunk.get('text', '')
                if not text.strip():
                    continue
                
                # Generate embedding for chunk
                embedding = await self.generate_embedding(text)
                
                # Create vector with metadata
                vector_id = f"{document_id}_{i}"
                vector = {
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {
                        "document_id": document_id,
                        "user_id": user_id,
                        "document_type": document_type,
                        "chunk_index": i,
                        "text": text[:1000],  # Store first 1000 chars for preview
                        "full_text": text,
                        "page_number": chunk.get('page_number'),
                        "section": chunk.get('section'),
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        **chunk.get('metadata', {})
                    }
                }
                vectors.append(vector)
            
            # Store vectors in Pinecone
            result = await pinecone_service.upsert_vectors(
                vectors=vectors,
                namespace=namespace
            )
            
            logger.info(
                "Document chunks stored successfully",
                document_id=document_id,
                chunk_count=len(vectors),
                stored_count=result.get('successful_upserts', 0)
            )
            
            return {
                "stored": True,
                "document_id": document_id,
                "chunk_count": len(vectors),
                "stored_count": result.get('successful_upserts', 0),
                "namespace": namespace
            }
            
        except Exception as e:
            logger.error(
                "Failed to store document chunks",
                error=str(e),
                document_id=document_id,
                chunk_count=len(chunks)
            )
            return {
                "stored": False,
                "error": str(e),
                "document_id": document_id
            }
    
    async def search_similar_content(
        self,
        query: str,
        user_id: str,
        top_k: int = 5,
        document_ids: Optional[List[str]] = None,
        document_types: Optional[List[str]] = None,
        namespace: str = ""
    ) -> Dict[str, Any]:
        """
        Search for similar content using semantic similarity.
        
        Args:
            query: Search query
            user_id: User ID for access control
            top_k: Number of results to return
            document_ids: Filter by specific document IDs
            document_types: Filter by document types
            namespace: Optional namespace to search within
            
        Returns:
            Dict containing search results and metadata
        """
        if not pinecone_service.is_connected:
            logger.warning("Pinecone not available, returning empty results")
            return {
                "results": [],
                "total_results": 0,
                "search_performed": False,
                "reason": "Pinecone not available"
            }
        
        try:
            # Generate embedding for query
            query_embedding = await self.generate_embedding(query)
            
            # Build metadata filter
            filter_metadata = {"user_id": user_id}
            
            if document_ids:
                filter_metadata["document_id"] = {"$in": document_ids}
            
            if document_types:
                filter_metadata["document_type"] = {"$in": document_types}
            
            # Search vectors
            search_result = await pinecone_service.query_vectors(
                query_vector=query_embedding,
                top_k=top_k,
                namespace=namespace,
                filter_metadata=filter_metadata,
                include_metadata=True
            )
            
            # Process results
            processed_results = []
            for result in search_result.get('results', []):
                metadata = result.get('metadata', {})
                processed_results.append({
                    "id": result["id"],
                    "score": result["score"],
                    "document_id": metadata.get("document_id"),
                    "document_type": metadata.get("document_type"),
                    "text": metadata.get("full_text", metadata.get("text", "")),
                    "preview": metadata.get("text", "")[:200],
                    "page_number": metadata.get("page_number"),
                    "section": metadata.get("section"),
                    "chunk_index": metadata.get("chunk_index"),
                    "created_at": metadata.get("created_at")
                })
            
            logger.info(
                "Semantic search completed",
                query_length=len(query),
                results_count=len(processed_results),
                top_k=top_k
            )
            
            return {
                "results": processed_results,
                "total_results": len(processed_results),
                "search_performed": True,
                "query": query,
                "search_time_ms": search_result.get('operation_time_ms', 0)
            }
            
        except Exception as e:
            logger.error(
                "Semantic search failed",
                error=str(e),
                query_length=len(query),
                top_k=top_k
            )
            return {
                "results": [],
                "total_results": 0,
                "search_performed": False,
                "error": str(e)
            }
    
    async def get_context_for_question(
        self,
        question: str,
        user_id: str,
        analysis_id: Optional[str] = None,
        max_context_length: int = 4000
    ) -> Dict[str, Any]:
        """
        Get relevant context for a question using semantic search.
        
        Args:
            question: Question to find context for
            user_id: User ID for access control
            analysis_id: Optional analysis ID to filter results
            max_context_length: Maximum context length in characters
            
        Returns:
            Dict containing context and source information
        """
        try:
            # Search for relevant content
            document_ids = [analysis_id] if analysis_id else None
            
            search_results = await self.search_similar_content(
                query=question,
                user_id=user_id,
                top_k=10,  # Get more results for better context
                document_ids=document_ids
            )
            
            if not search_results.get('search_performed') or not search_results.get('results'):
                return {
                    "context": "",
                    "sources": [],
                    "source_count": 0,
                    "context_length": 0
                }
            
            # Build context from top results
            context_parts = []
            sources = []
            current_length = 0
            
            for result in search_results['results']:
                text = result.get('text', '')
                if not text:
                    continue
                
                # Check if adding this text would exceed limit
                if current_length + len(text) > max_context_length:
                    # Try to add a truncated version
                    remaining_space = max_context_length - current_length - 50  # Leave some buffer
                    if remaining_space > 100:  # Only add if meaningful amount of text can fit
                        text = text[:remaining_space] + "..."
                    else:
                        break
                
                context_parts.append(text)
                current_length += len(text)
                
                # Add source information
                source_info = {
                    "document_id": result.get('document_id'),
                    "document_type": result.get('document_type'),
                    "page_number": result.get('page_number'),
                    "section": result.get('section'),
                    "relevance_score": result.get('score'),
                    "preview": result.get('preview', '')
                }
                sources.append(source_info)
            
            context = "\n\n".join(context_parts)
            
            logger.info(
                "Context retrieved for question",
                question_length=len(question),
                context_length=len(context),
                source_count=len(sources)
            )
            
            return {
                "context": context,
                "sources": sources,
                "source_count": len(sources),
                "context_length": len(context),
                "search_results_count": len(search_results['results'])
            }
            
        except Exception as e:
            logger.error(
                "Failed to get context for question",
                error=str(e),
                question_length=len(question)
            )
            return {
                "context": "",
                "sources": [],
                "source_count": 0,
                "context_length": 0,
                "error": str(e)
            }
    
    async def delete_document_vectors(
        self,
        document_id: str,
        user_id: str,
        namespace: str = ""
    ) -> Dict[str, Any]:
        """
        Delete all vectors for a specific document.
        
        Args:
            document_id: Document ID to delete
            user_id: User ID for access control
            namespace: Optional namespace
            
        Returns:
            Dict containing deletion results
        """
        if not pinecone_service.is_connected:
            logger.warning("Pinecone not available, skipping vector deletion")
            return {"deleted": False, "reason": "Pinecone not available"}
        
        try:
            # Query to get all vector IDs for this document
            search_result = await pinecone_service.query_vectors(
                query_vector=[0.0] * self.embedding_dimension,  # Dummy vector
                top_k=10000,  # Large number to get all chunks
                namespace=namespace,
                filter_metadata={
                    "document_id": document_id,
                    "user_id": user_id
                },
                include_metadata=False
            )
            
            if not search_result.get('results'):
                return {"deleted": True, "deleted_count": 0}
            
            # Extract vector IDs
            vector_ids = [result["id"] for result in search_result['results']]
            
            # Delete vectors
            delete_result = await pinecone_service.delete_vectors(
                vector_ids=vector_ids,
                namespace=namespace
            )
            
            logger.info(
                "Document vectors deleted",
                document_id=document_id,
                deleted_count=len(vector_ids)
            )
            
            return {
                "deleted": True,
                "document_id": document_id,
                "deleted_count": len(vector_ids)
            }
            
        except Exception as e:
            logger.error(
                "Failed to delete document vectors",
                error=str(e),
                document_id=document_id
            )
            return {
                "deleted": False,
                "error": str(e),
                "document_id": document_id
            }
    
    async def get_service_stats(self) -> Dict[str, Any]:
        """
        Get vector search service statistics.
        
        Returns:
            Dict containing service statistics
        """
        try:
            if not pinecone_service.is_connected:
                return {
                    "available": False,
                    "reason": "Pinecone not connected"
                }
            
            # Get Pinecone index stats
            index_stats = await pinecone_service.get_index_stats()
            
            return {
                "available": True,
                "pinecone_stats": index_stats,
                "embedding_model": self.embedding_model,
                "embedding_dimension": self.embedding_dimension
            }
            
        except Exception as e:
            logger.error("Failed to get service stats", error=str(e))
            return {
                "available": False,
                "error": str(e)
            }


# Global vector search service instance
vector_search_service = VectorSearchService()