"""
Pinecone Vector Database Service for AI Code Mentor
Implements production-grade vector storage and semantic search.

Compliance:
- RULE PERF-003: Optimized vector operations with connection pooling
- RULE SEC-005: Secure API key handling
- RULE LOG-001: Structured logging with performance metrics
"""

import time
from typing import Dict, List, Optional, Any, Tuple
import asyncio
from datetime import datetime, timezone

import structlog
import pinecone
from pinecone import Pinecone, ServerlessSpec
import numpy as np

from src.config.settings import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class PineconeService:
    """
    Production-grade Pinecone vector database service.
    
    Provides semantic search, vector storage, and metadata management
    with proper error handling and performance optimization.
    """
    
    def __init__(self):
        self.client: Optional[Pinecone] = None
        self.index = None
        self.is_connected = False
        self.index_name = settings.PINECONE_INDEX_NAME
        self.dimension = 1536  # OpenAI text-embedding-ada-002 dimension
        self.metric = "cosine"
        
        # Performance tracking
        self._operation_times: List[float] = []
        
    async def connect(self) -> bool:
        """
        Initialize Pinecone connection and ensure index exists.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            start_time = time.time()
            
            # Initialize Pinecone client
            self.client = Pinecone(api_key=settings.PINECONE_API_KEY)
            
            # Ensure index exists
            await self._ensure_index_exists()
            
            # Get index reference
            self.index = self.client.Index(self.index_name)
            
            self.is_connected = True
            connect_time = time.time() - start_time
            
            logger.info(
                "Pinecone connection established",
                index_name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                connect_time_ms=round(connect_time * 1000, 2)
            )
            
            return True
            
        except Exception as e:
            error_msg = str(e)
            
            # Provide specific guidance for common configuration errors
            if "NOT_FOUND" in error_msg and "region" in error_msg:
                logger.error(
                    "Failed to connect to Pinecone - Invalid region",
                    error=error_msg,
                    index_name=self.index_name,
                    configured_region=settings.PINECONE_ENVIRONMENT,
                    fix_suggestion="Update PINECONE_ENVIRONMENT in .env to a valid AWS region (e.g., 'us-east-1')"
                )
            else:
                logger.error(
                    "Failed to connect to Pinecone",
                    error=error_msg,
                    index_name=self.index_name
                )
            
            self.is_connected = False
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Pinecone."""
        try:
            self.client = None
            self.index = None
            self.is_connected = False
            
            logger.info("Pinecone connection closed")
            
        except Exception as e:
            logger.error("Error closing Pinecone connection", error=str(e))
    
    async def _ensure_index_exists(self) -> None:
        """Ensure the Pinecone index exists with proper configuration."""
        try:
            # Validate region format
            region = settings.PINECONE_ENVIRONMENT.strip()
            if not region or " " in region:
                raise ValueError(
                    f"Invalid Pinecone region format: '{region}'. "
                    "Use valid AWS region format like 'us-east-1'"
                )
            
            # List existing indexes
            existing_indexes = [index.name for index in self.client.list_indexes()]
            
            if self.index_name not in existing_indexes:
                logger.info(
                    "Creating new Pinecone index",
                    index_name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    region=region
                )
                
                # Create index with serverless spec (recommended for most use cases)
                self.client.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=region
                    )
                )
                
                # Wait for index to be ready
                await self._wait_for_index_ready()
                
                logger.info(
                    "Pinecone index created successfully",
                    index_name=self.index_name,
                    region=region
                )
            else:
                logger.info(
                    "Using existing Pinecone index",
                    index_name=self.index_name
                )
                
        except Exception as e:
            error_msg = str(e)
            
            # Provide specific guidance for common errors
            if "NOT_FOUND" in error_msg and "region" in error_msg:
                logger.error(
                    "Invalid Pinecone region configuration",
                    error=error_msg,
                    index_name=self.index_name,
                    configured_region=settings.PINECONE_ENVIRONMENT,
                    suggestion="Check PINECONE_ENVIRONMENT in .env file. Use valid AWS region like 'us-east-1'"
                )
            else:
                logger.error(
                    "Failed to ensure Pinecone index exists",
                    error=error_msg,
                    index_name=self.index_name
                )
            raise
    
    async def _wait_for_index_ready(self, timeout: int = 60) -> None:
        """Wait for index to be ready for operations."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                index_description = self.client.describe_index(self.index_name)
                if index_description.status.ready:
                    return
                
                logger.info(
                    "Waiting for index to be ready",
                    index_name=self.index_name,
                    status=index_description.status.state
                )
                
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.warning(
                    "Error checking index status",
                    error=str(e),
                    index_name=self.index_name
                )
                await asyncio.sleep(2)
        
        raise TimeoutError(f"Index {self.index_name} not ready after {timeout} seconds")
    
    async def upsert_vectors(
        self,
        vectors: List[Dict[str, Any]],
        namespace: str = "",
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Upsert vectors to Pinecone index with batching.
        
        Args:
            vectors: List of vector dictionaries with id, values, and metadata
            namespace: Optional namespace for organization
            batch_size: Number of vectors per batch
            
        Returns:
            Dict containing operation results and performance metrics
        """
        if not self.is_connected or not self.index:
            raise RuntimeError("Pinecone not connected")
        
        start_time = time.time()
        total_vectors = len(vectors)
        successful_upserts = 0
        
        try:
            # Process in batches for better performance
            for i in range(0, total_vectors, batch_size):
                batch = vectors[i:i + batch_size]
                
                try:
                    # Upsert batch
                    upsert_response = self.index.upsert(
                        vectors=batch,
                        namespace=namespace
                    )
                    
                    successful_upserts += upsert_response.upserted_count
                    
                    logger.debug(
                        "Batch upserted successfully",
                        batch_number=i // batch_size + 1,
                        batch_size=len(batch),
                        upserted_count=upsert_response.upserted_count,
                        namespace=namespace
                    )
                    
                except Exception as e:
                    logger.error(
                        "Failed to upsert batch",
                        batch_number=i // batch_size + 1,
                        batch_size=len(batch),
                        error=str(e),
                        namespace=namespace
                    )
                    # Continue with next batch
                    continue
            
            operation_time = time.time() - start_time
            self._track_operation_time(operation_time)
            
            logger.info(
                "Vector upsert completed",
                total_vectors=total_vectors,
                successful_upserts=successful_upserts,
                operation_time_ms=round(operation_time * 1000, 2),
                namespace=namespace
            )
            
            return {
                "total_vectors": total_vectors,
                "successful_upserts": successful_upserts,
                "failed_upserts": total_vectors - successful_upserts,
                "operation_time_ms": round(operation_time * 1000, 2),
                "namespace": namespace
            }
            
        except Exception as e:
            logger.error(
                "Vector upsert operation failed",
                error=str(e),
                total_vectors=total_vectors,
                namespace=namespace
            )
            raise
    
    async def query_vectors(
        self,
        query_vector: List[float],
        top_k: int = 10,
        namespace: str = "",
        filter_metadata: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
        include_values: bool = False
    ) -> Dict[str, Any]:
        """
        Query vectors for semantic search.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            namespace: Optional namespace to search within
            filter_metadata: Metadata filters to apply
            include_metadata: Whether to include metadata in results
            include_values: Whether to include vector values in results
            
        Returns:
            Dict containing search results and performance metrics
        """
        if not self.is_connected or not self.index:
            raise RuntimeError("Pinecone not connected")
        
        start_time = time.time()
        
        try:
            # Execute query
            query_response = self.index.query(
                vector=query_vector,
                top_k=top_k,
                namespace=namespace,
                filter=filter_metadata,
                include_metadata=include_metadata,
                include_values=include_values
            )
            
            operation_time = time.time() - start_time
            self._track_operation_time(operation_time)
            
            # Process results
            results = []
            for match in query_response.matches:
                result = {
                    "id": match.id,
                    "score": match.score,
                }
                
                if include_metadata and match.metadata:
                    result["metadata"] = match.metadata
                
                if include_values and match.values:
                    result["values"] = match.values
                
                results.append(result)
            
            logger.info(
                "Vector query completed",
                top_k=top_k,
                results_count=len(results),
                operation_time_ms=round(operation_time * 1000, 2),
                namespace=namespace
            )
            
            return {
                "results": results,
                "total_results": len(results),
                "operation_time_ms": round(operation_time * 1000, 2),
                "namespace": namespace
            }
            
        except Exception as e:
            logger.error(
                "Vector query operation failed",
                error=str(e),
                top_k=top_k,
                namespace=namespace
            )
            raise
    
    async def delete_vectors(
        self,
        vector_ids: List[str],
        namespace: str = ""
    ) -> Dict[str, Any]:
        """
        Delete vectors by IDs.
        
        Args:
            vector_ids: List of vector IDs to delete
            namespace: Optional namespace
            
        Returns:
            Dict containing deletion results
        """
        if not self.is_connected or not self.index:
            raise RuntimeError("Pinecone not connected")
        
        start_time = time.time()
        
        try:
            # Delete vectors
            self.index.delete(
                ids=vector_ids,
                namespace=namespace
            )
            
            operation_time = time.time() - start_time
            self._track_operation_time(operation_time)
            
            logger.info(
                "Vector deletion completed",
                deleted_count=len(vector_ids),
                operation_time_ms=round(operation_time * 1000, 2),
                namespace=namespace
            )
            
            return {
                "deleted_count": len(vector_ids),
                "operation_time_ms": round(operation_time * 1000, 2),
                "namespace": namespace
            }
            
        except Exception as e:
            logger.error(
                "Vector deletion failed",
                error=str(e),
                vector_ids_count=len(vector_ids),
                namespace=namespace
            )
            raise
    
    async def get_index_stats(self, namespace: str = "") -> Dict[str, Any]:
        """
        Get index statistics and health information.
        
        Args:
            namespace: Optional namespace to get stats for
            
        Returns:
            Dict containing index statistics
        """
        if not self.is_connected or not self.index:
            raise RuntimeError("Pinecone not connected")
        
        try:
            # Get index stats
            stats_response = self.index.describe_index_stats()
            
            # Calculate performance metrics
            avg_operation_time = (
                sum(self._operation_times) / len(self._operation_times)
                if self._operation_times else 0
            )
            
            stats = {
                "index_name": self.index_name,
                "dimension": self.dimension,
                "metric": self.metric,
                "total_vector_count": stats_response.total_vector_count,
                "namespaces": dict(stats_response.namespaces) if stats_response.namespaces else {},
                "index_fullness": stats_response.index_fullness,
                "performance_metrics": {
                    "avg_operation_time_ms": round(avg_operation_time * 1000, 2),
                    "total_operations": len(self._operation_times),
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            logger.debug(
                "Index stats retrieved",
                total_vectors=stats["total_vector_count"],
                index_fullness=stats["index_fullness"]
            )
            
            return stats
            
        except Exception as e:
            logger.error(
                "Failed to get index stats",
                error=str(e),
                index_name=self.index_name
            )
            raise
    
    def _track_operation_time(self, operation_time: float) -> None:
        """Track operation time for performance monitoring."""
        self._operation_times.append(operation_time)
        
        # Keep only last 1000 operations to prevent memory growth
        if len(self._operation_times) > 1000:
            self._operation_times = self._operation_times[-1000:]
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Pinecone service.
        
        Returns:
            Dict containing health status and performance metrics
        """
        start_time = time.time()
        
        try:
            if not self.is_connected or not self.index:
                return {
                    "status": "unhealthy",
                    "error": "Not connected to Pinecone",
                    "response_time_ms": 0
                }
            
            # Test with a simple describe operation
            stats = await self.get_index_stats()
            
            response_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "index_name": self.index_name,
                "vector_count": stats["total_vector_count"],
                "index_fullness": stats["index_fullness"],
                "response_time_ms": round(response_time * 1000, 2),
                "avg_operation_time_ms": stats["performance_metrics"]["avg_operation_time_ms"]
            }
            
        except Exception as e:
            response_time = time.time() - start_time
            
            return {
                "status": "unhealthy",
                "error": str(e),
                "response_time_ms": round(response_time * 1000, 2)
            }


# Global Pinecone service instance
pinecone_service = PineconeService()