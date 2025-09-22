"""
Health Check Utilities for AI Code Mentor
Implements RULE MON-001 with comprehensive system health monitoring.

Compliance:
- RULE MON-001: Health checks every 30 seconds
- RULE BACKUP-002: Disaster recovery monitoring
- RULE PERF-001: Database connection monitoring
"""

import asyncio
import time
from typing import Dict, Any, List
from datetime import datetime

import structlog
import asyncpg
import redis.asyncio as redis
import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from src.config.settings import get_settings
from src.services.redis_service import redis_service
from src.services.pinecone_service import pinecone_service

logger = structlog.get_logger(__name__)
settings = get_settings()


class HealthChecker:
    """Comprehensive health check system."""
    
    def __init__(self):
        self.checks = [
            self._check_database,
            self._check_redis,
            self._check_pinecone,
            self._check_disk_space,
            self._check_memory_usage,
            self._check_external_apis,
        ]
    
    async def _check_database(self) -> Dict[str, Any]:
        """Check database connectivity and performance."""
        start_time = time.time()
        
        try:
            # Create engine for health check
            engine = create_async_engine(
                settings.DATABASE_URL,
                pool_timeout=5,
                connect_args={"command_timeout": 5}
            )
            
            async with engine.begin() as conn:
                # Simple query to test connectivity
                result = await conn.execute(text("SELECT 1 as health_check"))
                row = result.fetchone()  # Remove await here
            
            await engine.dispose()
            
            response_time = (time.time() - start_time) * 1000
            
            return {
                "name": "database",
                "status": "healthy",
                "response_time_ms": round(response_time, 2),
                "details": {
                    "connection_successful": True,
                    "query_time_ms": round(response_time, 2),
                    "meets_sla": response_time < 100,  # RULE PERF-001
                }
            }
            
        except Exception as e:
            logger.error("Database health check failed", error=str(e))
            return {
                "name": "database",
                "status": "unhealthy",
                "error": str(e),
                "details": {
                    "connection_successful": False,
                }
            }
    
    async def _check_redis(self) -> Dict[str, Any]:
        """Check Redis connectivity and performance using our Redis service."""
        try:
            redis_health = await redis_service.health_check()
            
            if redis_health['status'] == 'healthy':
                return {
                    "name": "redis",
                    "status": "healthy",
                    "response_time_ms": redis_health.get('response_time_ms', 0),
                    "details": {
                        "connection_successful": True,
                        "service_available": True,
                        "meets_sla": redis_health.get('response_time_ms', 0) < 50,
                    }
                }
            else:
                return {
                    "name": "redis",
                    "status": "unhealthy",
                    "error": redis_health.get('error', 'Unknown error'),
                    "details": {
                        "connection_successful": False,
                        "service_available": False,
                    }
                }
                
        except Exception as e:
            logger.error("Redis health check failed", error=str(e))
            return {
                "name": "redis",
                "status": "unhealthy",
                "error": str(e),
                "details": {
                    "connection_successful": False,
                }
            }
    
    async def _check_pinecone(self) -> Dict[str, Any]:
        """Check Pinecone vector database connectivity and performance."""
        try:
            pinecone_health = await pinecone_service.health_check()
            
            if pinecone_health['status'] == 'healthy':
                return {
                    "name": "pinecone",
                    "status": "healthy",
                    "response_time_ms": pinecone_health.get('response_time_ms', 0),
                    "details": {
                        "connection_successful": True,
                        "service_available": True,
                        "index_name": pinecone_health.get('index_name'),
                        "vector_count": pinecone_health.get('vector_count'),
                        "meets_sla": pinecone_health.get('response_time_ms', 0) < 100,
                    }
                }
            else:
                return {
                    "name": "pinecone",
                    "status": "unhealthy",
                    "error": pinecone_health.get('error', 'Unknown error'),
                    "details": {
                        "connection_successful": False,
                        "service_available": False,
                    }
                }
                
        except Exception as e:
            logger.error("Pinecone health check failed", error=str(e))
            return {
                "name": "pinecone",
                "status": "unhealthy",
                "error": str(e),
                "details": {
                    "connection_successful": False,
                }
            }
    
    async def _check_disk_space(self) -> Dict[str, Any]:
        """Check disk space usage."""
        import shutil
        
        try:
            # Check disk space for upload directory
            total, used, free = shutil.disk_usage(settings.UPLOAD_DIR)
            
            usage_percent = (used / total) * 100
            is_healthy = usage_percent < 85  # RULE RESOURCE-001
            
            return {
                "name": "disk_space",
                "status": "healthy" if is_healthy else "warning",
                "details": {
                    "total_gb": round(total / (1024**3), 2),
                    "used_gb": round(used / (1024**3), 2),
                    "free_gb": round(free / (1024**3), 2),
                    "usage_percent": round(usage_percent, 2),
                    "within_limits": is_healthy,
                }
            }
            
        except Exception as e:
            logger.error("Disk space check failed", error=str(e))
            return {
                "name": "disk_space",
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage."""
        try:
            import psutil
        except ImportError:
            return {
                "name": "memory",
                "status": "skipped",
                "details": {
                    "reason": "psutil not installed",
                    "install_command": "pip install psutil"
                }
            }
        
        try:
            memory = psutil.virtual_memory()
            usage_percent = memory.percent
            is_healthy = usage_percent < 80  # RULE RESOURCE-001
            
            return {
                "name": "memory",
                "status": "healthy" if is_healthy else "warning",
                "details": {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "used_gb": round(memory.used / (1024**3), 2),
                    "usage_percent": round(usage_percent, 2),
                    "within_limits": is_healthy,
                }
            }
            
        except Exception as e:
            logger.error("Memory check failed", error=str(e))
            return {
                "name": "memory",
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def _check_external_apis(self) -> Dict[str, Any]:
        """Check external API connectivity."""
        checks = []
        
        # Check OpenAI API
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"}
                )
                
                checks.append({
                    "service": "openai",
                    "status": "healthy" if response.status_code == 200 else "unhealthy",
                    "response_code": response.status_code,
                })
                
        except Exception as e:
            checks.append({
                "service": "openai",
                "status": "unhealthy",
                "error": str(e)
            })
        
        # Check Pinecone API
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(
                    f"https://controller.{settings.PINECONE_ENVIRONMENT}.pinecone.io/actions/whoami",
                    headers={"Api-Key": settings.PINECONE_API_KEY}
                )
                
                checks.append({
                    "service": "pinecone",
                    "status": "healthy" if response.status_code == 200 else "unhealthy",
                    "response_code": response.status_code,
                })
                
        except Exception as e:
            checks.append({
                "service": "pinecone",
                "status": "unhealthy",
                "error": str(e)
            })
        
        # Overall status
        all_healthy = all(check["status"] == "healthy" for check in checks)
        
        return {
            "name": "external_apis",
            "status": "healthy" if all_healthy else "degraded",
            "details": {
                "services": checks,
                "all_services_healthy": all_healthy,
            }
        }


async def health_check() -> Dict[str, Any]:
    """
    Perform comprehensive health check.
    
    Returns:
        Dict containing health status and check results
    """
    start_time = time.time()
    checker = HealthChecker()
    
    try:
        # Run all health checks concurrently
        results = await asyncio.gather(
            *[check() for check in checker.checks],
            return_exceptions=True
        )
        
        # Process results
        checks = []
        overall_healthy = True
        critical_services_healthy = True
        
        for result in results:
            if isinstance(result, Exception):
                logger.error("Health check exception", error=str(result))
                checks.append({
                    "name": "unknown",
                    "status": "error",
                    "error": str(result)
                })
                overall_healthy = False
            else:
                checks.append(result)
                # Only critical services affect overall health
                # Redis and external APIs are considered optional for basic functionality
                if result["name"] in ["database"] and result["status"] not in ["healthy"]:
                    critical_services_healthy = False
                if result["status"] not in ["healthy", "warning", "skipped", "degraded"]:
                    overall_healthy = False
        
        # Override overall health - system is healthy if critical services are healthy
        overall_healthy = critical_services_healthy
        
        duration = (time.time() - start_time) * 1000
        
        health_data = {
            "healthy": overall_healthy,
            "timestamp": datetime.utcnow().isoformat(),
            "response_time_ms": round(duration, 2),
            "checks": checks,
            "environment": settings.ENVIRONMENT,
            "version": settings.VERSION,
        }
        
        # Log health check result
        if overall_healthy:
            logger.debug("Health check completed", duration_ms=duration)
        else:
            logger.warning(
                "Health check found issues",
                duration_ms=duration,
                failed_checks=[c["name"] for c in checks if c["status"] != "healthy"]
            )
        
        return health_data
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return {
            "healthy": False,
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
            "checks": [],
        }


async def deep_health_check() -> Dict[str, Any]:
    """
    Extended health check for detailed system monitoring.
    Used for comprehensive system status reporting.
    """
    basic_health = await health_check()
    
    # Add additional checks for deep monitoring
    try:
        import psutil
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Network stats
        net_io = psutil.net_io_counters()
        
        # Process info
        process = psutil.Process()
        process_info = {
            "cpu_percent": process.cpu_percent(),
            "memory_percent": process.memory_percent(),
            "num_threads": process.num_threads(),
            "num_fds": process.num_fds() if hasattr(process, "num_fds") else None,
        }
        
        basic_health["extended_metrics"] = {
            "system_cpu_percent": cpu_percent,
            "network_bytes_sent": net_io.bytes_sent,
            "network_bytes_recv": net_io.bytes_recv,
            "process_stats": process_info,
        }
        
    except Exception as e:
        logger.warning("Extended metrics collection failed", error=str(e))
    
    return basic_health