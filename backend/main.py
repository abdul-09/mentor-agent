"""
AI Code Mentor - Production-Grade FastAPI Application
Main application entry point with security, monitoring, and performance optimizations.

Compliance:
- RULE SEC-002: Security headers implementation
- RULE LOG-001: Structured logging with trace IDs
- RULE MON-001: Health checks and monitoring
- RULE PERF-004: Response time optimization
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any

import sentry_sdk
import structlog
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from prometheus_client import Counter, Histogram, generate_latest
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

from src.config.settings import get_settings
from src.api.v1.router import api_v1_router
from src.security.middleware import SecurityHeadersMiddleware, RateLimitMiddleware
from src.utils.logging import setup_logging
from src.utils.health import health_check

# Initialize settings
settings = get_settings()

# Setup structured logging (RULE LOG-001)
setup_logging()
logger = structlog.get_logger(__name__)

# Initialize Sentry for error tracking (RULE MON-001)
if settings.SENTRY_DSN:
    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        integrations=[
            FastApiIntegration(auto_enabling=True),
            SqlalchemyIntegration(),
        ],
        traces_sample_rate=settings.SENTRY_TRACES_SAMPLE_RATE,
        environment=settings.ENVIRONMENT,
    )

# Prometheus metrics for monitoring
REQUEST_COUNT = Counter(
    'http_requests_total', 
    'Total HTTP requests', 
    ['method', 'endpoint', 'status']
)
REQUEST_DURATION = Histogram(
    'http_request_duration_seconds', 
    'HTTP request duration', 
    ['method', 'endpoint']
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting AI Code Mentor API", version=settings.VERSION)
    
    # Initialize database connections, Redis, etc.
    # TODO: Add database initialization
    # TODO: Add Redis connection
    # TODO: Add Pinecone initialization
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Code Mentor API")
    # TODO: Close database connections
    # TODO: Close Redis connections


# Initialize FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="AI-powered code analysis and learning platform",
    version=settings.VERSION,
    docs_url="/api/docs" if settings.ENVIRONMENT != "production" else None,
    redoc_url="/api/redoc" if settings.ENVIRONMENT != "production" else None,
    openapi_url="/api/openapi.json" if settings.ENVIRONMENT != "production" else None,
    lifespan=lifespan,
)

# Security Middleware (RULE SEC-002)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RateLimitMiddleware)

# Trusted Host Middleware for production
if settings.ENVIRONMENT == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.ALLOWED_HOSTS
    )

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """
    Structured logging middleware with trace IDs (RULE LOG-001).
    Performance monitoring (RULE PERF-004).
    """
    # Generate trace ID for request tracking
    trace_id = str(uuid.uuid4())
    request.state.trace_id = trace_id
    
    # Start timer for performance monitoring
    start_time = time.time()
    
    # Log request
    logger.info(
        "HTTP request started",
        trace_id=trace_id,
        method=request.method,
        url=str(request.url),
        client_ip=request.client.host,
        user_agent=request.headers.get("user-agent"),
    )
    
    # Process request
    try:
        response = await call_next(request)
        
        # Calculate response time
        process_time = time.time() - start_time
        
        # Update Prometheus metrics
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        REQUEST_DURATION.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(process_time)
        
        # Log response
        logger.info(
            "HTTP request completed",
            trace_id=trace_id,
            status_code=response.status_code,
            response_time_ms=round(process_time * 1000, 2),
        )
        
        # Add trace ID to response headers
        response.headers["X-Trace-ID"] = trace_id
        
        return response
        
    except Exception as e:
        # Log error
        logger.error(
            "HTTP request failed",
            trace_id=trace_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        raise


# Health Check Endpoint (RULE MON-001)
@app.get("/health", tags=["health"])
async def health_endpoint():
    """
    Health check endpoint for monitoring systems.
    Returns system status and dependencies health.
    """
    try:
        health_status = await health_check()
        return {
            "status": "healthy" if health_status["healthy"] else "unhealthy",
            "timestamp": health_status["timestamp"],
            "version": settings.VERSION,
            "environment": settings.ENVIRONMENT,
            "checks": health_status["checks"]
        }
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service unavailable")


# Metrics endpoint for Prometheus
@app.get("/metrics", tags=["monitoring"])
async def metrics_endpoint():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )


# Include API routers (RULE API-001: Versioned under /api/v1/)
app.include_router(api_v1_router, prefix="/api/v1")


# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "AI Code Mentor API",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "docs_url": "/api/docs" if settings.ENVIRONMENT != "production" else None,
    }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with structured logging."""
    trace_id = getattr(request.state, "trace_id", "unknown")
    
    logger.error(
        "Unhandled exception",
        trace_id=trace_id,
        error=str(exc),
        error_type=type(exc).__name__,
        url=str(request.url),
        method=request.method,
    )
    
    return HTTPException(
        status_code=500,
        detail={
            "error": "Internal server error",
            "trace_id": trace_id,
            "timestamp": time.time(),
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.ENVIRONMENT == "development",
        log_config=None,  # Use our custom logging
    )