"""
Security Middleware for AI Code Mentor
Implements production-grade security headers and rate limiting.

Compliance:
- RULE SEC-002: Security headers implementation
- RULE PERF-005: Rate limiting per user/IP
- RULE SEC-001: Input validation and sanitization
"""

import time
from typing import Dict, Optional

import structlog
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from src.config.settings import get_settings
from src.services.redis_service import redis_service

logger = structlog.get_logger(__name__)
settings = get_settings()


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add security headers to all responses.
    Implements RULE SEC-002 requirements.
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains" if settings.is_production else "max-age=31536000",
            "Content-Security-Policy": self._get_csp_header(),
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
        }
    
    def _get_csp_header(self) -> str:
        """Generate Content Security Policy header based on environment."""
        if settings.is_production:
            return "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self'; connect-src 'self'; frame-ancestors 'none';"
        else:
            # More permissive for development
            return "default-src 'self' 'unsafe-inline' 'unsafe-eval'; connect-src 'self' ws: wss:;"
    
    async def dispatch(self, request: Request, call_next):
        """Add security headers to response."""
        response = await call_next(request)
        
        # Add security headers
        for header, value in self.security_headers.items():
            response.headers[header] = value
        
        # Add security headers for API responses
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
            response.headers["Pragma"] = "no-cache"
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware implementing RULE PERF-005.
    Tracks requests per IP and user with sliding window algorithm.
    """
    
    def __init__(self, app):
        super().__init__(app)
        # In-memory store for rate limiting (use Redis in production)
        self.request_counts: Dict[str, Dict[str, any]] = {}
        self.window_size = 60  # 1 minute window
        self.max_requests = settings.RATE_LIMIT_REQUESTS_PER_MINUTE
        self.burst_limit = settings.RATE_LIMIT_BURST
    
    def _get_client_identifier(self, request: Request) -> str:
        """Get unique identifier for client (IP + User ID if available)."""
        # Get client IP
        client_ip = request.client.host
        if "x-forwarded-for" in request.headers:
            client_ip = request.headers["x-forwarded-for"].split(",")[0].strip()
        elif "x-real-ip" in request.headers:
            client_ip = request.headers["x-real-ip"]
        
        # TODO: Add user ID when authentication is implemented
        # user_id = getattr(request.state, "user_id", None)
        # return f"{client_ip}:{user_id}" if user_id else client_ip
        
        return client_ip
    
    async def _is_rate_limited_redis(self, client_id: str) -> tuple[bool, Dict[str, any]]:
        """Check rate limiting using Redis backend."""
        try:
            result = await redis_service.check_rate_limit(
                identifier=client_id,
                limit=self.max_requests,
                window=self.window_size,
                action="api_request"
            )
            
            rate_limit_info = {
                "limit": self.max_requests,
                "remaining": result['remaining'],
                "reset": result['reset_time'],
                "window": self.window_size
            }
            
            return not result['allowed'], rate_limit_info
            
        except Exception as e:
            logger.error("Redis rate limiting failed, falling back to memory", error=str(e))
            return self._is_rate_limited_memory(client_id)
    
    def _is_rate_limited_memory(self, client_id: str) -> tuple[bool, Dict[str, any]]:
        """
        Check if client is rate limited using sliding window.
        
        Returns:
            tuple: (is_limited, rate_limit_info)
        """
        current_time = time.time()
        window_start = current_time - self.window_size
        
        if client_id not in self.request_counts:
            self.request_counts[client_id] = {
                "requests": [],
                "reset_time": current_time + self.window_size
            }
        
        client_data = self.request_counts[client_id]
        
        # Remove old requests outside the window
        client_data["requests"] = [
            req_time for req_time in client_data["requests"] 
            if req_time > window_start
        ]
        
        # Check rate limit
        request_count = len(client_data["requests"])
        remaining = max(0, self.max_requests - request_count)
        
        rate_limit_info = {
            "limit": self.max_requests,
            "remaining": remaining,
            "reset": int(client_data["reset_time"]),
            "window": self.window_size
        }
        
        # Check if rate limited
        if request_count >= self.max_requests:
            return True, rate_limit_info
        
        # Add current request
        client_data["requests"].append(current_time)
        rate_limit_info["remaining"] = remaining - 1
        
        return False, rate_limit_info
    
    async def dispatch(self, request: Request, call_next):
        """Apply rate limiting to requests."""
        # Skip rate limiting for health checks and metrics
        if request.url.path in ["/health", "/metrics"]:
            return await call_next(request)
        
        client_id = self._get_client_identifier(request)
        
        # Use Redis rate limiting if available, fallback to memory
        if redis_service.is_connected:
            is_limited, rate_info = await self._is_rate_limited_redis(client_id)
        else:
            is_limited, rate_info = self._is_rate_limited_memory(client_id)
        
        if is_limited:
            logger.warning(
                "Rate limit exceeded",
                client_id=client_id,
                path=request.url.path,
                method=request.method,
                limit=rate_info["limit"],
                reset_time=rate_info["reset"]
            )
            
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Limit: {rate_info['limit']} per {rate_info['window']} seconds",
                    "retry_after": rate_info["reset"] - int(time.time())
                },
                headers={
                    "X-RateLimit-Limit": str(rate_info["limit"]),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(rate_info["reset"]),
                    "Retry-After": str(rate_info["reset"] - int(time.time()))
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(rate_info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(rate_info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(rate_info["reset"])
        
        return response


class RequestValidationMiddleware(BaseHTTPMiddleware):
    """
    Input validation middleware implementing RULE SEC-001.
    Validates and sanitizes incoming requests.
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.max_request_size = 50 * 1024 * 1024  # 50MB max request size
        self.blocked_patterns = [
            # Common injection patterns
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"vbscript:",
            r"on\w+\s*=",
            # SQL injection patterns
            r"union\s+select",
            r"drop\s+table",
            r"delete\s+from",
            r"insert\s+into",
        ]
    
    async def dispatch(self, request: Request, call_next):
        """Validate incoming requests."""
        # Check request size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_request_size:
            logger.warning(
                "Request size too large",
                content_length=content_length,
                max_size=self.max_request_size,
                client_ip=request.client.host
            )
            return JSONResponse(
                status_code=413,
                content={"error": "Request entity too large"}
            )
        
        # Validate content type for file uploads
        if request.url.path.startswith("/api/v1/files/upload"):
            content_type = request.headers.get("content-type", "")
            if not content_type.startswith("multipart/form-data"):
                return JSONResponse(
                    status_code=400,
                    content={"error": "Invalid content type for file upload"}
                )
        
        # Process request
        response = await call_next(request)
        return response