"""
Advanced Rate Limiting System for AI Code Mentor
Implements endpoint-specific rate limiting with multiple strategies.

Compliance:
- RULE PERF-005: Advanced rate limiting per user/IP
- RULE SEC-002: DDoS protection
- RULE MON-001: Rate limiting metrics and monitoring
"""

import time
from functools import wraps
from typing import Dict, Optional, Callable, Any
from enum import Enum

import structlog
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse

from src.config.settings import get_settings
from src.services.redis_service import redis_service

logger = structlog.get_logger(__name__)
settings = get_settings()


class RateLimitStrategy(str, Enum):
    """Rate limiting strategies."""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    ADAPTIVE = "adaptive"


class RateLimitScope(str, Enum):
    """Rate limiting scopes."""
    IP = "ip"
    USER = "user"
    GLOBAL = "global"
    ENDPOINT = "endpoint"


class RateLimiter:
    """
    Advanced rate limiter with multiple strategies and scopes.
    """
    
    def __init__(self):
        self.memory_store: Dict[str, Dict[str, Any]] = {}
    
    async def check_rate_limit(
        self,
        identifier: str,
        limit: int,
        window: int,
        strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW,
        scope: RateLimitScope = RateLimitScope.IP
    ) -> Dict[str, Any]:
        """
        Check if request should be rate limited.
        
        Args:
            identifier: Client identifier (IP, user ID, etc.)
            limit: Number of requests allowed
            window: Time window in seconds
            strategy: Rate limiting strategy
            scope: Rate limiting scope
            
        Returns:
            Dict containing rate limit info
        """
        cache_key = f"rate_limit:{scope.value}:{identifier}"
        
        # Use Redis if available, fallback to memory
        if redis_service.is_connected:
            return await self._check_redis_rate_limit(
                cache_key, limit, window, strategy
            )
        else:
            return self._check_memory_rate_limit(
                cache_key, limit, window, strategy
            )
    
    async def _check_redis_rate_limit(
        self,
        cache_key: str,
        limit: int,
        window: int,
        strategy: RateLimitStrategy
    ) -> Dict[str, Any]:
        """Check rate limit using Redis backend."""
        try:
            current_time = int(time.time())
            
            if strategy == RateLimitStrategy.SLIDING_WINDOW:
                return await self._sliding_window_redis(cache_key, limit, window, current_time)
            elif strategy == RateLimitStrategy.FIXED_WINDOW:
                return await self._fixed_window_redis(cache_key, limit, window, current_time)
            elif strategy == RateLimitStrategy.TOKEN_BUCKET:
                return await self._token_bucket_redis(cache_key, limit, window, current_time)
            else:
                # Default to sliding window
                return await self._sliding_window_redis(cache_key, limit, window, current_time)
                
        except Exception as e:
            logger.error("Redis rate limiting failed", error=str(e))
            return self._check_memory_rate_limit(cache_key, limit, window, strategy)
    
    async def _sliding_window_redis(
        self,
        cache_key: str,
        limit: int,
        window: int,
        current_time: int
    ) -> Dict[str, Any]:
        """Sliding window rate limiting with Redis."""
        window_start = current_time - window
        
        # Remove old entries and count current requests
        await redis_service.redis.zremrangebyscore(cache_key, 0, window_start)
        current_count = await redis_service.redis.zcard(cache_key)
        
        if current_count >= limit:
            # Get oldest entry to determine reset time
            oldest_entries = await redis_service.redis.zrange(cache_key, 0, 0, withscores=True)
            reset_time = int(oldest_entries[0][1] + window) if oldest_entries else current_time + window
            
            return {
                "allowed": False,
                "limit": limit,
                "remaining": 0,
                "reset_time": reset_time,
                "retry_after": reset_time - current_time
            }
        
        # Add current request
        await redis_service.redis.zadd(cache_key, {str(current_time): current_time})
        await redis_service.redis.expire(cache_key, window)
        
        return {
            "allowed": True,
            "limit": limit,
            "remaining": limit - current_count - 1,
            "reset_time": current_time + window,
            "retry_after": 0
        }
    
    async def _fixed_window_redis(
        self,
        cache_key: str,
        limit: int,
        window: int,
        current_time: int
    ) -> Dict[str, Any]:
        """Fixed window rate limiting with Redis."""
        window_key = f"{cache_key}:{current_time // window}"
        
        current_count = await redis_service.redis.get(window_key)
        current_count = int(current_count) if current_count else 0
        
        if current_count >= limit:
            next_window = ((current_time // window) + 1) * window
            return {
                "allowed": False,
                "limit": limit,
                "remaining": 0,
                "reset_time": next_window,
                "retry_after": next_window - current_time
            }
        
        # Increment counter
        await redis_service.redis.incr(window_key)
        await redis_service.redis.expire(window_key, window)
        
        next_window = ((current_time // window) + 1) * window
        
        return {
            "allowed": True,
            "limit": limit,
            "remaining": limit - current_count - 1,
            "reset_time": next_window,
            "retry_after": 0
        }
    
    async def _token_bucket_redis(
        self,
        cache_key: str,
        limit: int,
        window: int,
        current_time: int
    ) -> Dict[str, Any]:
        """Token bucket rate limiting with Redis."""
        bucket_key = f"{cache_key}:bucket"
        last_refill_key = f"{cache_key}:last_refill"
        
        # Get current bucket state
        tokens = await redis_service.redis.get(bucket_key)
        last_refill = await redis_service.redis.get(last_refill_key)
        
        tokens = float(tokens) if tokens else limit
        last_refill = float(last_refill) if last_refill else current_time
        
        # Calculate tokens to add based on time elapsed
        time_elapsed = current_time - last_refill
        tokens_to_add = (time_elapsed / window) * limit
        tokens = min(limit, tokens + tokens_to_add)
        
        if tokens < 1:
            # Not enough tokens
            time_for_token = window / limit
            retry_after = int((1 - tokens) * time_for_token)
            
            return {
                "allowed": False,
                "limit": limit,
                "remaining": int(tokens),
                "reset_time": current_time + retry_after,
                "retry_after": retry_after
            }
        
        # Consume one token
        tokens -= 1
        
        # Update bucket state
        await redis_service.redis.setex(bucket_key, window * 2, str(tokens))
        await redis_service.redis.setex(last_refill_key, window * 2, str(current_time))
        
        return {
            "allowed": True,
            "limit": limit,
            "remaining": int(tokens),
            "reset_time": current_time + window,
            "retry_after": 0
        }
    
    def _check_memory_rate_limit(
        self,
        cache_key: str,
        limit: int,
        window: int,
        strategy: RateLimitStrategy
    ) -> Dict[str, Any]:
        """Fallback memory-based rate limiting."""
        current_time = time.time()
        
        if cache_key not in self.memory_store:
            self.memory_store[cache_key] = {
                "requests": [],
                "tokens": limit,
                "last_refill": current_time
            }
        
        data = self.memory_store[cache_key]
        
        if strategy == RateLimitStrategy.SLIDING_WINDOW:
            return self._sliding_window_memory(data, limit, window, current_time)
        elif strategy == RateLimitStrategy.TOKEN_BUCKET:
            return self._token_bucket_memory(data, limit, window, current_time)
        else:
            # Default to sliding window
            return self._sliding_window_memory(data, limit, window, current_time)
    
    def _sliding_window_memory(
        self,
        data: Dict[str, Any],
        limit: int,
        window: int,
        current_time: float
    ) -> Dict[str, Any]:
        """Memory-based sliding window."""
        window_start = current_time - window
        
        # Remove old requests
        data["requests"] = [req for req in data["requests"] if req > window_start]
        
        if len(data["requests"]) >= limit:
            oldest_request = min(data["requests"])
            reset_time = int(oldest_request + window)
            
            return {
                "allowed": False,
                "limit": limit,
                "remaining": 0,
                "reset_time": reset_time,
                "retry_after": reset_time - int(current_time)
            }
        
        # Add current request
        data["requests"].append(current_time)
        
        return {
            "allowed": True,
            "limit": limit,
            "remaining": limit - len(data["requests"]),
            "reset_time": int(current_time + window),
            "retry_after": 0
        }
    
    def _token_bucket_memory(
        self,
        data: Dict[str, Any],
        limit: int,
        window: int,
        current_time: float
    ) -> Dict[str, Any]:
        """Memory-based token bucket."""
        time_elapsed = current_time - data["last_refill"]
        tokens_to_add = (time_elapsed / window) * limit
        data["tokens"] = min(limit, data["tokens"] + tokens_to_add)
        data["last_refill"] = current_time
        
        if data["tokens"] < 1:
            time_for_token = window / limit
            retry_after = int((1 - data["tokens"]) * time_for_token)
            
            return {
                "allowed": False,
                "limit": limit,
                "remaining": int(data["tokens"]),
                "reset_time": int(current_time + retry_after),
                "retry_after": retry_after
            }
        
        data["tokens"] -= 1
        
        return {
            "allowed": True,
            "limit": limit,
            "remaining": int(data["tokens"]),
            "reset_time": int(current_time + window),
            "retry_after": 0
        }


# Global rate limiter instance
rate_limiter = RateLimiter()


def rate_limit(
    key: str,
    calls: int,
    period: int,
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW,
    scope: RateLimitScope = RateLimitScope.IP,
    per_user: bool = False
):
    """
    Rate limiting decorator for FastAPI endpoints.
    
    Args:
        key: Unique identifier for this rate limit
        calls: Number of calls allowed
        period: Time period in seconds
        strategy: Rate limiting strategy
        scope: Rate limiting scope
        per_user: If True, apply rate limiting per authenticated user
        
    Example:
        @rate_limit("login", calls=5, period=300)  # 5 calls per 5 minutes
        async def login_endpoint():
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request from arguments
            request = None
            current_user = None
            
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            # Look for current_user in kwargs
            if per_user and "current_user" in kwargs:
                current_user = kwargs["current_user"]
            
            if not request:
                # If no request found, skip rate limiting
                logger.warning("Rate limiting skipped - no request object found")
                return await func(*args, **kwargs)
            
            # Determine identifier
            if per_user and current_user:
                identifier = f"user:{current_user.id}"
                actual_scope = RateLimitScope.USER
            else:
                # Get client IP
                client_ip = request.client.host
                if "x-forwarded-for" in request.headers:
                    client_ip = request.headers["x-forwarded-for"].split(",")[0].strip()
                elif "x-real-ip" in request.headers:
                    client_ip = request.headers["x-real-ip"]
                
                identifier = f"ip:{client_ip}"
                actual_scope = scope
            
            # Add endpoint key to identifier
            full_identifier = f"{key}:{identifier}"
            
            # Check rate limit
            result = await rate_limiter.check_rate_limit(
                identifier=full_identifier,
                limit=calls,
                window=period,
                strategy=strategy,
                scope=actual_scope
            )
            
            if not result["allowed"]:
                logger.warning(
                    "Rate limit exceeded for endpoint",
                    endpoint=key,
                    identifier=identifier,
                    limit=calls,
                    period=period,
                    retry_after=result["retry_after"]
                )
                
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail={
                        "error": "Rate limit exceeded",
                        "message": f"Too many requests for {key}. Limit: {calls} per {period} seconds",
                        "retry_after": result["retry_after"],
                        "limit": result["limit"],
                        "remaining": result["remaining"],
                        "reset": result["reset_time"]
                    },
                    headers={
                        "X-RateLimit-Limit": str(result["limit"]),
                        "X-RateLimit-Remaining": str(result["remaining"]),
                        "X-RateLimit-Reset": str(result["reset_time"]),
                        "Retry-After": str(result["retry_after"])
                    }
                )
            
            # Add rate limit info to request state for response headers
            if hasattr(request, "state"):
                request.state.rate_limit_info = result
            
            # Execute the endpoint
            response = await func(*args, **kwargs)
            
            return response
        
        return wrapper
    return decorator


# Predefined rate limiters for common use cases
def auth_rate_limit(calls: int = 5, period: int = 300):
    """Rate limiter for authentication endpoints."""
    return rate_limit("auth", calls=calls, period=period, scope=RateLimitScope.IP)


def api_rate_limit(calls: int = 60, period: int = 60):
    """Rate limiter for general API endpoints."""
    return rate_limit("api", calls=calls, period=period, scope=RateLimitScope.IP)


def user_rate_limit(calls: int = 100, period: int = 3600):
    """Rate limiter per authenticated user."""
    return rate_limit("user_api", calls=calls, period=period, per_user=True)


def file_upload_rate_limit(calls: int = 10, period: int = 3600):
    """Rate limiter for file upload endpoints."""
    return rate_limit("file_upload", calls=calls, period=period, scope=RateLimitScope.IP)


def expensive_operation_rate_limit(calls: int = 3, period: int = 3600):
    """Rate limiter for expensive operations (analysis, AI calls)."""
    return rate_limit("expensive_ops", calls=calls, period=period, per_user=True)


# Rate limiting middleware for response headers
class RateLimitHeadersMiddleware:
    """Middleware to add rate limit headers to responses."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                # Check if request has rate limit info
                request = scope.get("request")
                if hasattr(request, "state") and hasattr(request.state, "rate_limit_info"):
                    rate_info = request.state.rate_limit_info
                    
                    # Add rate limit headers
                    headers = dict(message.get("headers", []))
                    headers[b"x-ratelimit-limit"] = str(rate_info["limit"]).encode()
                    headers[b"x-ratelimit-remaining"] = str(rate_info["remaining"]).encode()
                    headers[b"x-ratelimit-reset"] = str(rate_info["reset_time"]).encode()
                    
                    message["headers"] = list(headers.items())
            
            await send(message)
        
        return await self.app(scope, receive, send_wrapper)