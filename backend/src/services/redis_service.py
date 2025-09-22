"""
Redis Service for AI Code Mentor
Handles caching, session management, and rate limiting with Redis.

Compliance:
- RULE SEC-002: Secure session management
- RULE PERF-003: Efficient caching strategies
- RULE LOG-001: Structured logging with trace IDs
"""

import asyncio
import json
import time
from typing import Optional, Any, Dict, List, Union
from datetime import datetime, timedelta

import redis.asyncio as redis
import structlog
from fastapi import HTTPException

from src.config.settings import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class RedisServiceError(Exception):
    """Custom exception for Redis service errors."""
    pass


class RedisService:
    """Redis service for caching, sessions, and rate limiting."""
    
    def __init__(self):
        """Initialize Redis service."""
        self.client: Optional[redis.Redis] = None
        self.connection_pool: Optional[redis.ConnectionPool] = None
        self.is_connected = False
        
        # Key prefixes for different data types
        self.prefixes = {
            'session': 'session:',
            'cache': 'cache:',
            'rate_limit': 'rate_limit:',
            'user_data': 'user:',
            'analysis': 'analysis:',
            'conversation': 'conversation:'
        }
    
    async def connect(self) -> None:
        """Establish Redis connection."""
        try:
            # Parse Redis URL
            redis_url = settings.REDIS_URL
            redis_password = getattr(settings, 'REDIS_PASSWORD', None)
            
            # Create connection pool
            if redis_password:
                self.connection_pool = redis.ConnectionPool.from_url(
                    redis_url,
                    password=redis_password,
                    max_connections=20,
                    retry_on_timeout=True,
                    socket_timeout=5,
                    socket_connect_timeout=5
                )
            else:
                self.connection_pool = redis.ConnectionPool.from_url(
                    redis_url,
                    max_connections=20,
                    retry_on_timeout=True,
                    socket_timeout=5,
                    socket_connect_timeout=5
                )
            
            # Create Redis client
            self.client = redis.Redis(connection_pool=self.connection_pool)
            
            # Test connection
            await self.client.ping()
            self.is_connected = True
            
            logger.info(
                "Redis connection established",
                redis_url=redis_url.split('@')[-1] if '@' in redis_url else redis_url,  # Hide credentials
                max_connections=20
            )
            
        except redis.ConnectionError as e:
            logger.error("Failed to connect to Redis", error=str(e))
            self.is_connected = False
            # Don't raise exception - allow app to start without Redis for development
            
        except Exception as e:
            logger.error("Unexpected error connecting to Redis", error=str(e))
            self.is_connected = False
    
    async def disconnect(self) -> None:
        """Close Redis connection."""
        try:
            if self.client:
                await self.client.close()
                self.is_connected = False
                logger.info("Redis connection closed")
        except Exception as e:
            logger.error("Error closing Redis connection", error=str(e))
    
    def _get_key(self, prefix: str, key: str) -> str:
        """Generate Redis key with prefix."""
        return f"{self.prefixes[prefix]}{key}"
    
    async def _ensure_connected(self) -> None:
        """Ensure Redis is connected, handle graceful degradation."""
        if not self.is_connected or not self.client:
            logger.warning("Redis not available, operation will be skipped")
            return False
        
        try:
            await self.client.ping()
            return True
        except Exception:
            self.is_connected = False
            logger.warning("Redis connection lost, operation will be skipped")
            return False
    
    # Session Management
    async def create_session(self, user_id: str, session_data: Dict[str, Any], ttl: int = 86400) -> str:
        """Create a new user session."""
        if not await self._ensure_connected():
            return f"fallback_session_{user_id}_{int(time.time())}"
        
        try:
            session_id = f"sess_{user_id}_{int(time.time())}"
            session_key = self._get_key('session', session_id)
            
            # Add metadata
            session_data.update({
                'user_id': user_id,
                'created_at': datetime.utcnow().isoformat(),
                'last_accessed': datetime.utcnow().isoformat()
            })
            
            await self.client.setex(
                session_key,
                ttl,
                json.dumps(session_data, default=str)
            )
            
            logger.info(
                "Session created",
                session_id=session_id,
                user_id=user_id,
                ttl=ttl
            )
            
            return session_id
            
        except Exception as e:
            logger.error("Failed to create session", error=str(e), user_id=user_id)
            return f"fallback_session_{user_id}_{int(time.time())}"
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve session data."""
        if not await self._ensure_connected():
            return None
        
        try:
            session_key = self._get_key('session', session_id)
            data = await self.client.get(session_key)
            
            if data:
                session_data = json.loads(data)
                
                # Update last accessed time
                session_data['last_accessed'] = datetime.utcnow().isoformat()
                await self.client.set(session_key, json.dumps(session_data, default=str), keepttl=True)
                
                return session_data
            
            return None
            
        except Exception as e:
            logger.error("Failed to get session", error=str(e), session_id=session_id)
            return None
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if not await self._ensure_connected():
            return True  # Assume success for fallback
        
        try:
            session_key = self._get_key('session', session_id)
            result = await self.client.delete(session_key)
            
            logger.info(
                "Session deleted",
                session_id=session_id,
                existed=bool(result)
            )
            
            return bool(result)
            
        except Exception as e:
            logger.error("Failed to delete session", error=str(e), session_id=session_id)
            return False
    
    # Caching
    async def cache_set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set a cached value."""
        if not await self._ensure_connected():
            return False
        
        try:
            cache_key = self._get_key('cache', key)
            
            # Serialize value
            if isinstance(value, (dict, list)):
                serialized_value = json.dumps(value, default=str)
            else:
                serialized_value = str(value)
            
            await self.client.setex(cache_key, ttl, serialized_value)
            
            logger.debug(
                "Cache set",
                key=key,
                ttl=ttl,
                value_type=type(value).__name__
            )
            
            return True
            
        except Exception as e:
            logger.error("Failed to set cache", error=str(e), key=key)
            return False
    
    async def cache_get(self, key: str, default: Any = None) -> Any:
        """Get a cached value."""
        if not await self._ensure_connected():
            return default
        
        try:
            cache_key = self._get_key('cache', key)
            data = await self.client.get(cache_key)
            
            if data:
                try:
                    # Try to parse as JSON
                    return json.loads(data)
                except json.JSONDecodeError:
                    # Return as string if not JSON
                    return data.decode('utf-8')
            
            return default
            
        except Exception as e:
            logger.error("Failed to get cache", error=str(e), key=key)
            return default
    
    async def cache_delete(self, key: str) -> bool:
        """Delete a cached value."""
        if not await self._ensure_connected():
            return True
        
        try:
            cache_key = self._get_key('cache', key)
            result = await self.client.delete(cache_key)
            return bool(result)
            
        except Exception as e:
            logger.error("Failed to delete cache", error=str(e), key=key)
            return False
    
    # Rate Limiting
    async def check_rate_limit(
        self, 
        identifier: str, 
        limit: int, 
        window: int,
        action: str = "request"
    ) -> Dict[str, Any]:
        """
        Check if rate limit is exceeded using sliding window.
        
        Args:
            identifier: Unique identifier (user_id, IP, etc.)
            limit: Maximum requests allowed
            window: Time window in seconds
            action: Action type for logging
            
        Returns:
            Dict with allowed, remaining, reset_time
        """
        if not await self._ensure_connected():
            # Allow requests when Redis is not available
            return {
                'allowed': True,
                'remaining': limit,
                'reset_time': int(time.time()) + window,
                'fallback': True
            }
        
        try:
            rate_key = self._get_key('rate_limit', f"{action}:{identifier}")
            current_time = int(time.time())
            window_start = current_time - window
            
            # Use Redis pipeline for atomic operations
            pipe = self.client.pipeline()
            
            # Remove old entries
            pipe.zremrangebyscore(rate_key, 0, window_start)
            
            # Count current requests in window
            pipe.zcard(rate_key)
            
            # Add current request
            pipe.zadd(rate_key, {str(current_time): current_time})
            
            # Set expiration
            pipe.expire(rate_key, window)
            
            results = await pipe.execute()
            current_count = results[1] + 1  # +1 for the request we just added
            
            allowed = current_count <= limit
            remaining = max(0, limit - current_count)
            reset_time = current_time + window
            
            if not allowed:
                logger.warning(
                    "Rate limit exceeded",
                    identifier=identifier,
                    action=action,
                    current_count=current_count,
                    limit=limit,
                    window=window
                )
            
            return {
                'allowed': allowed,
                'remaining': remaining,
                'reset_time': reset_time,
                'current_count': current_count
            }
            
        except Exception as e:
            logger.error(
                "Rate limit check failed",
                error=str(e),
                identifier=identifier,
                action=action
            )
            # Allow requests when rate limiting fails
            return {
                'allowed': True,
                'remaining': limit,
                'reset_time': int(time.time()) + window,
                'error': True
            }
    
    # Conversation History
    async def save_conversation_message(
        self, 
        conversation_id: str, 
        message: Dict[str, Any],
        ttl: int = 604800  # 7 days
    ) -> bool:
        """Save a conversation message."""
        if not await self._ensure_connected():
            return False
        
        try:
            conv_key = self._get_key('conversation', conversation_id)
            message['timestamp'] = datetime.utcnow().isoformat()
            
            # Add message to list
            await self.client.lpush(conv_key, json.dumps(message, default=str))
            
            # Limit conversation history (keep last 100 messages)
            await self.client.ltrim(conv_key, 0, 99)
            
            # Set expiration
            await self.client.expire(conv_key, ttl)
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to save conversation message",
                error=str(e),
                conversation_id=conversation_id
            )
            return False
    
    async def get_conversation_history(
        self, 
        conversation_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get conversation history."""
        if not await self._ensure_connected():
            return []
        
        try:
            conv_key = self._get_key('conversation', conversation_id)
            messages = await self.client.lrange(conv_key, 0, limit - 1)
            
            return [json.loads(msg) for msg in messages]
            
        except Exception as e:
            logger.error(
                "Failed to get conversation history",
                error=str(e),
                conversation_id=conversation_id
            )
            return []
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for Redis service."""
        try:
            if not self.is_connected or not self.client:
                return {
                    'status': 'unhealthy',
                    'error': 'Not connected',
                    'timestamp': datetime.utcnow().isoformat()
                }
            
            # Test basic operations
            start_time = time.time()
            await self.client.ping()
            
            # Test set/get
            test_key = 'health_check_test'
            await self.client.setex(test_key, 10, 'test_value')
            value = await self.client.get(test_key)
            await self.client.delete(test_key)
            
            response_time = time.time() - start_time
            
            if value == b'test_value':
                return {
                    'status': 'healthy',
                    'response_time_ms': round(response_time * 1000, 2),
                    'timestamp': datetime.utcnow().isoformat()
                }
            else:
                return {
                    'status': 'unhealthy',
                    'error': 'Test operation failed',
                    'timestamp': datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def get_user_conversations(self, user_id: str) -> List[str]:
        """Get list of conversation IDs for a user."""
        if not await self._ensure_connected():
            return []
        
        try:
            # Search for conversation keys that contain messages from this user
            # Note: This is a simplified implementation
            # In production, you might want to maintain a user->conversations mapping
            
            pattern = f"{self.prefixes['conversation']}*"
            conversation_keys = []
            
            # Get all conversation keys
            async for key in self.client.scan_iter(match=pattern):
                conv_id = key.decode('utf-8').replace(self.prefixes['conversation'], '')
                
                # Check if user has messages in this conversation
                try:
                    messages = await self.client.lrange(key, 0, -1)
                    for msg_data in messages:
                        msg = json.loads(msg_data)
                        if msg.get('user_id') == user_id:
                            conversation_keys.append(conv_id)
                            break
                except (json.JSONDecodeError, Exception):
                    continue
            
            # Sort by most recent activity (this is simplified)
            return conversation_keys
            
        except Exception as e:
            logger.error(
                "Failed to get user conversations",
                error=str(e),
                user_id=user_id
            )
            return []
    
    async def delete_conversations_for_user(self, user_id: str) -> bool:
        """Delete all conversations for a user."""
        if not await self._ensure_connected():
            return True
        
        try:
            user_conversations = await self.get_user_conversations(user_id)
            
            for conv_id in user_conversations:
                conv_key = self._get_key('conversation', conv_id)
                await self.client.delete(conv_key)
            
            logger.info(
                "User conversations deleted",
                user_id=user_id,
                count=len(user_conversations)
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to delete user conversations",
                error=str(e),
                user_id=user_id
            )
            return False
    
    async def set_json(self, key: str, value: Dict[str, Any], ttl: int = 3600) -> bool:
        """Set JSON value."""
        if not await self._ensure_connected():
            return False
        
        try:
            serialized = json.dumps(value, default=str)
            cache_key = self._get_key('cache', key)
            await self.client.setex(cache_key, ttl, serialized)
            return True
            
        except Exception as e:
            logger.error("Failed to set JSON value", error=str(e), key=key)
            return False
    
    async def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        """Get JSON value."""
        if not await self._ensure_connected():
            return None
        
        try:
            cache_key = self._get_key('cache', key)
            data = await self.client.get(cache_key)
            
            if data:
                return json.loads(data)
            
            return None
            
        except Exception as e:
            logger.error("Failed to get JSON value", error=str(e), key=key)
            return None


# Global Redis service instance
redis_service = RedisService()