"""
Cache Service for Crypto Intelligence System
Redis-based caching with in-memory fallback
"""
import json
import hashlib
import asyncio
from typing import Any, Optional, Dict
from datetime import datetime, timedelta
from functools import lru_cache

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from src.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CacheService:
    """
    Unified caching service with Redis support and in-memory fallback
    
    Features:
    - Redis for distributed caching
    - In-memory fallback when Redis unavailable
    - TTL-based expiration
    - JSON serialization for complex objects
    """
    
    def __init__(self, redis_url: str = None):
        """
        Initialize cache service
        
        Args:
            redis_url: Redis connection URL
        """
        self.settings = get_settings()
        self.redis_url = redis_url or self.settings.REDIS_URL
        
        self._redis: Optional[Any] = None
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._use_redis = False
        
        logger.info("cache_service_initialized")
    
    async def connect(self) -> bool:
        """
        Connect to Redis
        
        Returns:
            True if connected successfully
        """
        if not REDIS_AVAILABLE:
            logger.warning("redis_not_available, using_memory_cache")
            return False
        
        try:
            self._redis = aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            await self._redis.ping()
            self._use_redis = True
            logger.info("redis_connected")
            return True
        except Exception as e:
            logger.warning(f"redis_connection_failed: {e}, using_memory_cache")
            self._use_redis = False
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Redis"""
        if self._redis:
            await self._redis.close()
            self._redis = None
            self._use_redis = False
            logger.info("redis_disconnected")
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        if self._use_redis and self._redis:
            try:
                value = await self._redis.get(key)
                if value:
                    return json.loads(value)
            except Exception as e:
                logger.debug(f"redis_get_error: {e}")
        
        # Fallback to memory cache
        return self._memory_get(key)
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: int = 300
    ) -> bool:
        """
        Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            
        Returns:
            True if successful
        """
        serialized = json.dumps(value, default=str)
        
        if self._use_redis and self._redis:
            try:
                await self._redis.setex(key, ttl, serialized)
                return True
            except Exception as e:
                logger.debug(f"redis_set_error: {e}")
        
        # Fallback to memory cache
        return self._memory_set(key, value, ttl)
    
    async def delete(self, key: str) -> bool:
        """
        Delete value from cache
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted
        """
        if self._use_redis and self._redis:
            try:
                await self._redis.delete(key)
            except Exception as e:
                logger.debug(f"redis_delete_error: {e}")
        
        # Also delete from memory cache
        if key in self._memory_cache:
            del self._memory_cache[key]
            return True
        
        return False
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache
        
        Args:
            key: Cache key
            
        Returns:
            True if exists and not expired
        """
        if self._use_redis and self._redis:
            try:
                return await self._redis.exists(key) > 0
            except Exception as e:
                logger.debug(f"redis_exists_error: {e}")
        
        return self._memory_exists(key)
    
    async def clear(self) -> None:
        """Clear all cached data"""
        if self._use_redis and self._redis:
            try:
                await self._redis.flushdb()
            except Exception as e:
                logger.debug(f"redis_clear_error: {e}")
        
        self._memory_cache.clear()
        logger.info("cache_cleared")
    
    def _memory_get(self, key: str) -> Optional[Any]:
        """Get from in-memory cache"""
        if key not in self._memory_cache:
            return None
        
        entry = self._memory_cache[key]
        if datetime.now() > entry["expires_at"]:
            del self._memory_cache[key]
            return None
        
        return entry["value"]
    
    def _memory_set(self, key: str, value: Any, ttl: int) -> bool:
        """Set in in-memory cache"""
        self._memory_cache[key] = {
            "value": value,
            "expires_at": datetime.now() + timedelta(seconds=ttl)
        }
        
        # Cleanup old entries periodically
        if len(self._memory_cache) > 1000:
            self._cleanup_memory_cache()
        
        return True
    
    def _memory_exists(self, key: str) -> bool:
        """Check if key exists in memory cache"""
        if key not in self._memory_cache:
            return False
        
        entry = self._memory_cache[key]
        if datetime.now() > entry["expires_at"]:
            del self._memory_cache[key]
            return False
        
        return True
    
    def _cleanup_memory_cache(self) -> None:
        """Remove expired entries from memory cache"""
        now = datetime.now()
        expired_keys = [
            k for k, v in self._memory_cache.items()
            if now > v["expires_at"]
        ]
        
        for key in expired_keys:
            del self._memory_cache[key]
        
        logger.debug(f"cleaned_up_cache_entries", count=len(expired_keys))
    
    @staticmethod
    def make_key(*args, **kwargs) -> str:
        """
        Generate a cache key from arguments
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Hash-based cache key
        """
        key_parts = list(args)
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        key_string = ":".join(str(p) for p in key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()


# Global cache instance
_cache_instance: Optional[CacheService] = None


def get_cache() -> CacheService:
    """Get global cache instance"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = CacheService()
    return _cache_instance


async def init_cache() -> CacheService:
    """Initialize and connect cache"""
    cache = get_cache()
    await cache.connect()
    return cache


def cached(ttl: int = 300, key_prefix: str = ""):
    """
    Decorator for caching function results
    
    Args:
        ttl: Cache TTL in seconds
        key_prefix: Prefix for cache key
    
    Usage:
        @cached(ttl=60)
        async def expensive_function(arg1, arg2):
            ...
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            cache = get_cache()
            
            # Generate cache key
            key = f"{key_prefix}{func.__name__}:{CacheService.make_key(*args, **kwargs)}"
            
            # Try to get from cache
            result = await cache.get(key)
            if result is not None:
                return result
            
            # Call function
            result = await func(*args, **kwargs)
            
            # Cache result
            await cache.set(key, result, ttl)
            
            return result
        
        return wrapper
    return decorator
