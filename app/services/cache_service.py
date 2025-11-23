"""
Redis-based caching service with async support
"""
import json
from typing import Dict, Any, Optional, Callable
from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Try to import redis, but make it optional
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning(
        'redis_not_available',
        message='Redis not available, using in-memory cache fallback'
    )


class CacheService:
    """
    Redis-based caching service with fallback to in-memory cache
    
    Implements cache-aside pattern for efficient data access
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.redis_client = None
        self.memory_cache: Dict[str, tuple[Any, float]] = {}  # key -> (value, expiry)
        
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(
                    self.settings.REDIS_URL,
                    decode_responses=True
                )
                logger.info('cache_service_initialized', backend='redis')
            except Exception as e:
                logger.error(
                    'redis_connection_failed',
                    error=str(e),
                    message='Falling back to in-memory cache'
                )
        else:
            logger.info('cache_service_initialized', backend='memory')
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get cached value by key
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        try:
            if self.redis_client:
                value = await self.redis_client.get(key)
                if value:
                    logger.debug('cache_hit', key=key, backend='redis')
                    return json.loads(value)
            else:
                # In-memory fallback
                import time
                if key in self.memory_cache:
                    value, expiry = self.memory_cache[key]
                    if time.time() < expiry:
                        logger.debug('cache_hit', key=key, backend='memory')
                        return value
                    else:
                        # Expired
                        del self.memory_cache[key]
            
            logger.debug('cache_miss', key=key)
            return None
            
        except Exception as e:
            logger.error('cache_get_error', key=key, error=str(e))
            return None
    
    async def set(self, key: str, value: Dict[str, Any], ttl: int):
        """
        Set cached value with TTL
        
        Args:
            key: Cache key
            value: Value to cache (must be JSON serializable)
            ttl: Time-to-live in seconds
        """
        try:
            if self.redis_client:
                await self.redis_client.setex(
                    key,
                    ttl,
                    json.dumps(value, default=str)
                )
                logger.debug('cache_set', key=key, ttl=ttl, backend='redis')
            else:
                # In-memory fallback
                import time
                expiry = time.time() + ttl
                self.memory_cache[key] = (value, expiry)
                logger.debug('cache_set', key=key, ttl=ttl, backend='memory')
                
        except Exception as e:
            logger.error('cache_set_error', key=key, error=str(e))
    
    async def get_or_compute(
        self,
        key: str,
        compute_fn: Callable,
        ttl: int
    ) -> Any:
        """
        Cache-aside pattern: get from cache or compute and store
        
        Args:
            key: Cache key
            compute_fn: Async function to compute value if not cached
            ttl: Cache TTL in seconds
            
        Returns:
            Cached or computed value
        """
        # Try cache first
        cached = await self.get(key)
        if cached is not None:
            return cached
        
        # Compute value
        logger.debug('cache_computing', key=key)
        value = await compute_fn()
        
        # Store in cache
        await self.set(key, value, ttl)
        
        return value
    
    async def invalidate(self, pattern: str):
        """
        Invalidate all keys matching pattern
        
        Args:
            pattern: Key pattern (e.g., 'signals:*')
        """
        try:
            if self.redis_client:
                keys = await self.redis_client.keys(pattern)
                if keys:
                    await self.redis_client.delete(*keys)
                    logger.info('cache_invalidated', pattern=pattern, count=len(keys))
            else:
                # In-memory: simple prefix matching
                keys_to_delete = [k for k in self.memory_cache.keys() if k.startswith(pattern.rstrip('*'))]
                for k in keys_to_delete:
                    del self.memory_cache[k]
                logger.info('cache_invalidated', pattern=pattern, count=len(keys_to_delete))
                
        except Exception as e:
            logger.error('cache_invalidate_error', pattern=pattern, error=str(e))
    
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info('cache_service_closed')


# Global cache instance
_cache_instance: Optional[CacheService] = None


def get_cache() -> CacheService:
    """Get global cache instance"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = CacheService()
    return _cache_instance
