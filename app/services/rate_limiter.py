"""
Token bucket rate limiter for API calls
"""
import asyncio
import time
from typing import Dict
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter for controlling request rates
    
    Allows bursts up to the rate limit, then throttles requests
    """
    
    def __init__(self, rate: int, per: int = 60, name: str = "unknown"):
        """
        Initialize rate limiter
        
        Args:
            rate: Number of requests allowed
            per: Time period in seconds (default: 60)
            name: Name of the service for logging
        """
        self.rate = rate
        self.per = per
        self.name = name
        self.tokens = float(rate)
        self.last_update = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """
        Wait until a token is available
        
        This method will block if no tokens are available,
        waiting for the token bucket to refill.
        Releases lock while sleeping to avoid blocking other coroutines.
        """
        while True:
            async with self.lock:
                now = time.time()
                elapsed = now - self.last_update
                
                # Refill tokens based on elapsed time
                self.tokens = min(
                    self.rate,
                    self.tokens + (elapsed * self.rate / self.per)
                )
                self.last_update = now
                
                # If we have tokens, consume one and return
                if self.tokens >= 1:
                    self.tokens -= 1
                    logger.debug(
                        'rate_limit_token_consumed',
                        service=self.name,
                        tokens_remaining=round(self.tokens, 2)
                    )
                    return
                
                # If no tokens, calculate wait time
                sleep_time = (1 - self.tokens) * self.per / self.rate
            
            # Sleep OUTSIDE the lock
            logger.debug(
                'rate_limit_waiting',
                service=self.name,
                wait_seconds=round(sleep_time, 2)
            )
            await asyncio.sleep(sleep_time)
    
    def get_available_tokens(self) -> float:
        """Get current number of available tokens"""
        now = time.time()
        elapsed = now - self.last_update
        tokens = min(
            self.rate,
            self.tokens + (elapsed * self.rate / self.per)
        )
        return tokens


# Global rate limiters for different services
rate_limiters: Dict[str, RateLimiter] = {
    'reddit': RateLimiter(60, 60, 'reddit'),
    'ccxt': RateLimiter(1200, 60, 'ccxt'),
    'blockchain': RateLimiter(20, 60, 'blockchain'),
    'groq': RateLimiter(30, 60, 'groq'),  # Groq API rate limit
}


def get_rate_limiter(service: str) -> RateLimiter:
    """Get rate limiter for a specific service"""
    if service not in rate_limiters:
        logger.warning(
            'rate_limiter_not_found',
            service=service,
            message=f'Creating default rate limiter for {service}'
        )
        rate_limiters[service] = RateLimiter(60, 60, service)
    return rate_limiters[service]
