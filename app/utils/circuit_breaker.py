"""
Circuit breaker pattern implementation for external API resilience
"""
import asyncio
import time
from enum import Enum
from typing import Callable, Any
from app.utils.logger import get_logger

logger = get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, reject requests immediately
    - HALF_OPEN: After timeout, allow one request to test recovery
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        timeout: int = 60,
        expected_exception: type = Exception
    ):
        """
        Initialize circuit breaker
        
        Args:
            name: Name of the protected service
            failure_threshold: Number of failures before opening circuit
            timeout: Seconds before attempting recovery (HALF_OPEN)
            expected_exception: Exception type to catch
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self.lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection
        
        Args:
            func: Async function to execute
            *args, **kwargs: Arguments to pass to function
            
        Returns:
            Result from function
            
        Raises:
            Exception: If circuit is OPEN or function fails
        """
        async with self.lock:
            # Check if we should transition from OPEN to HALF_OPEN
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    logger.info(
                        'circuit_breaker_half_open',
                        service=self.name,
                        message='Attempting recovery'
                    )
                else:
                    logger.warning(
                        'circuit_breaker_open',
                        service=self.name,
                        message='Circuit breaker is OPEN, rejecting request'
                    )
                    raise Exception(f"Circuit breaker OPEN for {self.name}")
        
        # Try to execute the function
        try:
            result = await func(*args, **kwargs)
            
            # Success - reset failure count and close circuit
            async with self.lock:
                if self.state == CircuitState.HALF_OPEN:
                    logger.info(
                        'circuit_breaker_closed',
                        service=self.name,
                        message='Service recovered, circuit CLOSED'
                    )
                self.state = CircuitState.CLOSED
                self.failure_count = 0
            
            return result
            
        except self.expected_exception as e:
            async with self.lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                # Open circuit if threshold exceeded
                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitState.OPEN
                    logger.error(
                        'circuit_breaker_opened',
                        service=self.name,
                        failure_count=self.failure_count,
                        message=f'Circuit breaker OPENED after {self.failure_count} failures'
                    )
                else:
                    logger.warning(
                        'circuit_breaker_failure',
                        service=self.name,
                        failure_count=self.failure_count,
                        threshold=self.failure_threshold
                    )
            
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery"""
        if self.last_failure_time is None:
            return True
        return (time.time() - self.last_failure_time) >= self.timeout
    
    def get_state(self) -> str:
        """Get current circuit state"""
        return self.state.value
    
    def reset(self):
        """Manually reset the circuit breaker"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        logger.info('circuit_breaker_reset', service=self.name)
