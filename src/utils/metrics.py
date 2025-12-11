"""
Performance Metrics Tracking for Crypto Intelligence System
Provides timing, counting, and monitoring utilities
"""
import time
import asyncio
from typing import Callable, Any, Dict, Optional
from functools import wraps
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import threading


@dataclass
class MetricPoint:
    """Single metric data point"""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    Collects and stores performance metrics
    Thread-safe implementation for concurrent access
    """
    
    def __init__(self):
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._timings: Dict[str, list] = defaultdict(list)
        self._lock = threading.Lock()
    
    def increment(self, name: str, value: int = 1, tags: Dict[str, str] = None) -> None:
        """Increment a counter metric"""
        with self._lock:
            key = self._make_key(name, tags)
            self._counters[key] += value
    
    def gauge(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        """Set a gauge metric (current value)"""
        with self._lock:
            key = self._make_key(name, tags)
            self._gauges[key] = value
    
    def timing(self, name: str, value_ms: float, tags: Dict[str, str] = None) -> None:
        """Record a timing metric in milliseconds"""
        with self._lock:
            key = self._make_key(name, tags)
            self._timings[key].append(value_ms)
            # Keep only last 1000 timings per metric
            if len(self._timings[key]) > 1000:
                self._timings[key] = self._timings[key][-1000:]
    
    def get_counter(self, name: str, tags: Dict[str, str] = None) -> int:
        """Get counter value"""
        key = self._make_key(name, tags)
        return self._counters.get(key, 0)
    
    def get_gauge(self, name: str, tags: Dict[str, str] = None) -> Optional[float]:
        """Get gauge value"""
        key = self._make_key(name, tags)
        return self._gauges.get(key)
    
    def get_timing_stats(self, name: str, tags: Dict[str, str] = None) -> Dict[str, float]:
        """Get timing statistics (min, max, avg, p50, p95, p99)"""
        key = self._make_key(name, tags)
        timings = self._timings.get(key, [])
        
        if not timings:
            return {}
        
        sorted_timings = sorted(timings)
        n = len(sorted_timings)
        
        return {
            "count": n,
            "min": min(sorted_timings),
            "max": max(sorted_timings),
            "avg": sum(sorted_timings) / n,
            "p50": sorted_timings[int(n * 0.5)],
            "p95": sorted_timings[int(n * 0.95)] if n >= 20 else sorted_timings[-1],
            "p99": sorted_timings[int(n * 0.99)] if n >= 100 else sorted_timings[-1],
        }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics"""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "timings": {
                    k: self.get_timing_stats(k) 
                    for k in self._timings.keys()
                }
            }
    
    def reset(self) -> None:
        """Reset all metrics"""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._timings.clear()
    
    @staticmethod
    def _make_key(name: str, tags: Optional[Dict[str, str]]) -> str:
        """Create a unique key from name and tags"""
        if not tags:
            return name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}{{{tag_str}}}"


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    """Get global metrics collector instance"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def timed(metric_name: str = None, tags: Dict[str, str] = None):
    """
    Decorator to time function execution
    
    Args:
        metric_name: Name for the timing metric (defaults to function name)
        tags: Additional tags for the metric
    
    Usage:
        @timed("api_request")
        def my_function():
            pass
    """
    def decorator(func: Callable) -> Callable:
        name = metric_name or f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                get_metrics().timing(name, elapsed_ms, tags)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                get_metrics().timing(name, elapsed_ms, tags)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def counted(metric_name: str = None, tags: Dict[str, str] = None):
    """
    Decorator to count function calls
    
    Args:
        metric_name: Name for the counter metric (defaults to function name)
        tags: Additional tags for the metric
    """
    def decorator(func: Callable) -> Callable:
        name = metric_name or f"{func.__module__}.{func.__name__}_calls"
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            get_metrics().increment(name, tags=tags)
            return func(*args, **kwargs)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            get_metrics().increment(name, tags=tags)
            return await func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


class Timer:
    """
    Context manager for timing code blocks
    
    Usage:
        with Timer("my_operation") as timer:
            do_something()
        print(f"Took {timer.elapsed_ms}ms")
    """
    
    def __init__(self, name: str, tags: Dict[str, str] = None, record: bool = True):
        self.name = name
        self.tags = tags
        self.record = record
        self.start_time: float = 0
        self.end_time: float = 0
        self.elapsed_ms: float = 0
    
    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args) -> None:
        self.end_time = time.perf_counter()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000
        if self.record:
            get_metrics().timing(self.name, self.elapsed_ms, self.tags)
    
    async def __aenter__(self) -> "Timer":
        return self.__enter__()
    
    async def __aexit__(self, *args) -> None:
        return self.__exit__(*args)
