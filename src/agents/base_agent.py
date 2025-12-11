"""
Base Agent Abstract Class for Crypto Intelligence System
Defines the interface for all specialized agents
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime
import asyncio

from src.config import get_settings
from src.utils.logger import get_logger
from src.utils.metrics import timed, get_metrics, Timer

logger = get_logger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all AI agents in the system
    
    All agents must implement the analyze() method and follow
    the common interface for consistent orchestration.
    """
    
    def __init__(self, name: str):
        """
        Initialize base agent
        
        Args:
            name: Unique name for this agent
        """
        self.name = name
        self.settings = get_settings()
        self._last_run: Optional[datetime] = None
        self._run_count = 0
        self._error_count = 0
        
        logger.info(f"agent_initialized", agent=self.name)
    
    @abstractmethod
    async def analyze(self, symbol: str = "BTC/USDT", **kwargs) -> Dict[str, Any]:
        """
        Perform analysis for the given symbol
        
        Args:
            symbol: Trading pair to analyze
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with analysis results
        """
        pass
    
    async def safe_analyze(
        self, 
        symbol: str = "BTC/USDT", 
        timeout: int = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Safely run analysis with error handling and timeout
        
        Args:
            symbol: Trading pair to analyze
            timeout: Maximum seconds to wait
            **kwargs: Additional parameters
            
        Returns:
            Analysis results or error dict
        """
        timeout = timeout or self.settings.AGENT_TIMEOUT
        
        try:
            with Timer(f"agent_{self.name}_analyze"):
                result = await asyncio.wait_for(
                    self.analyze(symbol, **kwargs),
                    timeout=timeout
                )
            
            self._run_count += 1
            self._last_run = datetime.utcnow()
            get_metrics().increment(f"agent_{self.name}_success")
            
            return result
            
        except asyncio.TimeoutError:
            self._error_count += 1
            get_metrics().increment(f"agent_{self.name}_timeout")
            logger.error(f"agent_timeout", agent=self.name, timeout=timeout)
            return self._get_error_result("Timeout exceeded")
            
        except Exception as e:
            self._error_count += 1
            get_metrics().increment(f"agent_{self.name}_error")
            logger.error(f"agent_error", agent=self.name, error=str(e))
            return self._get_error_result(str(e))
    
    def _get_error_result(self, error: str) -> Dict[str, Any]:
        """Get standardized error result"""
        return {
            "success": False,
            "error": error,
            "agent": self.name,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _get_neutral_result(self) -> Dict[str, Any]:
        """Get neutral/default result when no data available"""
        return {
            "success": True,
            "data": {},
            "agent": self.name,
            "is_fallback": True,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            "name": self.name,
            "run_count": self._run_count,
            "error_count": self._error_count,
            "last_run": self._last_run.isoformat() if self._last_run else None,
            "success_rate": (
                (self._run_count - self._error_count) / self._run_count * 100
                if self._run_count > 0 else 0
            )
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
