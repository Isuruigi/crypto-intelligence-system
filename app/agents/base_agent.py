"""
Base agent class for all specialized agents
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
from app.utils.logger import get_logger


class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, name: str):
        """
        Initialize base agent
        
        Args:
            name: Agent name for logging
        """
        self.name = name
        self.logger = get_logger(f"agent.{name}")
    
    @abstractmethod
    async def analyze(self, **kwargs) -> Dict[str, Any]:
        """
        Analyze data and return results
        
        Returns:
            Dictionary containing analysis results
        """
        pass
    
    def _validate_output(self, output: Dict[str, Any], required_keys: list) -> bool:
        """
        Validate that output contains required keys
        
        Args:
            output: Output dictionary to validate
            required_keys: List of required key names
            
        Returns:
            True if valid, False otherwise
        """
        for key in required_keys:
            if key not in output:
                self.logger.error(
                    'invalid_output',
                    agent=self.name,
                    missing_key=key
                )
                return False
        return True
