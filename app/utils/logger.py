"""
Structured logging utility for production-grade logging
"""
import logging
import json
from datetime import datetime
from typing import Dict, Any
from app.config import get_settings


class StructuredLogger:
    """JSON-structured logger for production environments"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        settings = get_settings()
        
        # Set log level from config
        log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
        self.logger.setLevel(log_level)
        
        # Create console handler
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(handler)
    
    def _log(self, level: str, event: str, **kwargs):
        """Internal log method with structured format"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': level,
            'event': event,
            **kwargs
        }
        
        log_func = getattr(self.logger, level.lower())
        log_func(json.dumps(log_entry))
    
    def info(self, event: str, **kwargs):
        """Log info level event"""
        self._log('INFO', event, **kwargs)
    
    def warning(self, event: str, **kwargs):
        """Log warning level event"""
        self._log('WARNING', event, **kwargs)
    
    def error(self, event: str, **kwargs):
        """Log error level event"""
        self._log('ERROR', event, **kwargs)
    
    def debug(self, event: str, **kwargs):
        """Log debug level event"""
        self._log('DEBUG', event, **kwargs)
    
    def log_signal(self, signal: Dict[str, Any]):
        """Log trading signal generation"""
        self.info(
            'signal_generated',
            signal=signal.get('signal'),
            confidence=signal.get('confidence'),
            symbol=signal.get('symbol'),
            price=signal.get('price')
        )
    
    def log_api_call(self, service: str, endpoint: str, duration_ms: float, success: bool):
        """Log external API call"""
        self.info(
            'api_call',
            service=service,
            endpoint=endpoint,
            duration_ms=round(duration_ms, 2),
            success=success
        )
    
    def log_error_with_context(self, error: Exception, context: Dict[str, Any]):
        """Log error with additional context"""
        self.error(
            'error_occurred',
            error_type=type(error).__name__,
            error_message=str(error),
            **context
        )
    
    def error_with_context(self, error: Exception, context: Dict[str, Any]):
        """Alias for log_error_with_context for backward compatibility"""
        self.log_error_with_context(error, context)


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance"""
    return StructuredLogger(name)
