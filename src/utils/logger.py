"""
Structured Logging for Crypto Intelligence System
Uses loguru for structured, colored logging with context
"""
import sys
from typing import Any, Dict
from loguru import logger
from functools import lru_cache


# Remove default logger
logger.remove()


def _serialize_record(record: Dict[str, Any]) -> str:
    """Serialize log record to JSON format for production"""
    import json
    from datetime import datetime
    
    subset = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "message": record["message"],
        "module": record["module"],
        "function": record["function"],
        "line": record["line"],
    }
    
    # Add extra data if present
    if record.get("extra"):
        subset["extra"] = record["extra"]
    
    # Add exception info if present
    if record.get("exception"):
        subset["exception"] = str(record["exception"])
    
    return json.dumps(subset)


def setup_logger(
    level: str = "INFO",
    json_logs: bool = False,
    log_file: str = None
) -> None:
    """
    Configure the logger with appropriate settings
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_logs: If True, output JSON formatted logs (for production)
        log_file: Optional file path to write logs to
    """
    # Console output format
    if json_logs:
        format_string = "{extra[serialized]}"
        logger.add(
            sys.stdout,
            format=format_string,
            level=level,
            serialize=False,
            filter=lambda record: record["extra"].update(
                {"serialized": _serialize_record(record)}
            ) or True
        )
    else:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
        logger.add(
            sys.stdout,
            format=format_string,
            level=level,
            colorize=True
        )
    
    # File output if specified
    if log_file:
        logger.add(
            log_file,
            rotation="10 MB",
            retention="7 days",
            compression="gz",
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
        )


class Logger:
    """
    Wrapper around loguru logger with additional context methods
    """
    
    def __init__(self, name: str):
        self.name = name
        self._logger = logger.bind(module=name)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message"""
        self._logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message"""
        self._logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message"""
        self._logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message"""
        self._logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message"""
        self._logger.critical(message, **kwargs)
    
    def exception(self, message: str, **kwargs) -> None:
        """Log exception with traceback"""
        self._logger.exception(message, **kwargs)
    
    def error_with_context(self, error: Exception, context: Dict[str, Any]) -> None:
        """Log error with additional context"""
        self._logger.bind(**context).exception(str(error))
    
    def bind(self, **kwargs) -> "Logger":
        """Return logger with bound context"""
        new_logger = Logger(self.name)
        new_logger._logger = self._logger.bind(**kwargs)
        return new_logger


@lru_cache()
def get_logger(name: str = "crypto-intel") -> Logger:
    """
    Get a logger instance for the given module name
    
    Args:
        name: Module or component name
        
    Returns:
        Configured Logger instance
    """
    return Logger(name)


# Initialize default logger on module import
setup_logger(level="INFO", json_logs=False)
