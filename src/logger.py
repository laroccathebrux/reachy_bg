"""
Logging configuration for Eldritch Horror Reachy Agent

Provides structured logging across modules.
"""

import logging
import sys
from pathlib import Path
from .config import LOG_LEVEL, LOG_FILE

# Configure root logger
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE),
    ]
)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    Args:
        name: Module name (typically __name__)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    return logger


class StructuredLogger:
    """Wrapper for structured logging with context"""
    
    def __init__(self, name: str, context: dict = None):
        self.logger = get_logger(name)
        self.context = context or {}
    
    def set_context(self, **kwargs):
        """Update logging context"""
        self.context.update(kwargs)
    
    def _format_message(self, msg: str) -> str:
        """Add context to message"""
        if self.context:
            context_str = " | ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{msg} | {context_str}"
        return msg
    
    def info(self, msg: str, **kwargs):
        """Log info level"""
        self.set_context(**kwargs)
        self.logger.info(self._format_message(msg))
    
    def debug(self, msg: str, **kwargs):
        """Log debug level"""
        self.set_context(**kwargs)
        self.logger.debug(self._format_message(msg))
    
    def warning(self, msg: str, **kwargs):
        """Log warning level"""
        self.set_context(**kwargs)
        self.logger.warning(self._format_message(msg))
    
    def error(self, msg: str, **kwargs):
        """Log error level"""
        self.set_context(**kwargs)
        self.logger.error(self._format_message(msg))
    
    def critical(self, msg: str, **kwargs):
        """Log critical level"""
        self.set_context(**kwargs)
        self.logger.critical(self._format_message(msg))


__all__ = ["get_logger", "StructuredLogger"]
