"""
Logging Module
Provides structured logging capabilities for the Universal AI Optimizer
"""

import logging
import json
import sys
from typing import Dict, Any, Optional
from universal_ai_optimizer.core.base import BaseOptimizerModule

SENSITIVE_KEYS = {'api_key', 'password', 'token', 'secret', 'authorization', 'cookie'}

# Custom JSON formatter for structured logging
class JSONFormatter(logging.Formatter):
    """Formatter that outputs JSON strings after parsing the log record."""
    
    def format(self, record):
        log_entry = {
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra attributes from the record
        if hasattr(record, 'extra') and record.extra:
            log_entry.update(record.extra)
        
        # Add any other custom attributes with PII redaction
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                           'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                           'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                           'thread', 'threadName', 'processName', 'process', 'getMessage',
                           'message', 'asctime']:
                if not key.startswith('_'):
                    if key.lower() in SENSITIVE_KEYS:
                        log_entry[key] = '***REDACTED***'
                    else:
                        log_entry[key] = value
        
        # Also redact sensitive keys in nested dicts and extra
        if 'extra' in log_entry:
            extra = log_entry['extra']
            if isinstance(extra, dict):
                for extra_key in list(extra.keys()):
                    if extra_key.lower() in SENSITIVE_KEYS:
                        extra[extra_key] = '***REDACTED***'
        
        return json.dumps(log_entry, ensure_ascii=False)

class Logging(BaseOptimizerModule):
    """
    Enhanced logging module for the Universal AI Optimizer
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.enabled = self.config.get('enabled', True)
        self.level = self.config.get('level', 'INFO')
        self.format_type = self.config.get('format', 'json')  # json or text
        self.log_file = self.config.get('log_file', None)
        self.console_output = self.config.get('console_output', True)
        
        # Initialize logging
        self._setup_logging()
        
        # Context for structured logging
        self.context = {}
        
        self.logger = logging.getLogger('universal_ai_optimizer')
        self.logger.debug(f"Logging initialized with level={self.level}, format={self.format_type}")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        # Get the root logger for our module
        logger = logging.getLogger('universal_ai_optimizer')
        logger.setLevel(getattr(logging, self.level.upper()))
        
        # Clear any existing handlers
        logger.handlers.clear()
        
        # Create formatter
        if self.format_type.lower() == 'json':
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # Add console handler if enabled
        if self.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # Add file handler if specified
        if self.log_file:
            try:
                file_handler = logging.FileHandler(self.log_file)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except Exception as e:
                # Fallback to console if file logging fails
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)
                logger.error(f"Failed to setup file logging: {e}")
    
    def set_context(self, **kwargs):
        """Set context variables for structured logging"""
        self.context.update(kwargs)
    
    def clear_context(self):
        """Clear context variables"""
        self.context.clear()
    
    def _log_with_context(self, level: str, message: str, **kwargs):
        """Internal method to log with context"""
        if not self.enabled:
            return
        
        # Merge context with any additional kwargs
        extra = {**self.context, **kwargs}
        
        # Create a logger adapter to add context
        logger = logging.getLogger('universal_ai_optimizer')
        
        # Add extra to the record
        extra_record = {'extra': extra}
        
        # Log the message
        if level.upper() == 'DEBUG':
            logger.debug(message, extra=extra_record)
        elif level.upper() == 'INFO':
            logger.info(message, extra=extra_record)
        elif level.upper() == 'WARNING':
            logger.warning(message, extra=extra_record)
        elif level.upper() == 'ERROR':
            logger.error(message, extra=extra_record)
        elif level.upper() == 'CRITICAL':
            logger.critical(message, extra=extra_record)
    
    def debug(self, message: str, **kwargs):
        """Log a debug message"""
        self._log_with_context('DEBUG', message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log an info message"""
        self._log_with_context('INFO', message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log a warning message"""
        self._log_with_context('WARNING', message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log an error message"""
        self._log_with_context('ERROR', message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log a critical message"""
        self._log_with_context('CRITICAL', message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log an exception with traceback"""
        if not self.enabled:
            return
        
        extra = {**self.context, **kwargs}
        logger = logging.getLogger('universal_ai_optimizer')
        logger.exception(message, extra={'extra': extra})
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get logging metrics"""
        base_metrics = super().get_metrics()
        base_metrics.update({
            'enabled': self.enabled,
            'level': self.level,
            'format': self.format_type,
            'log_file': self.log_file,
            'console_output': self.console_output
        })
        return base_metrics