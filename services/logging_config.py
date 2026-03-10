"""
Centralized logging configuration for the application.
Provides structured logging to both console and file with different verbosity levels.
"""

import logging
import logging.handlers
import os
from datetime import datetime


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    app_name: str = "pb-app"
) -> logging.Logger:
    """
    Configure application-wide logging with file and console handlers.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files
        app_name: Application name for log files
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(app_name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Remove any existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Formatter with timestamp and context
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler (INFO and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (DEBUG and above) - rotating to keep file size manageable
    log_file = os.path.join(log_dir, f"{app_name}.log")
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5  # Keep 5 backup files
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Token usage file handler (separate log for token tracking)
    token_log_file = os.path.join(log_dir, f"{app_name}_tokens.log")
    token_handler = logging.handlers.RotatingFileHandler(
        token_log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=5
    )
    token_formatter = logging.Formatter(
        fmt="%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    token_handler.setFormatter(token_formatter)
    
    # Create a separate logger for token tracking
    token_logger = logging.getLogger(f"{app_name}.tokens")
    token_logger.setLevel(logging.INFO)
    token_logger.addHandler(token_handler)
    # Don't propagate to main logger
    token_logger.propagate = False
    
    logger.info(f"Logging initialized. Level: {log_level}. Log dir: {log_dir}")
    
    return logger


# Module-level logger instance (lazy initialization)
_logger = None

def get_logger(name: str = "pb-app") -> logging.Logger:
    """
    Get or create the application logger.
    
    Args:
        name: Logger name (default: "pb-app")
        
    Returns:
        Logger instance
    """
    global _logger
    if _logger is None:
        _logger = setup_logging(app_name=name)
    return _logger


def get_token_logger(name: str = "pb-app") -> logging.Logger:
    """
    Get the token usage logger.
    
    Args:
        name: Logger name (default: "pb-app")
        
    Returns:
        Token usage logger instance
    """
    return logging.getLogger(f"{name}.tokens")
