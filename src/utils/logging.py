"""Logging utilities for the project."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> None:
    """Setup logging configuration for the project.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        format_string: Custom format string for log messages
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[
            logging.StreamHandler(sys.stdout),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    # Ensure basic configuration exists
    if not logging.getLogger().handlers:
        setup_logging()
        
    return logging.getLogger(name)


# Convenience loggers for common modules
service_logger = get_logger("service")
pipeline_logger = get_logger("pipeline") 
model_logger = get_logger("model")
dashboard_logger = get_logger("dashboard")
