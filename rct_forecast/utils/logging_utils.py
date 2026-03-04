"""
Logging utilities for RCT forecast project
"""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional


def setup_logging(config_manager, logger_name: str = "rct_forecast") -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        config_manager: Configuration manager instance
        logger_name: Name of the logger
        
    Returns:
        Configured logger
    """
    
    # Get logging configuration
    log_level = config_manager.get('logging.level', 'INFO').upper()
    log_format = config_manager.get('logging.format', 
                                   '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = config_manager.get('logging.file', 'logs/rct_forecast.log')
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, log_level))
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5  # 10MB files, keep 5 backups
    )
    file_handler.setLevel(getattr(logging, log_level))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    logger.info(f"Logging setup complete. Level: {log_level}, File: {log_file}")
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger by name"""
    return logging.getLogger(f"rct_forecast.{name}")
