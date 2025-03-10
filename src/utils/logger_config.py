#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Logger Configuration Module
-------------------------
This module provides utilities for configuring logging in the application.
"""

import logging
import os
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(name: str, log_level: int = logging.INFO, log_file: str = None) -> logging.Logger:
    """
    Set up a logger with the specified name and configuration.
    
    Args:
        name: The name of the logger.
        log_level: The logging level (e.g., logging.INFO, logging.DEBUG).
        log_file: The path to the log file. If None, logs will only be sent to the console.
        
    Returns:
        A configured logger instance.
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is specified
    if log_file:
        # Create the directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_default_log_file() -> str:
    """
    Get the default log file path.
    
    Returns:
        The path to the default log file.
    """
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent
    
    # Create logs directory if it doesn't exist
    logs_dir = project_root / 'logs'
    logs_dir.mkdir(exist_ok=True)
    
    # Create a log file with the current timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = logs_dir / f'text_mining_{timestamp}.log'
    
    return str(log_file)

def configure_root_logger(log_level: int = logging.INFO, log_file: str = None) -> None:
    """
    Configure the root logger.
    
    Args:
        log_level: The logging level (e.g., logging.INFO, logging.DEBUG).
        log_file: The path to the log file. If None, a default log file will be used.
    """
    if log_file is None:
        log_file = get_default_log_file()
    
    # Configure the root logger
    setup_logger('', log_level, log_file)
    
    # Suppress verbose logging from libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('gensim').setLevel(logging.WARNING)
    logging.getLogger('nltk').setLevel(logging.WARNING)
    logging.getLogger('spacy').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('tensorflow').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    
    # Log the start of the application
    logging.getLogger('').info(f"Logging configured. Log file: {log_file}")

def get_logger(name: str, log_level: int = None) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: The name of the logger.
        log_level: The logging level. If None, the logger's current level will be used.
        
    Returns:
        A logger instance.
    """
    logger = logging.getLogger(name)
    
    if log_level is not None:
        logger.setLevel(log_level)
    
    return logger 