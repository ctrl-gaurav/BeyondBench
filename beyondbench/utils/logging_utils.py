import logging
import os
import sys
from datetime import datetime

def setup_logging(output_dir=None, log_level="INFO", enable_file_logging=True, enable_console_logging=True, level=None):
    """Set up logging to file and console with enhanced error handling.

    Args:
        output_dir: Directory for log files. If None, file logging is disabled.
        log_level: Logging level string (DEBUG, INFO, WARNING, ERROR).
        enable_file_logging: Whether to log to a file.
        enable_console_logging: Whether to log to console.
        level: Alias for log_level (for convenience).
    """
    if level is not None:
        log_level = level
    if output_dir is None:
        enable_file_logging = False
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"eval_log_{timestamp}.txt") if output_dir else None

    # Clear any existing handlers to avoid conflicts
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create formatters
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # Convert log_level string to logging level
    log_level_obj = getattr(logging, log_level.upper(), logging.INFO)

    # Configure root logger
    logging.root.setLevel(log_level_obj)

    # File handler (if enabled and log_file is available)
    if enable_file_logging and log_file:
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(log_level_obj)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)

    # Console handler (if enabled)
    if enable_console_logging:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level_obj)
        console_handler.setFormatter(formatter)
        logging.root.addHandler(console_handler)

    # Test the logging setup
    if enable_file_logging:
        logging.info(f"🔧 Logging initialized - file: {log_file}")
    else:
        logging.info("🔧 Logging initialized - console only")

    return log_file

def get_logger(name):
    """Get a logger instance"""
    return logging.getLogger(name)