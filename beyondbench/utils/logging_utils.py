import logging
import os
import sys
from datetime import datetime

def setup_logging(output_dir, log_level="INFO", enable_file_logging=True, enable_console_logging=True):
    """Set up logging to file and console with enhanced error handling"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"eval_log_{timestamp}.txt")

    # Clear any existing handlers to avoid conflicts
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create formatters
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # Convert log_level string to logging level
    log_level_obj = getattr(logging, log_level.upper(), logging.INFO)

    # Configure root logger
    logging.root.setLevel(log_level_obj)

    # File handler (if enabled)
    if enable_file_logging:
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