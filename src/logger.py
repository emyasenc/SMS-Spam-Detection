import os
import logging

def get_logger(log_path=None):
    """
    Initialize and return a logger. If a log path is provided, log to the file.
    If no log path is provided, log to the console.

    Args:
        log_path (str): Optional path to log to a file.
    
    Returns:
        logger: Configured logger instance.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Set up log output based on environment
    if log_path:
        # Log to a file if log_path is provided
        log_dir = os.path.dirname(log_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
    else:
        # Log to console if no log_path is provided
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    return logger

