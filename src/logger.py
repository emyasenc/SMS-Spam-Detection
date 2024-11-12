import logging
import os

def get_logger(log_file='logs/model_training.log'):
    """
    Creates and configures a logger for the SMS spam detection project.

    Args:
        log_file (str): The file path where logs should be saved.

    Returns:
        logging.Logger: Configured logger instance, or a disabled logger if logging is disabled.
    """
    # Check if logging should be disabled
    if os.getenv("DISABLE_LOGGING") == "1":
        # Return a logger that does nothing
        logger = logging.getLogger("null_logger")
        logger.addHandler(logging.NullHandler())
        return logger

    # Create and configure the logger as before
    logger = logging.getLogger('sms_spam_detector')
    logger.setLevel(logging.DEBUG)

    # Create file handler for logging to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Create console handler for logging to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Define the format for logging messages
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger if they haven't been added
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger