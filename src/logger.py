import os
import logging

def get_logger(log_path="logs/predict_pipeline.log"):
    # Get the absolute path for the log file based on the project root directory
    log_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'logs')  # Navigate to 'logs' folder

    # Ensure the logs directory exists
    os.makedirs(log_dir, exist_ok=True)  # Creates 'logs' if it doesn't exist

    # Full path for the log file
    log_file = os.path.join(log_dir, 'predict_pipeline.log')

    # Set up logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create file handler for logging to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Create console handler for logging to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Define the formatter for the log output
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
