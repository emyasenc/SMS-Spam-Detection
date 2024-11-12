import os
import logging

def get_logger(log_file="predict_pipeline.log"):
    # Ensure the logs directory exists
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, log_file)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # Create file handler for logging to a file
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    
    # Create console handler for logging to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

