import os
import logging

def get_logger(log_path=None):
    # If log_path is not provided, use the default relative path
    if log_path is None:
        log_path = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'predict_pipeline.log')

    # Commenting out directory creation and logger setup for now
    # log_dir = os.path.dirname(log_path)
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir, exist_ok=True)

    # Set up logging (commented out for deployment)
    # logger = logging.getLogger(__name__)
    # logger.setLevel(logging.DEBUG)
    
    # Create file handler
    # file_handler = logging.FileHandler(log_path)
    # file_handler.setLevel(logging.DEBUG)
    
    # Create console handler for logging to console
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.INFO)
    
    # Define formatter
    # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # file_handler.setFormatter(formatter)
    # console_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    # logger.addHandler(file_handler)
    # logger.addHandler(console_handler)
    
    # return logger
    return None  # Return None instead of the logger when logging is disabled

