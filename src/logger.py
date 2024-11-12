import os
import logging

def get_logger(log_path="logs/predict_pipeline.log"):
    # Check if we are in production or development environment
    if os.getenv("ENV") != "production":
        # Enable logging only in development
        log_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'predict_pipeline.log')
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        return logger
    else:
        # Return a no-op logger in production
        class NoOpLogger:
            def info(self, msg): pass
            def debug(self, msg): pass
            def error(self, msg): pass

        return NoOpLogger()

