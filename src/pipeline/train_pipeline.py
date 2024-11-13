import logging
from src.components.data_ingestion import load_data
from src.components.data_transformation import preprocess_data, split_data
from src.components.model_trainer import create_model, train_model, evaluate_model, save_model
from src.logger import get_logger
import os
from datetime import datetime

# Set log file path for local development or testing
log_file_path = os.path.join(os.getcwd(), 'logs', 'model_training.log')

# Get logger
logger = get_logger(log_file_path)  # Pass the log file path or None for console logging

def train_pipeline():
    """
    Runs the training pipeline for SMS spam detection.
    """
    filepath = '/Users/sophia/Desktop/SMS-Spam-Detection/data/raw/spam.csv'
    
    if logger:
        logger.info(f"Loading data from {filepath}")
    
    df = load_data(filepath)
    
    if logger:
        logger.info("Preprocessing data")
    
    X, y, tokenizer = preprocess_data(df)
    
    if logger:
        logger.info("Splitting data")
    
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    vocab_size = len(tokenizer.word_index) + 1
    model = create_model(vocab_size)
    
    if logger:
        logger.info("Training model")
    
    # Ensure that logger is passed correctly and not mistaken for an integer
    history = train_model(model, X_train, y_train, X_test, y_test, epochs=10, batch_size=32, logger=logger)
    
    if logger:
        logger.info("Evaluating model")
    
    metrics = evaluate_model(model, X_test, y_test, logger=logger)
    
    if logger:
        logger.info(f"Model Evaluation - Loss: {metrics['loss']}, Accuracy: {metrics['accuracy']}")
    
    if logger:
        logger.info("Saving model")
    
    save_model(model, logger=logger)

if __name__ == "__main__":
    train_pipeline()

