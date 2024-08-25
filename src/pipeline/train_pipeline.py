import logging
from src.components.data_ingestion import load_data
from src.components.data_transformation import preprocess_data, split_data
from src.components.model_trainer import create_model, train_model, evaluate_model, save_model
from src.logger import get_logger
from datetime import datetime

# Initialize logger using get_logger
logger = get_logger('/Users/sophia/Desktop/SMS-Spam-Detection/logs/model_training.log')

def train_pipeline():
    """
    Runs the training pipeline for SMS spam detection.
    """
    filepath = '/Users/sophia/Desktop/SMS-Spam-Detection/data/raw/spam.csv'
    
    logger.info(f"Loading data from {filepath}")
    df = load_data(filepath)
    
    logger.info("Preprocessing data")
    X, y, tokenizer = preprocess_data(df)
    
    logger.info("Splitting data")
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    vocab_size = len(tokenizer.word_index) + 1
    model = create_model(vocab_size)
    
    logger.info("Training model")
    # Ensure that logger is passed correctly and not mistaken for an integer
    history = train_model(model, X_train, y_train, X_test, y_test, epochs=10, batch_size=32, logger=logger)
    
    logger.info("Evaluating model")
    metrics = evaluate_model(model, X_test, y_test, logger=logger)
    
    logger.info(f"Model Evaluation - Loss: {metrics['loss']}, Accuracy: {metrics['accuracy']}")
    
    logger.info("Saving model")
    save_model(model, logger=logger)

if __name__ == "__main__":
    train_pipeline()
