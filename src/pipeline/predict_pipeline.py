import logging
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle
from src.logger import get_logger
import os

# Set log file path for local development or testing
log_file_path = os.path.join(os.getcwd(), 'logs', 'predict_pipeline.log')

# Get logger
logger = get_logger(log_file_path)  # Pass the log file path or None for console logging

def load_trained_model(filepath='/Users/sophia/Desktop/SMS-Spam-Detection/src/models/sms_spam_detector.h5'):
    """
    Loads a pre-trained model from a file.
    
    Args:
        filepath (str): The path to the saved model file.
        
    Returns:
        model: The loaded Keras model.
    """
    if logger:
        logger.info(f"Loading model from {filepath}")
    
    model = load_model(filepath)
    return model

def load_tokenizer(filepath='/Users/sophia/Desktop/SMS-Spam-Detection/src/models/tokenizer.pickle'):
    """
    Loads a pre-trained tokenizer from a file.
    
    Args:
        filepath (str): The path to the saved tokenizer file.
        
    Returns:
        tokenizer: The loaded Keras Tokenizer.
    """
    with open(filepath, 'rb') as file:
        tokenizer = pickle.load(file)
    return tokenizer

def make_predictions(model, tokenizer, texts, max_len=100):
    """
    Makes predictions on new data using the loaded model.
    
    Args:
        model: The trained Keras model.
        tokenizer: Keras Tokenizer instance used for preprocessing.
        texts (list): List of texts to predict.
        max_len (int): Maximum length of input sequences.
        
    Returns:
        predictions (list): List of predicted labels.
    """
    if logger:
        logger.info("Preprocessing texts for prediction")
    
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=max_len)
    
    if logger:
        logger.info("Making predictions")
    
    predictions = model.predict(X)
    return predictions

def predict(texts):
    """
    Function to handle the entire prediction process.
    
    Args:
        texts (list): List of texts to predict.
        
    Returns:
        predictions: Predictions for the input texts.
    """
    model = load_trained_model()
    tokenizer = load_tokenizer()
    return make_predictions(model, tokenizer, texts)