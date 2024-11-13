import logging
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle
import os
from src.logger import get_logger

# Set log file path for local development or testing
log_file_path = os.path.join(os.getcwd(), 'logs', 'predict_pipeline.log')

# Get logger
logger = get_logger(log_file_path)  # Pass the log file path or None for console logging

def load_trained_model():
    """
    Loads a pre-trained model from the 'models' directory inside the 'src' folder.
    """
    # Build the relative path to the model file (corrected name)
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'sms_spam_detector.h5')

    # Log the model path for debugging purposes
    if logger:
        logger.info(f"Loading model from {model_path}")

    # Check if the model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Load and return the model
    model = load_model(model_path)
    return model

def load_tokenizer(filepath=None):
    """
    Loads a pre-trained tokenizer from a file.
    
    Args:
        filepath (str): The path to the saved tokenizer file.
        
    Returns:
        tokenizer: The loaded Keras Tokenizer.
    """
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), '..', 'models', 'tokenizer.pickle')
    
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
    
    # Convert texts to sequences and pad them
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
    # Load the model and tokenizer
    model = load_trained_model()
    tokenizer = load_tokenizer()
    
    # Make predictions and return the results
    return make_predictions(model, tokenizer, texts)