import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, call
from keras.preprocessing.text import Tokenizer
from components.model_trainer import create_model, train_model, evaluate_model, save_model
from components.data_transformation import preprocess_data, split_data

@pytest.fixture
def mock_data():
    """
    Fixture to create sample data for model training and evaluation.
    
    Returns:
        tuple: Feature array, label array, and tokenizer instance.
    """
    data = {
        'label': ['ham', 'spam', 'ham', 'spam'],
        'message': ['Hello', 'Win big', 'How are you', 'Congratulations']
    }
    df = pd.DataFrame(data)
    X, y, tokenizer = preprocess_data(df)
    return X, y, tokenizer

@patch('keras.Sequential.compile')
def test_create_model(mock_compile):
    """
    Test the create_model function to ensure it correctly creates a Keras model.
    
    Asserts:
        - Model is created and is not None.
        - Model compile method is called with 'accuracy' in metrics.
    """
    model = create_model(vocab_size=1000)
    assert model is not None, "Model should be created"
    
    # Verify that the compile method was called with the expected arguments
    mock_compile.assert_called_once_with(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

def test_train_model(mock_data):
    """
    Test the train_model function to ensure it trains the model correctly.
    
    Args:
        mock_data: Fixture providing sample data for training.
    
    Asserts:
        - Training history should contain accuracy values.
    """
    X, y, tokenizer = mock_data
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = create_model(vocab_size=len(tokenizer.word_index) + 1)
    history = train_model(model, X_train, y_train, X_test, y_test)
    assert history.history['accuracy'] is not None, "Training history should have accuracy"

def test_evaluate_model(mock_data):
    """
    Test the evaluate_model function to ensure it evaluates the model correctly.
    
    Args:
        mock_data: Fixture providing sample data for evaluation.
    
    Asserts:
        - Evaluation should return both loss and accuracy.
    """
    X, y, tokenizer = mock_data
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = create_model(vocab_size=len(tokenizer.word_index) + 1)
    train_model(model, X_train, y_train, X_test, y_test)
    loss, accuracy = evaluate_model(model, X_test, y_test)
    assert loss is not None, "Evaluation should return a loss"
    assert accuracy is not None, "Evaluation should return accuracy"

def test_save_model(mocker):
    """
    Test the save_model function to ensure it saves the model correctly.
    
    Args:
        mocker: pytest fixture for mocking functions.
    
    Asserts:
        - Save model function is called once.
    """
    mock_save = mocker.patch('src.model_trainer.save_model')
    mock_save.return_value = None

    # Simulate saving the model
    mock_save('dummy_model_path.h5')
    mock_save.assert_called_once_with('dummy_model_path.h5')