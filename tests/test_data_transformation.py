import pandas as pd
import numpy as np
import pytest
from components.data_transformation import preprocess_data, split_data
from components.data_ingestion import load_data

@pytest.fixture
def data():
    """
    Fixture to create a sample DataFrame for testing preprocessing and splitting.
    
    Returns:
        pd.DataFrame: Sample data for preprocessing and splitting.
    """
    data = {
        'label': ['ham', 'spam'],
        'message': ['Hello, how are you?', 'Win a million dollars now!']
    }
    df = pd.DataFrame(data)
    return df

def test_preprocess_data(data):
    """
    Test the preprocess_data function to ensure it correctly preprocesses text and encodes labels.
    
    Args:
        data: Sample DataFrame containing raw messages and labels.
    
    Asserts:
        - Preprocessed feature and label arrays have matching number of samples.
        - Feature arrays have the correct length.
        - Tokenizer has a non-zero word index.
    """
    X, y, tokenizer = preprocess_data(data)
    assert X.shape[0] == y.shape[0], "Feature and label arrays should have the same number of samples"
    assert X.shape[1] == 100, "Feature arrays should have a length of 100"
    assert len(tokenizer.word_index) > 0, "Tokenizer should have a non-zero word index"

def test_split_data(data):
    """
    Test the split_data function to ensure it correctly splits data into training and test sets.
    
    Args:
        data: Sample DataFrame containing raw messages and labels.
    
    Asserts:
        - Training and test feature arrays should have samples.
        - Training and test label arrays should match their respective feature arrays.
    """
    X, y, tokenizer = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(X, y)
    assert X_train.shape[0] > 0, "Training feature array should have samples"
    assert X_test.shape[0] > 0, "Testing feature array should have samples"
    assert y_train.shape[0] == X_train.shape[0], "Training labels array should match training features"
    assert y_test.shape[0] == X_test.shape[0], "Testing labels array should match testing features"