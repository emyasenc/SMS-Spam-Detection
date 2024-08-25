import os
import pytest
import pandas as pd
from components.data_ingestion import load_data

@pytest.fixture
def data_filepath(tmpdir):
    """
    Fixture to create a temporary file with example data for testing.
    
    Args:
        tmpdir: pytest fixture for creating temporary directories and files.
    
    Returns:
        str: Path to the temporary file containing example data.
    """
    file = tmpdir.join("spam.csv")
    data = "label\tmessage\nham\tHello, how are you?\nspam\tWin a million dollars now!"
    file.write(data)
    return str(file)

def test_load_data(data_filepath):
    """
    Test the load_data function to ensure it correctly loads data into a DataFrame.
    
    Args:
        data_filepath: Path to the temporary file containing example data.
    
    Asserts:
        - Data is loaded into a pandas DataFrame.
        - DataFrame is not empty.
        - DataFrame contains the expected columns.
    """
    df = load_data(data_filepath)
    assert isinstance(df, pd.DataFrame), "Data should be a pandas DataFrame"
    assert not df.empty, "DataFrame should not be empty"
    assert 'label' in df.columns, "DataFrame should contain 'label' column"
    assert 'message' in df.columns, "DataFrame should contain 'message' column"