import pandas as pd

def load_data(filepath):
    """
    Loads data from a CSV file into a pandas DataFrame.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    """
    df = pd.read_csv(filepath, sep='\t', header=None, names=['label', 'message'])
    return df