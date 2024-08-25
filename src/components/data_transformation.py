import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def clean_text(text):
    """
    Cleans the input text by removing non-alphanumeric characters and excess whitespace.

    Args:
        text (str): The text to clean.

    Returns:
        str: The cleaned text.
    """
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

def preprocess_data(df, max_len=100):
    """
    Preprocesses the data by cleaning text, encoding labels, and padding sequences.

    Args:
        df (pd.DataFrame): DataFrame containing raw data.
        max_len (int): The maximum length of input sequences.

    Returns:
        tuple: Processed feature array, label array, and tokenizer instance.
    """
    df['message'] = df['message'].apply(clean_text)
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df['message'])
    sequences = tokenizer.texts_to_sequences(df['message'])
    X = pad_sequences(sequences, maxlen=max_len)
    y = df['label'].values
    
    return X, y, tokenizer

def split_data(X, y, test_size=0.2):
    """
    Splits the data into training and testing sets.

    Args:
        X (np.array): Feature array.
        y (np.array): Label array.
        test_size (float): Proportion of the dataset to include in the test split.

    Returns:
        tuple: Training and testing feature and label arrays.
    """
    return train_test_split(X, y, test_size=test_size, random_state=42)