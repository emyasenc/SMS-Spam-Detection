from src.logger import get_logger
from keras.models import Sequential
from keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def create_model(vocab_size, embedding_dim=16, max_len=100):
    """
    Creates and compiles a Sequential model for SMS spam detection.

    Args:
        vocab_size (int): The size of the vocabulary.
        embedding_dim (int): The dimensionality of the embedding vectors.
        max_len (int): The maximum length of input sequences.

    Returns:
        model: A compiled TensorFlow Sequential model.
    """
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        GlobalAveragePooling1D(),
        Dense(16, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32, logger=None):
    """
    Trains the model on the training data and validates on the validation data.

    Args:
        model: The model to train.
        X_train: Training data features.
        y_train: Training data labels.
        X_val: Validation data features.
        y_val: Validation data labels.
        epochs (int): Number of epochs to train.
        batch_size (int): Number of samples per gradient update.
        logger: Logger instance for logging.

    Returns:
        history: A History object containing training details.
    """
    if logger:
        logger.info("Starting model training")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=2
    )
    if logger:
        logger.info("Model training complete")
    return history

def evaluate_model(model, X_test, y_test, logger=None):
    """
    Evaluates the trained model on the test data with additional metrics.

    Args:
        model: The trained model to evaluate.
        X_test: Test data features.
        y_test: Test data labels.
        logger: Logger instance for logging.

    Returns:
        dict: Dictionary containing loss, accuracy, precision, recall, and F1-score.
    """
    if logger:
        logger.info("Starting extended model evaluation")
    
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()  # Convert probabilities to class labels

    # Calculate metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    loss, accuracy = model.evaluate(X_test, y_test)

    # Log results
    if logger:
        logger.info(f"Evaluation Results - Loss: {loss}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save plot to file instead of showing if not interactive
    plt.savefig('confusion_matrix.png')
    plt.close()

    return {
        'loss': loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def save_model(model, filepath='models/sms_spam_detector.h5', logger=None):
    """
    Saves the trained model to a file.

    Args:
        model: The model to save.
        filepath (str): The path to save the model file.
        logger: Logger instance for logging.
    """
    if logger:
        logger.info(f"Saving model to {filepath}")
    model.save(filepath)
    if logger:
        logger.info("Model saved successfully")
