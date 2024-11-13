import os
import streamlit as st
from keras.models import load_model  # Import load_model from tensorflow
from src.pipeline.predict_pipeline import predict  # Assuming this function is implemented to handle prediction

# Function to load the model using a relative path
def load_model_from_relative_path():
    # Get the current file directory
    current_directory = os.path.dirname(__file__)
    
    # Define the relative path to the model file
    model_path = os.path.join(current_directory, 'src', 'models', 'sms_spam_detector.h5')
    
    # Load and return the model
    return load_model(model_path)

# Load the model at the start of the app
model = load_model_from_relative_path()

st.title("Spam Detection App")
st.write("Enter a message to check if it’s spam or not.")

# Input field for the user message
message = st.text_input("Message:")

# Prediction button
if st.button("Predict"):
    if message:
        # Call the prediction function directly with the message
        predictions = predict([message])  # Make the prediction
        
        # Display raw predictions (for debugging/logging purposes)
        st.write(f"Raw predictions: {predictions}")
        
        # Apply threshold: 0.5 means the cutoff between spam and not spam
        if predictions[0] > 0.5:  # If probability is greater than 0.5
            prediction_text = "Spam"
        else:  # If probability is less than or equal to 0.5
            prediction_text = "Not Spam"
        
        st.write(f"Prediction: **{prediction_text}**")
    else:
        st.warning("Please enter a message to analyze.")
