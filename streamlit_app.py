# streamlit_app.py
import streamlit as st
from src.pipeline.predict_pipeline import predict  # Import your predict function directly

st.title("Spam Detection App")
st.write("Enter a message to check if it’s spam or not.")

# Input field for the user message
message = st.text_input("Message:")

# Prediction button
if st.button("Predict"):
    if message:
        # Call the prediction function directly
        predictions = predict([message])
        predicted_labels = (predictions > 0.5).astype(int).tolist()
        prediction_text = "Spam" if predicted_labels[0] == [1] else "Not Spam"
        st.write(f"Prediction: **{prediction_text}**")
    else:
        st.warning("Please enter a message to analyze.")
