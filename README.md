# SMS Spam Detection

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Running the Application](#running-the-application)
  - [Making Predictions](#making-predictions)
- [Components](#components)
  - [1. Data Ingestion](#1-data-ingestion)
  - [2. Data Transformation](#2-data-transformation)
  - [3. Model Training](#3-model-training)
  - [4. Prediction Pipeline](#4-prediction-pipeline)
  - [5. Training Pipeline](#5-training-pipeline)
  - [6. Web Application](#6-web-application)
- [Logging](#logging)
- [Requirements](#requirements)
- [License](#license)
- [Author](#author)

## Overview

This project is an SMS spam detection web application that utilizes deep learning techniques to classify messages as "Spam" or "Not Spam." The application employs a Keras-based neural network model trained on a dataset of SMS messages.


## Installation

To get started with the SMS Spam Detection application, follow these installation steps:

1. Clone the repository:
   git clone https://github.com/yourusername/SMS-Spam-Detection.git
   cd SMS-Spam-Detection
   
2. Install the required packages:
   pip install -r requirements.txt

## Usage

# Training the Model

To train the spam detection model, run the following command. This will load the dataset, preprocess the data, and train the neural network model.

python src/pipeline/train_pipeline.py

## Running the Application

To run the Flask web application, execute the following command. Once the server starts, you can access the application at http://127.0.0.1:5000/ in your web browser.

python app.py

## Making Predictions

You can submit SMS messages through the web interface, and the model will classify them as "Spam" or "Not Spam." The application provides immediate feedback on the classification.

## Components

1. Data Ingestion
data_ingestion.py: This module is responsible for loading the dataset from a CSV file. It ensures that the data is correctly formatted and ready for processing.

2. Data Transformation
data_transformation.py:

Cleans and preprocesses the text data by removing unwanted characters and normalizing the text.
Encodes labels and pads sequences for input to the model, ensuring consistent input sizes for training.

3. Model Training
model_trainer.py:

Defines the neural network architecture, including layers and activation functions.
Trains the model using the processed data and evaluates performance on test data, providing insights into model accuracy.

4. Prediction Pipeline
predict_pipeline.py:

Loads the trained model and tokenizer to prepare for making predictions.
Prepares input for prediction and returns predictions, allowing users to classify new messages.

5. Training Pipeline
train_pipeline.py: This orchestrates the entire training process, from loading data to saving the trained model, ensuring a seamless training experience.

6. Web Application
app.py: The main Flask application that serves as the user interface. Users can input SMS messages and receive predictions from the trained model, enhancing user engagement.

## Logging

The project includes logging functionality for monitoring the training and prediction processes. Logs are stored in the logs/ directory, allowing for easy tracking of activities and debugging.

## Requirements

Python 3.8+
Required packages listed in requirements.txt to ensure compatibility and functionality.

## License

This project is licensed under the MIT License, promoting open-source collaboration and use.
