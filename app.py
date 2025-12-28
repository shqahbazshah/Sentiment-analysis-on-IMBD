import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import re
import streamlit as st

# Load the IMDB word index
word_index = imdb.get_word_index()

# Load pre-trained model
model = load_model('simple_RNN_imdb.h5')

# Parameters
max_features = 10000
max_len = 500

# Preprocess input text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    encoded = []
    for word in text.split():
        # Add 3 to shift indices because Keras reserves 0-3
        idx = word_index.get(word, 2) + 3
        if idx >= max_features:
            idx = 2  # unknown token
        encoded.append(idx)
    # Pad sequence
    return pad_sequences([encoded], maxlen=max_len)

# Predict sentiment
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    return sentiment, prediction

# Streamlit interface
st.title('IMDb Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

user_input = st.text_area('Movie Review')

if st.button('Classify'):
    sentiment, prediction = predict_sentiment(user_input)
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction}')
else:
    st.write('Please enter a movie review.')
