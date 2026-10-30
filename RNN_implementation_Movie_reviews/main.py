# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.models import load_model

max_features = 10000
max_length = 500

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

model = load_model('simple_rnn.keras')

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    encoded_review = [1]
    for word in text_to_word_sequence(text):
        word_id = word_index.get(word)
        token_id = word_id + 3 if word_id is not None and word_id + 3 < max_features else 2
        encoded_review.append(token_id)
    return sequence.pad_sequences([encoded_review], maxlen=max_length)


import streamlit as st
## streamlit app
# Streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User input
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    if not user_input.strip():
        st.warning('Enter a movie review before classifying it.')
    else:
        preprocessed_input = preprocess_text(user_input)
        prediction = float(model.predict(preprocessed_input, verbose=0)[0, 0])
        sentiment = 'Positive' if prediction >= 0.5 else 'Negative'

        st.write(f'Sentiment: {sentiment}')
        st.write(f'Prediction Score: {prediction:.4f}')
else:
    st.write('Please enter a movie review.')
