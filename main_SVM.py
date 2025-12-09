import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import nltk
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer

# NLTK setup
nltk.data.path.append(os.path.join(os.path.dirname(__file__), 'nltk_data'))
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Load trained model and vectorizer
model = load('svm_intent_model.joblib')
vectorizer = load('tfidf_vectorizer_SVM.joblib')

# Responses dictionary
responses = {
    "ask_room_price": "Our rooms start from RM180 per night.",
    "ask_availability": "We currently have several rooms available.",
    "ask_facilities": "We offer free Wi-Fi, breakfast, pool, gym and parking.",
    "ask_location": "We are located in Kuala Lumpur City Centre (KLCC).",
    "ask_checkin_time" : "Check-in time is from 2:00 PM.",
    "ask_checkout_time" : "Check-out time is at 12:00 PM.",
    "ask_booking" : "You can book directly through our website or at the front desk.",
    "ask_cancellation" : "Cancellations are free up to 24 hours before arrival.",
    "greeting" : "Hello! How may I assist you today?",
    "goodbye" : "Goodbye! Have a great day!"
}

# Chatbot function
def chatbot_reply_svm(user_input):
    cleaned = preprocess_text(user_input)
    vector = vectorizer.transform([cleaned])
    intent = model.predict(vector)[0]
    return responses.get(intent, f"Sorry, I predicted the intent '{intent}', but I don't have a specific response for that yet. Please rephrase your question.")

# Streamlit UI
st.title("Hotel Chatbot (SVM)")

user_input = st.text_input("Enter your message:")

if user_input:
    response = chatbot_reply_svm(user_input)
    st.markdown(f"**Chatbot:** {response}")
