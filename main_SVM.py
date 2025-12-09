import streamlit as st
import numpy as np
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
from sklearn.svm import SVC

# Configuration Parameters
CONFIDENCE_THRESHOLD = 0.7 

# Chatbot Response Lookup Table
RESPONSE_DICT = {
    "ask_room_price": "Our rooms start from RM180 per night.",
    "ask_availability": "We currently have several rooms available.",
    "ask_facilities": "We offer free Wi-Fi, breakfast, pool, gym and parking.",
    "ask_location": "We are located in Kuala Lumpur City Centre (KLCC).",
    "ask_checkin_time" : "Check-in time is from 2:00 PM.",
    "ask_checkout_time" : "Check-out time is at 12:00 PM.",
    "ask_booking" : "You can book directly through our website or at the front desk.",
    "ask_cancellation" : "Cancellations are free up to 24 hours before arrival.",
    "greeting" : "Hello! How may I assist you today?",
    "goodbye" : "Goodbye! Have a great day!",
    "unrecognized_intent": "I apologize, but I currently cannot understand your request. Could you please try rephrasing your question?", 
}

# NLTK Download and Preprocessing Setup
@st.cache_resource(show_spinner="Downloading NLTK resources...")
def download_nltk_resources():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('stopwords', quiet=True)
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        return True, stop_words, lemmatizer
    except Exception as e:
        st.error(f"Failed to download NLTK resources: {e}")
        return False, set(), None

is_nltk_ready, stop_words, lemmatizer = download_nltk_resources()

def preprocess_text(text):
    if not lemmatizer:
        return "" 
        
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Model Loading and Caching
@st.cache_resource
def load_resources():
    try:
        svm_model = joblib.load('svm_intent_model.joblib')
        vectorizer = joblib.load('tfidf_vectorizer_SVM.joblib') 
        return svm_model, vectorizer
    except FileNotFoundError as e:
        st.error(f"Error loading required model files. Missing file: {e.filename}")
        return None, None

svm_model, vectorizer = load_resources()

# Prediction Function
def predict_intent(text):
    if svm_model is None or vectorizer is None or not is_nltk_ready:
        return "setup_error", RESPONSE_DICT.get("unrecognized_intent"), "N/A"

    user_input_cleaned = preprocess_text(text)
    vector = vectorizer.transform([user_input_cleaned])

    try:
        predictions_proba = svm_model.predict_proba(vector)[0]
    except AttributeError:
        st.error("Error: The SVM model was not trained with `probability=True`.")
        return "model_error", RESPONSE_DICT.get("unrecognized_intent"), "N/A"
    
    predicted_index = np.argmax(predictions_proba)
    confidence_score = np.max(predictions_proba)
    predicted_intent_name = svm_model.classes_[predicted_index]

    if confidence_score < CONFIDENCE_THRESHOLD:
        intent_name = "unrecognized_intent"
        response = RESPONSE_DICT.get(intent_name)
    else:
        intent_name = predicted_intent_name
        response = RESPONSE_DICT.get(intent_name, RESPONSE_DICT['unrecognized_intent'])

    confidence_display = f"{confidence_score*100:.2f}%"
    
    return intent_name, response, confidence_display

# Streamlit App Layout
def main():
    st.set_page_config(page_title="SVM Intent Chatbot (Chat History)", layout="centered")
    st.title("🛡️ Hotel Chatbot (Support Vector Machine)")
    st.caption(f"Confidence Threshold: **{CONFIDENCE_THRESHOLD*100:.0f}%**")

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": RESPONSE_DICT['greeting']})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "intent" in message:
                st.caption(f"Intent: **{message['intent']}** | Confidence: **{message['confidence']}**")
            st.markdown(message["content"])

    user_input = st.chat_input("How may I assist you with your reservation?")
    
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner('Analyzing query...'):
            intent_name, response, confidence_display = predict_intent(user_input)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "intent": intent_name,
                "confidence": confidence_display
            })

            with st.chat_message("assistant"):
                st.caption(f"Intent: **{intent_name}** | Confidence: **{confidence_display}**")
                st.markdown(response)

if __name__ == "__main__":
    main()
