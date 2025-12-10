# =========================
# hotel_chatbot.py
# =========================
# Install SpaCy
# !pip install spacy
# !python -m spacy download en_core_web_sm
# Install NLTK
# !pip install nltk
# Install scikit-learn
# !pip install scikit-learn
# Install joblib
# !pip install joblib

# -------------------------
# Import Libraries
# -------------------------
import pandas as pd
import numpy as np
import re
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from joblib import dump, load
from collections import defaultdict

# -------------------------
# 1️⃣ NLTK Preprocessing Setup
# -------------------------
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """
    Preprocess text: lowercase, remove punctuation, tokenize,
    remove stopwords, lemmatize.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# -------------------------
# 2️Load Dataset
# -------------------------
df = pd.read_csv('dataset.csv')  # Ensure columns: instruction, intent
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

df['cleaned_text'] = df['instruction'].apply(preprocess_text)

X_text = df['cleaned_text']
y = df['intent']

# -------------------------
# 3️TF-IDF Vectorization
# -------------------------
vectorizer = TfidfVectorizer(ngram_range=(1,2))  # unigrams + bigrams
X = vectorizer.fit_transform(X_text)

# -------------------------
# 4️Train/Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=y if len(df['intent'].unique()) > 1 else None
)

# -------------------------
# 5️Train SVM Model
# -------------------------
svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# Evaluate
pred = svm_model.predict(X_test)
print("--- SVM Model Evaluation ---")
print("Accuracy:", accuracy_score(y_test, pred))
if len(y_test.unique()) > 1:
    print(classification_report(y_test, pred, zero_division=0))

# Save model & vectorizer
dump(svm_model, 'svm_intent_model.joblib')
dump(vectorizer, 'tfidf_vectorizer_SVM.joblib')

# -------------------------
# 6️SpaCy NER for Entities
# -------------------------
nlp = spacy.load("en_core_web_sm")  # or en_core_web_lg if installed

def get_entities(text):
    """
    Extract entities from user text using SpaCy
    Returns dict: entity_label -> [values]
    """
    doc = nlp(text)
    entities = defaultdict(list)
    for ent in doc.ents:
        entities[ent.label_].append(ent.text)
    return entities

# -------------------------
# 7️Policy for Contextual Responses
# -------------------------
policy = {
    ("ask_room_price", "ROOM_TYPE"): "The {ROOM_TYPE} costs RM180 per night.",
    ("ask_booking", "NUM_GUESTS"): "I can book a {ROOM_TYPE} for {NUM_GUESTS} starting {DATE}.",
    ("ask_checkin_time", "TIME"): "Check-in starts at {TIME}.",
    ("ask_checkout_time", "TIME"): "Check-out is at {TIME}.",
    # Default response
    ("default", "none"): "Sorry, I don't understand. Could you please rephrase?"
}

# -------------------------
# 8️Chatbot Response Function
# -------------------------
def get_intent(text, model=svm_model, vectorizer=vectorizer, threshold=0.5):
    """
    Predict intent using SVM model.
    If confidence < threshold, return 'default'
    """
    text_cleaned = preprocess_text(text)
    vector = vectorizer.transform([text_cleaned])
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vector)[0]
        max_prob = max(probs)
        if max_prob < threshold:
            return "default"
    return model.predict(vector)[0]

def respond(text):
    """
    Generate chatbot response with entity handling.
    """
    intent = get_intent(text)
    entities = get_entities(text)
    
    if intent != "default":
        # Try to find the first matching entity type in policy
        for key in ["ROOM_TYPE", "NUM_GUESTS", "TIME", "DATE"]:
            if (intent, key) in policy:
                response_template = policy[(intent, key)]
                response = response_template.format(
                    ROOM_TYPE=entities.get("ROOM_TYPE", ["room"])[0],
                    NUM_GUESTS=entities.get("CARDINAL", ["1"])[0],
                    DATE=entities.get("DATE", ["today"])[0],
                    TIME=entities.get("TIME", ["2 PM"])[0]
                )
                return response
        # Fallback if no entity matches
        response = "Sorry, I cannot handle this request."
    else:
        response = policy[("default", "none")]
    
    return response

# -------------------------
# 9️ Test Chatbot
# -------------------------
if __name__ == "__main__":
    test_inputs = [
        "What is the price for a deluxe room?",
        "Can I book a room for 2 adults next Monday?",
        "When can I check in?",
        "When is check-out?",
        "Hi there!",
        "Blah blah random text"
    ]
    
    for text in test_inputs:
        print(f"User: {text}")
        print(f"Bot: {respond(text)}")
        print("-"*50)
