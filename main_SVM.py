# %%
#install necessary libraries (assuming the initial block has been run)
!pip install scikit-learn
!pip install pandas
!pip install nltk

# download necessary nltk resources (assuming the initial block has been run)
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

print("NLTK Resources downloaded successfully!")

# Import necessary libraries
import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# 🌟 替换：导入支持向量机 (SVM)
from sklearn.svm import SVC 
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump,load

# Import NLTK
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

# Initialize NLTK resources
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Placeholder: Load the actual dataset. Ensure it has 'text' (user query) and 'intent' (label) columns
df = pd.read_csv('dataset.csv')


# shuffle the data for robust splitting
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print(df.head())

def preprocess_text(text):
    # 1. Convert to Lowercase
    text = text.lower()
    
    # 2. Remove Punctuation and Special Characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # 3. Tokenization
    tokens = word_tokenize(text)
    
    # 4. Stopword Removal
    tokens = [word for word in tokens if word not in stop_words]
    
    # 5. Lemmatization (Key Enhancement)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Rejoin tokens into a single string
    return ' '.join(tokens)

# Apply the new preprocessing function to the text column
df['cleaned_text'] = df['text'].apply(preprocess_text)
print("--- Preprocessing Complete (with NLTK Lemmatization) ---")
print(df[['text', 'cleaned_text']].head())

# Prepare the cleaned text and intents for the model training section
X = df['cleaned_text']
y = df['intent']

# Initialize the TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the text data to create the feature matrix X
X = vectorizer.fit_transform(X)
y = df['intent']

print(f"Feature matrix X shape: {X.shape}")
print(f"Labels y shape: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size=0.2,
    random_state=42,
    stratify=y if len(df['intent'].unique()) > 1 else None, # 仅在有多个类别时分层
)

print(f"Train set size:{X_train.shape[0]} samples")
print(f"Test set size:{X_test.shape[0]} samples")

# 🌟 替换：实例化支持向量机 (SVM) 模型
# 使用 'linear' 核通常在文本分类中表现良好，因为它在高维空间中表现高效
svm_model = SVC(kernel='linear', random_state=42, probability=True)

# 训练模型
svm_model.fit(X_train, y_train)

# Make predictions on the test set
pred = svm_model.predict(X_test)

print("--- SVM Model Evaluation ---")
print("Accuracy:", accuracy_score(y_test, pred))
print("\nClassification Report:\n")
# 检查是否有足够的样本进行分类报告
if len(y_test.unique()) > 1:
    print(classification_report(y_test, pred, zero_division=0))
else:
    print("Classification Report skipped: Only one class in test set.")

# 🌟 替换：保存训练好的 SVM 模型和 Vectorizer
dump(svm_model, 'svm_intent_model.joblib')
dump(vectorizer, 'tfidf_vectorizer_SVM.joblib')
print("Model and Vectorizer saved using joblib.")

# Predefined fixed responses (Retrieval System)
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

# 🌟 替换：更新函数名
def chatbot_reply_svm(user_input, model, vectorizer, responses):
    # 1. Preprocessing
    user_input = user_input.lower()

    # 2. Feature Extraction: Transform the input using the fitted vectorizer
    # ⚠️ 注意: 尽管这里只进行了小写，但对于生产环境，应该使用完整的 `preprocess_text` 函数
    # 为了与之前的代码保持一致，这里使用简洁版本。
    user_input_cleaned = preprocess_text(user_input)
    
    vector = vectorizer.transform([user_input_cleaned])

    # 3. Intent Prediction
    intent = model.predict(vector)[0]

    # 4. Retrieval (Check for unknown intent/fallback)
    # If the predicted intent exists in the dictionary, return the specific response
    # Otherwise, return a fallback message
    return responses.get(intent, f"Sorry, I predicted the intent '{intent}', but I don't have a specific response for that yet. Please rephrase your question.")

# Test the chatbot function
print("\n --- SVM Chatbot Test ---")
test_input = "What is your best price?"
predicted_response = chatbot_reply_svm(test_input, svm_model, vectorizer, responses)
print(f"User Input: {test_input}")
print(f"Chatbot Reply: {predicted_response}")

# %%
!pip install scikit-learn
!pip install pandas
!pip install nltk

# %%
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

# %%
import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# %%
from sklearn.svm import SVC 
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump,load
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

# %%
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# %%
df = pd.read_csv('dataset.csv')
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print(df.head())

# %%
def preprocess_text(text):
    # 1. Convert to Lowercase
    text = text.lower()
    
    # 2. Remove Punctuation and Special Characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # 3. Tokenization
    tokens = word_tokenize(text)
    
    # 4. Stopword Removal
    tokens = [word for word in tokens if word not in stop_words]
    
    # 5. Lemmatization (Key Enhancement)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Rejoin tokens into a single string
    return ' '.join(tokens)
df['cleaned_text'] = df['text'].apply(preprocess_text)
print("--- Preprocessing Complete (with NLTK Lemmatization) ---")
print(df[['text', 'cleaned_text']].head())

# %%
X = df['cleaned_text']
y = df['intent']
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)
y = df['intent']

# %%
X = df['cleaned_text']
y = df['intent']

# Initialize the TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the text data to create the feature matrix X
X = vectorizer.fit_transform(X)
y = df['intent']

print(f"Feature matrix X shape: {X.shape}")
print(f"Labels y shape: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size=0.2,
    random_state=42,
    stratify=y if len(df['intent'].unique()) > 1 else None, # 仅在有多个类别时分层
)

print(f"Train set size:{X_train.shape[0]} samples")
print(f"Test set size:{X_test.shape[0]} samples")

# %%
svm_model = SVC(kernel='linear', random_state=42, probability=True)

svm_model.fit(X_train, y_train)

pred = svm_model.predict(X_test)

print("--- SVM Model Evaluation ---")
print("Accuracy:", accuracy_score(y_test, pred))
print("\nClassification Report:\n")

if len(y_test.unique()) > 1:
    print(classification_report(y_test, pred, zero_division=0))
else:
    print("Classification Report skipped: Only one class in test set.")


# %%
dump(svm_model, 'svm_intent_model.joblib')
dump(vectorizer, 'tfidf_vectorizer_SVM.joblib')
print("Model and Vectorizer saved using joblib.")


# %%
responses = {
    "ask_room_price": "Our rooms start from RM180 per night.",
    "ask_availability": "We currently have several rooms available.",
    "ask_facilities": "We offer free Wi-Fi, breakfast, pool, gym and parking.",
    "ask_location": "We are located in Kuala Lumpur City Centre (KLCC).",
    "ask_checkin_time": "Check-in time is from 2:00 PM.",
    "ask_checkout_time": "Check-out time is at 12:00 PM.",
    "ask_booking": "You can book directly through our website or at the front desk.",
    "ask_cancellation": "Cancellations are free up to 24 hours before arrival.",
    "greeting": "Hello! How may I assist you today?",
    "goodbye": "Goodbye! Have a great day!"
}


# %%
def chatbot_reply_svm(user_input, model, vectorizer, responses):
    user_input = user_input.lower()

    # Use preprocessing
    user_input_cleaned = preprocess_text(user_input)

    vector = vectorizer.transform([user_input_cleaned])

    intent = model.predict(vector)[0]

    return responses.get(
        intent,
        f"Sorry, I predicted the intent '{intent}', but no response is configured for it yet."
    )


# %%
print("\n--- SVM Chatbot Test ---")
test_input = "What is your best price?"
predicted_response = chatbot_reply_svm(test_input, svm_model, vectorizer, responses)

print(f"User Input: {test_input}")
print(f"Chatbot Reply: {predicted_response}")



