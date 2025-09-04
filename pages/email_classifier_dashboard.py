import os
import re
import numpy as np
import joblib
import textstat
import streamlit as st
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy.sparse import hstack, csr_matrix
import nltk

# -----------------------
# Download NLTK resources
# -----------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# -----------------------
# Paths and Artifacts
# -----------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "linear_svc_optuna.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# -----------------------
# Preprocessing Functions
# -----------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\d+", " num ", text)
    text = re.sub(r"[^\w\s!?]", "", text)
    text = " ".join([word for word in text.split() if word not in stopwords.words('english')])
    return text

def extract_features(text: str) -> csr_matrix:
    flesch = textstat.flesch_reading_ease(text)
    gunning = textstat.gunning_fog(text)
    analyzer = SentimentIntensityAnalyzer()
    vader_scores = analyzer.polarity_scores(text)

    features = np.array([
        flesch,
        gunning,
        vader_scores['neg'],
        vader_scores['neu'],
        vader_scores['pos'],
        vader_scores['compound']
    ]).reshape(1, -1)

    return csr_matrix(features)

def prepare_input(text: str):
    cleaned = clean_text(text)
    X_tfidf = vectorizer.transform([cleaned])
    X_features = extract_features(text)
    return hstack([X_tfidf, X_features])

# -----------------------
# Category Map
# -----------------------
category_map = {
    0: "forum",
    1: "promotions",
    2: "social_media",
    3: "spam",
    4: "updates",
    5: "verify_code"
}

# -----------------------
# Streamlit UI
# -----------------------
st.title("📧 Email Classifier")
st.write("Enter an email text below and let the model classify it.")

with st.form("email_form"):
    email_text = st.text_area("Email Text", height=200)
    submitted = st.form_submit_button("Classify Email")

if submitted:
    if not email_text.strip():
        st.error("⚠️ Please enter some text to classify.")
    else:
        try:
            X_input = prepare_input(email_text)
            prediction = model.predict(X_input)[0]
            predicted_category = category_map.get(prediction, "Unknown")

            st.success(f"✅ Predicted Category: **{predicted_category}**")
        except Exception as e:
            st.error(f"Something went wrong during prediction: {e}")
