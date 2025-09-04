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
from supabase import create_client
from datetime import datetime

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

url = st.secrets["supabase"]["url"]
key = st.secrets["supabase"]["key"]
supabase = create_client(url, key)

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
CATEGORY_MAP = {
    0: "forum",
    1: "promotions",
    2: "social_media",
    3: "spam",
    4: "updates",
    5: "verify_code"
}
CATEGORIES = list(CATEGORY_MAP.values())

# -----------------------
# Streamlit UI
# -----------------------
st.title("📧 Email Classifier with Feedback")
st.write("Enter an email to classify its category and provide feedback.")

with st.form("email_form"):
    email_text = st.text_area("Email text")
    submitted = st.form_submit_button("Classify Email")

if submitted:
    if not email_text:
        st.error("Please enter some text.")
    else:
        # Predict
        X_input = prepare_input(email_text)
        pred = model.predict(X_input)[0]
        predicted_label = CATEGORY_MAP[pred]

        st.success(f"Predicted Category: **{predicted_label}**")

        # Feedback Buttons
        col1, col2 = st.columns(2)

        with col1:
            if st.button("✅ Correct"):
                supabase.table("feedback").insert({
                    "email_text": email_text,
                    "predicted_label": predicted_label,
                    "is_correct": True,
                    "correct_label": None,
                    "created_at": datetime.utcnow().isoformat()
                }).execute()
                st.info("Feedback saved as correct ✅. This helps validate the model performance.")

        with col2:
            if st.button("❌ Incorrect"):
                st.write("Select the correct category or enter a new one:")
                correct_label = st.selectbox("Choose category", CATEGORIES, index=0, key="select_correct")
                custom_label = st.text_input("Or enter a new category (optional)", key="custom_correct")

                if st.button("Save Correction"):
                    final_label = custom_label if custom_label else correct_label

                    supabase.table("feedback").insert({
                        "email_text": email_text,
                        "predicted_label": predicted_label,
                        "is_correct": False,
                        "correct_label": final_label,
                        "created_at": datetime.utcnow().isoformat()
                    }).execute()

                    st.info(f"Feedback saved as incorrect ❌ with correction: {final_label}. "
                            f"This will help improve the model.")
