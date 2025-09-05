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
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore

# -----------------------
# Download NLTK resources
# -----------------------
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("vader_lexicon")

# -----------------------
# Paths and Artifacts
# -----------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "linear_svc_optuna.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# Firebase setup
firebase_dict = dict(st.secrets["Firebase"])
firebase_dict["private_key"] = firebase_dict["private_key"].replace("\\n", "\n")

if not firebase_admin._apps:  # Prevent re-init
    cred = credentials.Certificate(firebase_dict)
    firebase_admin.initialize_app(cred)

db = firestore.client()

# -----------------------
# Preprocessing Functions
# -----------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\d+", " num ", text)
    text = re.sub(r"[^\w\s!?]", "", text)
    text = " ".join(
        [word for word in text.split() if word not in stopwords.words("english")]
    )
    return text


def extract_features(text: str) -> csr_matrix:
    flesch = textstat.flesch_reading_ease(text)
    gunning = textstat.gunning_fog(text)
    analyzer = SentimentIntensityAnalyzer()
    vader_scores = analyzer.polarity_scores(text)

    features = np.array(
        [
            flesch,
            gunning,
            vader_scores["neg"],
            vader_scores["neu"],
            vader_scores["pos"],
            vader_scores["compound"],
        ]
    ).reshape(1, -1)

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
    5: "verify_code",
}
CATEGORIES = list(CATEGORY_MAP.values())

# -----------------------
# Streamlit UI
# -----------------------
st.title("📧 Email Classifier with Feedback")

# Init session state
if "predicted_label" not in st.session_state:
    st.session_state.predicted_label = None
if "email_text" not in st.session_state:
    st.session_state.email_text = None
if "feedback_mode" not in st.session_state:
    st.session_state.feedback_mode = None  # None | correct | incorrect

with st.form("email_form"):
    email_text = st.text_area("Email text")
    submitted = st.form_submit_button("Classify Email")

if submitted:
    if not email_text.strip():
        st.error("⚠️ Please enter some text.")
    elif email_text.isdigit():
        st.error("⚠️ Text cannot be only numbers.")
    elif re.fullmatch(r"[^\w\s]+", email_text):
        st.error("⚠️ Text cannot be only special characters.")
    elif len(email_text.strip()) < 50:
        st.error("⚠️ Text must be at least 50 characters long.")
    else:
        # Predict
        X_input = prepare_input(email_text)
        pred = model.predict(X_input)[0]
        predicted_label = CATEGORY_MAP[pred]

        st.session_state.predicted_label = predicted_label
        st.session_state.email_text = email_text
        st.session_state.feedback_mode = None

        st.success(f"Predicted Category: **{predicted_label}**")

# Show feedback options only if prediction exists
if st.session_state.predicted_label:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Correct"):
            st.session_state.feedback_mode = "correct"
    with col2:
        if st.button("❌ Incorrect"):
            st.session_state.feedback_mode = "incorrect"

    # Handle Correct feedback
    if st.session_state.feedback_mode == "correct":
        try:
            db.collection("feedback").document().set(
                {
                    "email_text": st.session_state.email_text,
                    "predicted_label": st.session_state.predicted_label,
                    "is_correct": True,
                }
            )
            st.success("Feedback saved as correct ✅")
            st.session_state.feedback_mode = None
        except Exception as e:
            st.error(f"⚠️ Failed to save feedback: {e}")

    # Handle Incorrect feedback
    elif st.session_state.feedback_mode == "incorrect":
        st.warning("Please provide the correct category below 👇")
        correct_label = st.selectbox(
            "Choose category", CATEGORIES, index=0, key="select_correct"
        )
        custom_label = st.text_input(
            "Or enter a new category (optional)", key="custom_correct"
        )

        if st.button("Save Correction"):
            final_label = custom_label if custom_label else correct_label
            try:
                db.collection("feedback").document().set(
                    {
                        "email_text": st.session_state.email_text,
                        "predicted_label": st.session_state.predicted_label,
                        "is_correct": False,
                        "correct_label": final_label,
                    }
                )
                st.success(
                    f"Feedback saved as incorrect ❌ with correction: {final_label}"
                )
                st.session_state.feedback_mode = None
            except Exception as e:
                st.error(f"⚠️ Failed to save feedback: {e}")
