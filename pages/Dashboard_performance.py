import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack, issparse
import textstat
import re
import nltk
import joblib
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from supabase import create_client
import logging
import pickle

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Download NLTK resources
try:
    logger.info("Downloading NLTK resources...")
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("vader_lexicon")
    logger.info("NLTK resources downloaded successfully.")
except Exception as e:
    logger.error(f"Failed to download NLTK resources: {e}", exc_info=True)
    st.error("Failed to download NLTK resources. Please check your internet connection.")
    st.stop()  # Stop the app if resources can't be downloaded

try:
    logger.info("Application started. Attempting to connect to Supabase.")
    
    # Database connection
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    supabase = create_client(url, key)

    response = supabase.table("test_data").select("*").execute()
    logger.info(f"Supabase response data: {response.data}")

    test_df = pd.DataFrame(response.data)
    logger.info(f"Columns in the loaded DataFrame: {test_df.columns.tolist()}")

    # --- Helper functions ---
    def clean_texts(col):
        col = col.lower()
        col = re.sub(r"\d+", " num ", col)
        col = re.sub(r"[^\w\s!?]", "", col)
        col = " ".join([word for word in col.split() if word not in stopwords.words("english")])
        return col

    def combine_features(X_tfidf, engineered_features):
        if not issparse(engineered_features):
            engineered_features = engineered_features.values
        return hstack([X_tfidf, engineered_features])

    def text_features(df, text_col):
        df = df.copy()
        # Readability features
        df["flesch"] = df[text_col].apply(lambda x: textstat.flesch_reading_ease(str(x)))
        df["gunning_fog"] = df[text_col].apply(lambda x: textstat.gunning_fog(str(x)))
        # VADER features
        analyzer = SentimentIntensityAnalyzer()
        df["vader_neg"] = df[text_col].apply(lambda x: analyzer.polarity_scores(str(x))["neg"])
        df["vader_neu"] = df[text_col].apply(lambda x: analyzer.polarity_scores(str(x))["neu"])
        df["vader_pos"] = df[text_col].apply(lambda x: analyzer.polarity_scores(str(x))["pos"])
        df["vader_compound"] = df[text_col].apply(lambda x: analyzer.polarity_scores(str(x))["compound"])
        return df

    # --- Preprocessing ---
    logger.info("Starting data preprocessing.")
    df1 = test_df.copy()

    # Safely check for column existence
    required_cols = ["subject", "body", "text"]
    for col in required_cols:
        if col not in df1.columns:
            raise KeyError(f"Column '{col}' not found in the database data. Please check your Supabase schema.")

    df1["combined_text_original"] = df1["subject"] + " " + df1["body"] + " " + df1["text"]
    df1["combined_text_normalized"] = df1["combined_text_original"].apply(clean_texts)
    df1 = text_features(df1, "combined_text_original")
    logger.info("Data preprocessing completed.")

    X = df1[["flesch", "gunning_fog", "vader_neg", "vader_neu", "vader_pos", "vader_compound"]]
    y_true = df1["category_id"]

    # --- Load model + vectorizer ---
    logger.info("Loading model and vectorizer...")

    BASE_DIR = os.path.dirname(__file__)
    MODEL_PATH = os.path.join(BASE_DIR, "linear_svc_optuna.pkl")
    VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    logger.info("Model and vectorizer loaded successfully.")

    logger.info("Transforming input data.")
    X_tfidf = vectorizer.transform(df1["combined_text_normalized"])
    X_input = combine_features(X_tfidf, X)
    logger.info("Data transformation completed.")

    logger.info("Making predictions.")
    y_pred = model.predict(X_input)
    logger.info("Predictions completed.")

    # --- Map categories ---
    CATEGORY_MAP = {
        0: "forum",
        1: "promotions",
        2: "social_media",
        3: "spam",
        4: "updates",
        5: "verify_code",
    }

    y_true_named = y_true.map(CATEGORY_MAP)
    y_pred_named = pd.Series(y_pred).map(CATEGORY_MAP)

    # --- Dashboard ---
    logger.info("Rendering dashboard components.")

    st.markdown("## 📊 **Classification Report**")
    report = classification_report(y_true_named, y_pred_named, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.background_gradient(cmap="Oranges").format("{:.2f}"))

    st.markdown("## 🔎 **Confusion Matrix**")
    cm = confusion_matrix(y_true_named, y_pred_named, labels=list(CATEGORY_MAP.values()))
    cm_df = pd.DataFrame(cm, index=list(CATEGORY_MAP.values()), columns=list(CATEGORY_MAP.values()))
    fig_cm = px.imshow(cm_df, text_auto=True, aspect="auto", color_continuous_scale="Tealgrn")
    st.plotly_chart(fig_cm)

    st.markdown("## 📈 **Class Distribution in Test Data**")
    fig_dist = px.bar(
        y_true_named.value_counts().sort_index(),
        labels={"index": "Category", "value": "Count"},
        color=y_true_named.value_counts().sort_index(),
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig_dist)

    logger.info("Dashboard components rendered successfully.")

except Exception as e:
    logger.error(f"An unexpected error occurred: {str(e)}", exc_info=True)
    st.error(f"⚠️ Something went wrong: {str(e)}. Check the 'app.log' file for details.")
