import os
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import psycopg2
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

# Download NLTK resources
nltk.download('punkt')
nltk.download("stopwords")
nltk.download("vader_lexicon")

# Load environment variables
load_dotenv()

try:
    # Database connection
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT")
    DB_NAME = os.getenv("DB_NAME")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")

    DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(DATABASE_URL)

    test_df = pd.read_sql("SELECT * FROM test_data", engine)

    # --- Helper functions ---
    def clean_texts(col): 
        col = col.lower() 
        col = re.sub(r"\d+", " num ", col)  
        col = re.sub(r"[^\w\s!?]", "", col)  
        col = " ".join([word for word in col.split() if word not in stopwords.words('english')])  
        return col

    def combine_features(X_tfidf, engineered_features):
        if not issparse(engineered_features):
            engineered_features = engineered_features.values
        return hstack([X_tfidf, engineered_features])

    def text_features(df, text_col):
        df = df.copy()
        # Readability features
        df['flesch'] = df[text_col].apply(lambda x: textstat.flesch_reading_ease(str(x)))
        df['gunning_fog'] = df[text_col].apply(lambda x: textstat.gunning_fog(str(x)))
        # VADER features
        analyzer = SentimentIntensityAnalyzer()
        df['vader_neg'] = df[text_col].apply(lambda x: analyzer.polarity_scores(str(x))['neg'])
        df['vader_neu'] = df[text_col].apply(lambda x: analyzer.polarity_scores(str(x))['neu'])
        df['vader_pos'] = df[text_col].apply(lambda x: analyzer.polarity_scores(str(x))['pos'])
        df['vader_compound'] = df[text_col].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])
        return df

    # --- Preprocessing ---
    df1 = test_df.copy()
    df1['combined_text_original'] = df1['subject'] + " " + df1['body'] + " " + df1['text']
    df1['combined_text_normalized'] = df1['combined_text_original'].apply(clean_texts)
    df1 = text_features(df1, 'combined_text_original')

    X = df1[['flesch','gunning_fog','vader_neg','vader_neu','vader_pos','vader_compound']]
    y_true = df1['category_id']

    # --- Load model + vectorizer ---
    model = joblib.load("Email_Classifier/linear_svc_optuna.pkl")
    tfidf = joblib.load("Email_Classifier/tfidf_vectorizer.pkl")

    X_tfidf = tfidf.transform(df1['combined_text_normalized'])
    X_input = combine_features(X_tfidf, X)

    y_pred = model.predict(X_input)

    # --- Dashboard ---
    st.header("Classification Report") 
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    st.header("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred, labels=model.classes_)
    cm_df = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)
    fig_cm = px.imshow(cm_df, text_auto=True, aspect="auto", color_continuous_scale="Blues") 
    st.plotly_chart(fig_cm)

    st.header("Class Distribution in Test Data") 
    st.bar_chart(y_true.value_counts())

except Exception as e:
    st.error(f"⚠️ Something went wrong: {str(e)}")
