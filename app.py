import streamlit as st

# Page config
st.set_page_config(
    page_title="Email Classifier Dashboard",
    page_icon="📧",
    layout="wide"
)

# Landing page
st.title("📧 Email Classifier Dashboard")
st.markdown(
    """
    Welcome to the **Email Classifier App** 🎉  

    This project demonstrates:
    - Training and optimizing a machine learning model (LinearSVC with Optuna).  
    - Feature engineering with TF-IDF, readability, and sentiment analysis.  
    - Model evaluation and error analysis with interactive dashboards.  

    ---
    ### Available Dashboards
    - **📊 Model Performance**: View classification reports, confusion matrix, class distribution, and misclassified examples.  
    - **📝 Try it Yourself (coming soon)**: Enter your own email text and see the predicted category in real time.  

    Use the **sidebar** on the left to navigate between dashboards.
    """
)
