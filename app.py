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
    """
)

st.markdown("## Available Dashboards")

st.markdown(
    """
    - 📊 **Model Performance**: View classification reports, confusion matrix, and class distribution.  
    - 📝 **Feedback Dashboard**: Review user-submitted emails, analyze correct ✅ / incorrect ❌ feedback, and explore category-level trends.  
    - ✉️ **Try it Yourself**: Enter your own email text, see the predicted category in real time, and provide feedback to improve the model.  

    👉 Use the **sidebar** on the left to navigate between dashboards.
    """
)

