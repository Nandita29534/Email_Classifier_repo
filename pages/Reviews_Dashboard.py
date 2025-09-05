import streamlit as st
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore

# -------------------------
# Firebase Connection
# -------------------------
if not firebase_admin._apps:
    cred = credentials.Certificate(st.secrets["Firebase"])
    firebase_admin.initialize_app(cred)

db = firestore.client()

# -------------------------
# Helper Function
# -------------------------
def get_feedback_data():
    feedback_ref = db.collection("feedback")
    docs = feedback_ref.stream()
    data = []
    for doc in docs:
        entry = doc.to_dict()
        entry["id"] = doc.id
        data.append(entry)
    return data

# -------------------------
# Dashboard
# -------------------------
st.markdown("## 📝 Feedback Dashboard")

feedback_data = get_feedback_data()

if feedback_data:
    df = pd.DataFrame(feedback_data)

    # Show raw table
    st.subheader("All Feedback")
    st.dataframe(df[['is_correct','predicted_label','email_text','correct_label']])

    # Summary stats
    st.subheader("Summary")
    correct_count = df["is_correct"].sum()
    incorrect_count = len(df) - correct_count
    col1, col2 = st.columns(2)
    with col1:
        st.metric("✅ Correct", correct_count)
    with col2:
        st.metric("❌ Incorrect", incorrect_count)

    # Predicted categories distribution
    st.subheader("📊 Predicted Categories Distribution")
    if "predicted_label" in df.columns:
        pred_counts = df["predicted_label"].value_counts()
        st.bar_chart(pred_counts)

    # Correct vs Incorrect by category
    st.subheader("✅ vs ❌ Feedback by Category")
    if "predicted_label" in df.columns and "is_correct" in df.columns:
        chart_data = (
            df.groupby("predicted_label")["is_correct"]
            .value_counts()
            .unstack(fill_value=0)
        )
        st.bar_chart(chart_data)

else:
    st.info("No feedback collected yet.")
