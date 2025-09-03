# 📧 Email Categorization Tool  

An automated tool designed to **classify emails into predefined categories** using **Natural Language Processing (NLP)** and **Machine Learning**.  
By leveraging TF-IDF vectors, sentiment, readability scores, and advanced models.  

---

## 🛠️ Methodology  

### 🔹 Feature Engineering  
- **TF-IDF Vectorization** – Core text representation  
- **Sentiment Features** – VADER sentiment polarity  
- **Readability Metrics** – Flesch Reading Ease & Gunning Fog Index  

👉 Best performance came from combining **TF-IDF + engineered features**.  

### 🔹 Model Experimentation  
The following models were tested:  
- Logistic Regression  
- Random Forest  
- Multinomial Naive Bayes  
- Support Vector Machine (**Best**)  
- LightGBM  
- XGBoost  

---

## 📈 Key Findings  
- Adding sentiment & readability features improved accuracy compared to TF-IDF alone.  
- **SVM outperformed other strong models** like XGBoost and LightGBM.  
