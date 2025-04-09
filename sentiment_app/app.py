import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ---------- TEXT PREPROCESSING ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# ---------- LOAD MODELS ----------
try:
    vectorizer = joblib.load('../decision_tree/vectorizers/tfidf_vectorizer.pkl')
    svd = joblib.load('../decision_tree/vectorizers/svd_tfidf.pkl')
    model = joblib.load('../decision_tree/models/decision_tree_tf-idf.pkl')
    label_encoder = joblib.load('../decision_tree/vectorizers/label_encoder.pkl')
except Exception as e:
    vectorizer = None
    model = None
    svd = None
    label_encoder = None
    print("Error loading models:", e)

# ---------- LABELS ----------
label_mapping = {0: "Negative", 1: "Positive", 2: "Neutral"}

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Sentiment App", layout="wide")

# ---------- SIDEBAR ----------
st.sidebar.title("🧠 Model Explorer")
page = st.sidebar.radio("Navigate to:", ["Home", "Naive Bayes", "Decision Tree", "ANN", "Clustering", "Simulation"])

# ---------- HOME ----------
if page == "Home":
    st.title("📊 Sentiment Analysis Models Overview")
    st.write("""
        Welcome to the Sentiment Analysis Dashboard. Here you can:
        - View model performances
        - Run a live text simulation with our Decision Tree model
    """)
    st.image("https://miro.medium.com/v2/resize:fit:1000/1*dJJ6tG0MNk6fD4O73bpq7A.png", use_container_width=True)

# ---------- DUMMY MODEL ANALYSIS ----------
def model_analysis_page(name, accuracy, precision, recall, f1, notes):
    st.header(f"📘 {name} Model Analysis")
    st.metric("Accuracy", f"{accuracy}%")
    st.metric("Precision", f"{precision}")
    st.metric("Recall", f"{recall}")
    st.metric("F1 Score", f"{f1}")
    st.markdown("**Insights:**")
    st.info(notes)

if page == "Naive Bayes":
    model_analysis_page("Naive Bayes", 82.5, 0.83, 0.80, 0.81,
        "Performs well with smaller datasets. Fast but less expressive.")

elif page == "Decision Tree":
    model_analysis_page("Decision Tree", 85.2, 0.86, 0.83, 0.84,
        "Captures patterns well, interpretable, might overfit on small data.")

elif page == "ANN":
    model_analysis_page("ANN (Artificial Neural Network)", 88.7, 0.89, 0.87, 0.88,
        "High performance on larger datasets. Needs more training time.")

elif page == "Clustering":
    model_analysis_page("K-Means Clustering", "-", "-", "-", "-",
        "Unsupervised model used to group sentiments. Not directly comparable to classifiers.")

# ---------- SIMULATION ----------
elif page == "Simulation":
    st.title("🚀 Live Sentiment Simulation")
    st.markdown("Enter text and let our Decision Tree model predict its sentiment!")

    user_input = st.text_area("📝 Your Text:")
    if st.button("Predict Sentiment"):
        if model and vectorizer and label_encoder:
            if user_input.strip() == "":
                st.warning("Please enter some text.")
            else:
                cleaned = clean_text(user_input)
                st.write("✅ **Cleaned Text:**", cleaned)

                try:
                    X_vec = vectorizer.transform([cleaned])
                    X_reduced = svd.transform(X_vec)  
                    pred = model.predict(X_reduced)[0]
                    pred_label = label_encoder.inverse_transform([pred])[0] if label_encoder else label_mapping.get(pred, str(pred))

                    st.success(f"🎯 Predicted Sentiment: **{pred_label}**")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
        else:
            st.error("Models not loaded correctly.Please recheck your files.")
