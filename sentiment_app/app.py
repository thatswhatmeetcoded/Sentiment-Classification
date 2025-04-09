import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------- LABELS ----------
label_mapping = {0: "Negative", 1: "Positive", 2: "Neutral"}

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Sentiment App", layout="wide")

# ---------- SIDEBAR ----------
st.sidebar.title("🧠 Model Explorer")
page = st.sidebar.radio("Navigate to:", ["Home","Linear Regression","Naive Bayes", "Decision Tree", "ANN", "Clustering", "Simulation"])

# ---------- HOME ----------
if page == "Home":
    st.title("📊 Sentiment Analysis Models Overview")
    st.write("""
        Welcome to the Sentiment Analysis Dashboard. Here you can:
        - View model performances
        - Run a live text simulation with our Decision Tree model
    """)
    st.image("https://miro.medium.com/v2/resize:fit:1000/1*dJJ6tG0MNk6fD4O73bpq7A.png", use_container_width=True)

elif page == "Naive Bayes":
    import subpages.naive_bayes as nb
    nb.render()

elif page == "Linear Regression":
    import subpages.linear_regression as lb
    lb.render()

elif page == "Decision Tree":
    import subpages.decision_tree as dt
    dt.render()
    
elif page == "ANN":
    import subpages.ann as ann
    ann.render()

elif page == "Clustering":
    import subpages.clustering as cl
    cl.render()

elif page == "Simulation":
    import subpages.simulation as sm
    sm.render()
    