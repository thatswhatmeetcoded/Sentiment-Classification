import streamlit as st
# from utils.load_models import load_decision_tree_models
from utils.preprocess import clean_text
from utils.model_analysis import model_analysis_page
import pandas as pd

# def render():
#     # st.title("📊 Sentiment Analysis Models Overview")
#     # st.write("""
#     #     Welcome to the Sentiment Analysis Dashboard. Here you can:
#     #     - View model performances
#     #     - Run a live text simulation with our Decision Tree model
#     # """)
#     # st.image("https://miro.medium.com/v2/resize:fit:1000/1*dJJ6tG0MNk6fD4O73bpq7A.png", use_container_width=True)
#     import streamlit as st

def render():
    # st.set_page_config(page_title="Sentiment Analysis App", layout="wide")

    st.title("📊 AI-Powered Sentiment Analysis Dashboard")

    st.markdown("""
    ### Welcome to the Sentiment Insight Engine 
    This platform allows you to analyze text data using various machine learning models. Whether you're dealing with product reviews, social media posts, or customer feedback — this app can help decode **emotions and opinions** in real-time.
    
    ---
    """)

    # st.image("https://miro.medium.com/v2/resize:fit:1000/1*dJJ6tG0MNk6fD4O73bpq7A.png", use_container_width=True)

    st.markdown("""
    ##  Real-World Use Cases
    -  **E-Commerce**: Understand customer sentiment from product reviews
    -  **Social Media**: Analyze public mood on Twitter/X, Reddit, etc.
    -  **Finance**: Gauge market sentiment from financial news and discussions
    -  **Games & Entertainment**: Monitor fan feedback in communities
    -  **Customer Support**: Analyze and prioritize customer queries by tone
    
    ---
    """)

    st.markdown("##  Models Included in This App")

    models = [
        " Decision Tree",
        " Logistic Regression",
        " K-Nearest Neighbors (KNN)",
        " Artificial Neural Network (ANN)",
        " Naive Bayes",
        " Support Vector Machine (SVM)",
        " Clustering (Unsupervised)"
    ]
    for model in models:
        st.markdown(f"- {model}")

    st.info("All models are trained using TF-IDF features , BOW(Bag of Words) and compared on common sentiment datasets.")

    st.markdown("""
    ---
    ##  Model Performance Comparison

    > _Below are visual comparisons of different model accuracies, precision, recall, and F1-scores on validation data._

    - 💡 **[Insert Accuracy Comparison Chart Here]**
    - 💡 **[Insert Confusion Matrix Heatmaps]**
    - 💡 **[Insert Training Time vs Accuracy Graph]**

    _Charts are generated dynamically from stored logs and experiment results._

    ---

    ## Try the Live Sentiment Simulator
    Head over to the **Live Simulation** tab and enter any text (tweets, review, comment) to instantly see what the models predict.

    👉 It’s fast. It’s smart. It’s accurate.

    ---

    ##  Feedback & Improvements
    We'd love to hear your feedback. This app is open-source and continuously evolving. If you have ideas, contributions, or datasets to try — reach out!

    _Built using **Streamlit**, **scikit-learn**, and python libraries_
    """)

    st.success(" All models are pre-loaded and ready to simulate!")
