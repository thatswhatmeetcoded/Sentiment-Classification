import streamlit as st
from utils.preprocess import clean_text
from utils.load_models import load_decision_tree_models

def render():
    model, vectorizer, svd, label_encoder = load_decision_tree_models()

    st.title("🚀 Live Sentiment Simulation")

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
                    pred_label = label_encoder.inverse_transform([pred])[0]
                    st.success(f"🎯 Predicted Sentiment: **{pred_label}**")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
        else:
            st.error("Models not loaded. Check paths.")
