import joblib

def load_decision_tree_models():
    try:
        vectorizer = joblib.load("../decision_tree/vectorizers/tfidf_vectorizer.pkl")
        svd = joblib.load("../decision_tree/vectorizers/svd_tfidf.pkl")
        model = joblib.load("../decision_tree/models/decision_tree_tf-idf.pkl")
        label_encoder = joblib.load("../decision_tree/vectorizers/label_encoder.pkl")
        return model, vectorizer, svd, label_encoder
    except Exception as e:
        print(f"Error loading Decision Tree models: {e}")
        return None, None, None, None
