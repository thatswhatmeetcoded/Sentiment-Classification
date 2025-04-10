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


# def load_naive_bayes():
#     try:
#         vectorizer = joblib.load("../naive_bayes/vectorizers/tfidf_vectorizer.pkl")
#         svd = joblib.load("../decision_tree/vectorizers/svd_tfidf.pkl")
#         model = joblib.load("../naive_bayes/model/nb_model_tfidf.pkl")
#         label_encoder = joblib.load("../decision_tree/vectorizers/label_encoder.pkl")
#         return model, vectorizer,svd,label_encoder

#     except Exception as e:
#         print(f"Error loading Naive Bayes model models: {e}")
#         return None, None, None, None
def load_naive_bayes():
    # try:
    #     print("Loading Naive Bayes models...")
    #     #vectorizer = joblib.load("../naive_bayes/vectorisers/tfidf_vectorizer.pkl")
    #     vectorizer = joblib.load("../naive_bayes/vectorisers/tokenizer_data.pkl")
    #     print("Vectorizer loaded")
        
    #     svd = joblib.load("../decision_tree/vectorizers/svd_tfidf.pkl")  # Confirm if needed
    #     print("SVD loaded (although unused)")

    #     #model = joblib.load("../naive_bayes/model/nb_model_tfidf.pkl")  # Check if file exists
    #     model = joblib.load("../naive_bayes/model/nb_model_scratch.pkl")  # Check if file exists
    #     print("Model loaded")

    #     label_encoder = joblib.load("../decision_tree/vectorizers/label_encoder.pkl")  # Confirm same encoder?
    #     print("Label encoder loaded")

    #     return model, vectorizer, svd, label_encoder
    try:
        print("Loading from-scratch Naive Bayes model...")
        with open("../naive_bayes/model/naive_bayes_full_model.pkl", "rb") as f:
            classifier = pickle.load(f)
        print("Naive Bayes model loaded successfully.")
        return classifier
    except Exception as e:
        print(f"❌ Error loading Naive Bayes model: {e}")
        return None


    except Exception as e:
        print(f"❌ Error loading Naive Bayes model: {e}")
        return None, None, None, None

def load_ann():
    try:
        print("Loading ANN models...")
        vectorizer = joblib.load("../ANN/vectorisers/tfidf_vectorizer.pkl")
        print("Vectorizer loaded")
        
        svd = joblib.load("../ANN/vectorizers/svd_tfidf.pkl")  # Confirm if needed
        print("SVD loaded (although unused)")

        model = joblib.load("../ANN/model/nb_model_tfidf.pkl")  # Check if file exists
        print("Model loaded")

        label_encoder = joblib.load("../ANN/vectorizers/label_encoder.pkl")  # Confirm same encoder?
        print("Label encoder loaded")

        return model, vectorizer, svd, label_encoder

    except Exception as e:
        print(f"❌ Error loading ANN model: {e}")
        return None, None, None, None

def load_clusterring():
    try:
        print("Loading Clusterring models...")
        vectorizer = joblib.load("../clusterring/vectorisers/tfidf_vectorizer.pkl")
        print("Vectorizer loaded")
        
        svd = joblib.load("../clusterring/vectorizers/svd_tfidf.pkl")  # Confirm if needed
        print("SVD loaded (although unused)")

        model = joblib.load("../clusterring/model/nb_model_tfidf.pkl")  # Check if file exists
        print("Model loaded")

        label_encoder = joblib.load("../clusterring/vectorizers/label_encoder.pkl")  # Confirm same encoder?
        print("Label encoder loaded")

        return model, vectorizer, svd, label_encoder

    except Exception as e:
        print(f"❌ Error loading Naive Bayes model: {e}")
        return None, None, None, None

def load_KNN():
    try:
        print("Loading KNN models...")
        vectorizer = joblib.load("../knn/vectorisers/tfidf_vectorizer.pkl")
        print("Vectorizer loaded")
        
        svd = joblib.load("../knn/vectorizers/svd_tfidf.pkl")  # Confirm if needed
        print("SVD loaded (although unused)")

        model = joblib.load("../knn/model/nb_model_tfidf.pkl")  # Check if file exists
        print("Model loaded")

        label_encoder = joblib.load("../knn/vectorizers/label_encoder.pkl")  # Confirm same encoder?
        print("Label encoder loaded")

        return model, vectorizer, svd, label_encoder

    except Exception as e:
        print(f"❌ Error loading Naive Bayes model: {e}")
        return None, None, None, None

def load_svm():
    try:
        print("Loading SVM models...")
        vectorizer = joblib.load("../svm/vectorisers/tfidf_vectorizer.pkl")
        print("Vectorizer loaded")
        
        svd = joblib.load("../svm/vectorizers/svd_tfidf.pkl")  # Confirm if needed
        print("SVD loaded (although unused)")

        model = joblib.load("../svm/model/nb_model_tfidf.pkl")  # Check if file exists
        print("Model loaded")

        label_encoder = joblib.load("../svm/vectorizers/label_encoder.pkl")  # Confirm same encoder?
        print("Label encoder loaded")

        return model, vectorizer, svd, label_encoder

    except Exception as e:
        print(f"❌ Error loading Naive Bayes model: {e}")
        return None, None, None, None

def load_logistic_regression():
    try:
        print("Loading Naive Bayes models...")
        vectorizer = joblib.load("../naive_bayes/vectorisers/tfidf_vectorizer.pkl")
        print("Vectorizer loaded")
        
        svd = joblib.load("../decision_tree/vectorizers/svd_tfidf.pkl")  # Confirm if needed
        print("SVD loaded (although unused)")

        model = joblib.load("../naive_bayes/model/nb_model_tfidf.pkl")  # Check if file exists
        print("Model loaded")

        label_encoder = joblib.load("../decision_tree/vectorizers/label_encoder.pkl")  # Confirm same encoder?
        print("Label encoder loaded")

        return model, vectorizer, svd, label_encoder

    except Exception as e:
        print(f"❌ Error loading Naive Bayes model: {e}")
        return None, None, None, None
