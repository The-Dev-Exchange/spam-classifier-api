import os
import logging
import pickle
import pandas as pd
import re
import mlflow
import mlflow.sklearn
from contractions import fix
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Download required NLTK datasets
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

mlflow.set_tracking_uri("http://localhost:5000")



# Preprocessing Function
def preprocess_text(dataset):
    lemmatizer = WordNetLemmatizer()
    corpus = []

    for text in dataset:
        text = fix(text)  # Expand contractions
        text = re.sub(r'\W', ' ', text)  # Remove special characters
        text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)  # Remove single characters
        text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)  # Remove single char from start
        text = re.sub(r'\s+', ' ', text, flags=re.I)  # Remove extra spaces
        text = text.lower()
        text = ' '.join([lemmatizer.lemmatize(w) for w in text.split()])
        corpus.append(text)

    return corpus


# Train Model
def train_model():
    logging.info("Loading dataset for training...")
    
    # Load dataset
    df = pd.read_csv("train.csv")
    x = preprocess_text(df["sms"])
    y = df["label"]

    # Split into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Vectorization
    vectorizer = TfidfVectorizer(max_features=3500, min_df=5, max_df=0.7, stop_words=stopwords.words("english"))
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_test_tfidf = vectorizer.transform(x_test)

    models = {
        "Logistic Regression": LogisticRegression(),
        "Naïve Bayes": MultinomialNB(),
        "SVM": SVC(probability=True, kernel="linear"),
    }

    best_model = None
    best_accuracy = 0

    # Start MLflow tracking
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Spam Classification")

    with mlflow.start_run():
        for name, model in models.items():
            logging.info(f"Training {name}...")
            model.fit(x_train_tfidf, y_train)
            y_pred = model.predict(x_test_tfidf)
            accuracy = accuracy_score(y_test, y_pred)

            # Log Metrics to MLflow
            mlflow.log_metric(f"{name}_accuracy", accuracy)

            if accuracy > best_accuracy:
                best_model = model
                best_accuracy = accuracy

        # Save Best Model & Vectorizer
        logging.info("Saving the best model...")
        os.makedirs("model", exist_ok=True)
        pickle.dump(best_model, open("model/spam_model.pkl", "wb"))
        pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))
        mlflow.sklearn.log_model(best_model, "spam_model")

    logging.info(f"Best model trained with accuracy: {best_accuracy:.4f}")

    return best_model, vectorizer, x_train_tfidf, y_train  # Return these for hyperparameter tuning


# Hyperparameter Tuning
def hyperparameter_tuning():
    logging.info("Starting hyperparameter tuning...")

    # Ensure model exists before tuning
    if not os.path.exists("model/spam_model.pkl"):
        logging.error("No trained model found! Training model first...")
        best_model, vectorizer, x_train_tfidf, y_train = train_model()
    else:
        best_model = pickle.load(open("model/spam_model.pkl", "rb"))
        vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

        # Reload dataset for hyperparameter tuning
        df = pd.read_csv("train.csv")
        x = preprocess_text(df["sms"])
        y = df["label"]
        x_train, _, y_train, _ = train_test_split(x, y, test_size=0.2, random_state=42)
        x_train_tfidf = vectorizer.transform(x_train)

    # Hyperparameter Grid
    param_dist = {
        "kernel": ["linear", "rbf", "poly", "sigmoid"],
        "degree": [2, 3, 4],
    }

    # Perform Hyperparameter Search
    search = RandomizedSearchCV(best_model, param_distributions=param_dist, n_iter=5, cv=5, scoring="accuracy", n_jobs=-1)
    
    with mlflow.start_run():
        search.fit(x_train_tfidf, y_train)
        best_params = search.best_params_
        mlflow.log_params(best_params)
        logging.info(f"Best hyperparameters: {best_params}")

    return best_params
