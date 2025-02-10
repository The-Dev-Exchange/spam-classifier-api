import pandas as pd
import re
import pickle
import mlflow
import mlflow.sklearn
from contractions import fix
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc

mlflow.set_tracking_uri('http://localhost:5001')

# nltk.download('stopwords')
# nltk.download('wordnet')

def preprocessing(dataset, num_of_rows=1):
    stemmer = WordNetLemmatizer()
    corpus = []

    for i in range(0, num_of_rows):
        document = fix(dataset[i])  # Expand contractions
        document = re.sub(r'\W', ' ', document)  # Remove special characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)  # Remove single characters
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)  # Remove single char from start
        document = re.sub(r'\s+', ' ', document, flags=re.I)  # Remove extra spaces
        document = document.lower()
        document = ' '.join([stemmer.lemmatize(w) for w in document.split()])
        corpus.append(document)

    return corpus

# Load Dataset
email_dataset = pd.read_csv("train.csv")

print(email_dataset.head())
print(email_dataset.groupby('label').count())


num_of_rows = email_dataset.shape[0]

x = preprocessing(email_dataset['sms'], num_of_rows)
y=email_dataset['label']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

tfidfvectorizer = TfidfVectorizer(max_features=3500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
x_train_tfidf = tfidfvectorizer.fit_transform(x_train)
x_test_tfidf=tfidfvectorizer.transform(x_test)


with mlflow.start_run():
    models = {
        "Logistic Regression": LogisticRegression(),
        "Naïve Bayes": MultinomialNB(),
        "SVM": SVC(probability=True, kernel="linear")
    }

    results = {}

    for name, model in models.items():
        model.fit(x_train_tfidf, y_train)
        y_predict = model.predict(x_test_tfidf)

        results[name] = {
            "Accuracy": accuracy_score(y_test, y_predict),
            "Report": classification_report(y_test, y_predict)
        }

        mlflow.sklearn.log_model(model, name)
        mlflow.log_metric(f"{name}_accuracy", results[name]["Accuracy"])

    for name, metrics in results.items():
        print(f"\n{name} Performance:\n")
        print(f"Accuracy: {metrics['Accuracy']:.4f}")
        print(metrics["Report"])


    classifier=SVC(probability=True, kernel="linear")
    classifier.fit(x_train_tfidf, y_train)

    pickle.dump(classifier, open("./model/spam_model.pkl", "wb"))
    pickle.dump(tfidfvectorizer, open("./model/vectorizer.pkl", "wb"))

    print("Model and vectorizer saved successfully!")

    mlflow.sklearn.log_model(classifier, "final_SVM_model")
    mlflow.log_artifact("./model/spam_model.pkl")
    mlflow.log_artifact("./model/vectorizer.pkl")


    param_dist = {
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # Kernel types
        'degree': [2, 3, 4],  # Only used for 'poly' kernel
    }

    random_search = RandomizedSearchCV(
        estimator=classifier, param_distributions=param_dist,
        n_iter=5, cv=5, scoring='accuracy', random_state=42, n_jobs=-1
    )

    random_search.fit(x_train_tfidf, y_train)
    print("Best Parameters:", random_search.best_params_)
    print("Best Cross-Validation Accuracy:", random_search.best_score_)

    mlflow.log_params(random_search.best_params_)
    mlflow.log_metric("best_cv_accuracy", random_search.best_score_)

