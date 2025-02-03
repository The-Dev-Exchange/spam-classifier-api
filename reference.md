## Reference steps for the backend api


### 1️⃣ Install Dependencies
```bash
pip install flask flask-cors scikit-learn pandas numpy
```

### 2️⃣ Create Flask Backend
```python
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Load the trained model and vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    email_text = data.get("email", "")

    if not email_text:
        return jsonify({"error": "No email content provided"}), 400

    # Transform the email text using the vectorizer
    email_vector = vectorizer.transform([email_text])

    # Predict using the model
    prediction = model.predict(email_vector)[0]
    result = "Spam" if prediction == 1 else "Not Spam"

    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True)

```


### 3️⃣ Train & Save Model
```python

import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Load dataset (replace with your dataset)
df = pd.read_csv("spam.csv", encoding="latin-1")  # Update path if needed
df = df[['v1', 'v2']]  # Keeping only relevant columns
df.columns = ['label', 'email']

# Convert labels to binary (Spam = 1, Not Spam = 0)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(df['email'], df['label'], test_size=0.2, random_state=42)

# Convert text to numerical data
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Save model & vectorizer
pickle.dump(model, open("spam_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model training completed and saved!")

```

```bash
    python train_model.py
```

```bash
    python app.py
```


