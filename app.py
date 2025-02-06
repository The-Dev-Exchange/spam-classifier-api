from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

model = pickle.load(open("./model/spam_model.pkl", "rb"))
vectorizer = pickle.load(open("./model/vectorizer.pkl", "rb"))

@app.route('/health', methods=['GET'])
def healthcheck():
    return jsonify({"status": "Healthy"})


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
