from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
import logging
from train_model import train_model, hyperparameter_tuning

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Create Flask App
def create_app():
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}})

    logging.info("Starting Flask Application...")

    if not os.path.exists("model/spam_model.pkl"):
        logging.info("No model found. Training a new model...")
        train_model()

    model = pickle.load(open("model/spam_model.pkl", "rb"))
    vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

    @app.route('/health', methods=['GET'])
    def healthcheck():
        return jsonify({"status": "Healthy"})

    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.json
        email_text = data.get("email", "")

        if not email_text:
            return jsonify({"error": "No email content provided"}), 400

        email_vector = vectorizer.transform([email_text])
        prediction = model.predict(email_vector)[0]
        result = "Spam" if prediction == 1 else "Not Spam"

        return jsonify({"result": result})

    @app.route('/train', methods=['GET'])
    def train():
        train_model()
        return jsonify({"message": "Model training completed!"})

    @app.route('/best-params', methods=['GET'])
    def best_params():
        params = hyperparameter_tuning()
        return jsonify({"best_params": params})

    return app

app = create_app()

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=9000) 
