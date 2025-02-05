from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing



@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    email_text = data.get("email", "")


    return jsonify({"result": "Spam"})

if __name__ == '__main__':
    app.run(debug=True)
