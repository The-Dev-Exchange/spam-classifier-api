# Email Spam Detection API
## Use Case
The **Email Spam Detection** system classifies emails as **Spam** or **Not Spam** using **Natural Language Processing (NLP) techniques**. This helps in filtering unwanted emails and improving inbox organization by leveraging **Machine Learning models** trained on a labeled spam dataset.
## Dataset
The model is trained using the **Spam Dataset**, which consists of labeled email samples categorized as:
- **Spam (1)**
- **Not Spam (0)**
The dataset is preprocessed using NLP techniques such as **tokenization, stop-word removal, and TF-IDF vectorization**.
## 📁 Directory Structure


```bash
spam-classifier-api/
│── hyperparameter_tuning/           # Work done by team to optimize various hyperparameters 
│   ├── abhishek/                    # Contributions by Abhishek
│   ├── archita/                      # Contributions by Archita
│   ├── narayan/                      # Contributions by Narayan
│   │   ├── hyperparameter_logging.png      # Logging of hyperparameter tuning
│   │   ├── model_accuracy.png              # Accuracy results of tuning
│   │   ├── overall_tuning_mlflow_setup.png # MLflow setup visualization
│   │   ├── response.json                   # JSON response of tuning results
│   ├── shivraj/                      # Contributions by Shivraj
│
│── model/                             # Directory for trained model artifacts
│── spam-classifier-fe/                 # Front-end application for spam classification
│── app.py                               # Root file of the classifier
│── requirements.txt                      # Dependencies and package requirements
│── train_model.py                        # Model training file
```


## Getting Started
### Prerequisites
Make sure you have Docker installed on your machine.

### Build and Start the Application
Run the following command to build and start the application using Docker:

```bash
docker-compose up --build
```

### Application UI  
Once the application is running, access the **Spam Classifier UI** at:  
🔗 **[http://localhost:3000/](http://localhost:3000/)**


## 🔗 API Endpoints & Usage

###  Health Check
- **Endpoint:** `/health`
- **Method:** `GET`
- **Response:**

```json
  {
    "status": "Healthy"
  }
```
- Description: Checks if the API is running.


###  Predict Email Spam
- **Endpoint:** `/predict`
- **Method:** `POST`
- **Response:**

```json
 {
    "email": "Congratulations! You have won a free gift card."
}
```
- Response:

```json
{
  "result": "Spam"
}
```
- Description: Takes an email as input and predicts whether it is Spam or Not Spam.

### Train the Model
- **Endpoint:** `/train`
- **Method:** `GET`
- **Response:**

```json
  {
  "message": "Model training completed!"
}

```
- Description: Triggers model training using the dataset.

### Get Best Hyperparameters
- **Endpoint:** `/best-params`
- **Method:** `GET`
- **Response:**

```json
  {
  "best_params": {
    "C": 1.0,
    "solver": "lbfgs"
  }
}

```
- Description: Returns the best hyperparameters for the model after hyperparameter tuning.

## Technologies Used
- **Python**  - Core programming language used for model development.
- **Flask** - Lightweight web framework for building the API.
- **Docker & Docker Compose** - Containerization for easy deployment and scalability.
- **Scikit-learn** - Machine learning library used for training and classification.
- **NLTK & TF-IDF** - Natural language processing tools for text preprocessing.
- **MLflow** - Experiment tracking and hyperparameter tuning for model optimization.


## Contributors
- **Narayan Khanna**  
- **Abhishek**  
- **Archita**  
- **Shivraj**  

## Contact
For any queries, reach out via **email** or **GitHub Issues**.
