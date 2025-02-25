FROM python:3.9

# Set the working directory
WORKDIR /app

# Create a virtual environment and add it to PATH
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy backend files into the container
COPY . /app

# Upgrade pip and install dependencies from requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download required NLTK datasets
RUN python -c "import nltk; nltk.download('wordnet'); nltk.download('stopwords'); nltk.download('omw-1.4')"

# Expose ports for MLflow (default is 5000) and the Flask API via Gunicorn (9000)
EXPOSE 9000 5000

# Start MLflow and the Flask API concurrently.
# Note: The MLflow UI is started in the background with '&'.
CMD sh -c "mlflow ui & gunicorn -w 4 -b 0.0.0.0:9000 app:app"
