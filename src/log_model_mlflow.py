import os
import mlflow
import mlflow.sklearn
import joblib

# Load the trained model
model = joblib.load("models/logistic_model.pkl")

# Load the scaler (optional tracking, not logged here)
scaler = joblib.load("models/scaler.pkl")

# Load metrics with fallback
if os.path.exists("metrics.txt"):
    with open("metrics.txt", "r") as f:
        accuracy = float(f.read())
else:
    print("metrics.txt not found. Setting accuracy to 0.")
    accuracy = 0.0

# Set MLflow tracking URI and experiment
mlflow.set_tracking_uri("http://localhost:5000")  # Change this if you're using a remote server
mlflow.set_experiment("Fraud Detection")

# Start and log the run
with mlflow.start_run():
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")
    print("Model and metrics logged successfully.")

