import mlflow
import mlflow.sklearn
import joblib

# Load the trained model
model = joblib.load("models/logistic_model.pkl")

# Load the scaler (if you want to track it too)
scaler = joblib.load("models/scaler.pkl")

# Load metrics
with open("metrics.txt", "r") as f:
    accuracy = float(f.read())

# Set tracking URI and experiment
mlflow.set_tracking_uri("http://localhost:5000")  # Change if using a remote MLflow server
mlflow.set_experiment("Fraud Detection")

# Start MLflow run
with mlflow.start_run():
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")
    print("Model and metrics logged successfully.")

