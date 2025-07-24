import os
import mlflow
import mlflow.sklearn
import joblib

# Load the trained model
model = joblib.load("models/logistic_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Load metrics or use default
if os.path.exists("metrics.txt"):
    with open("metrics.txt", "r") as f:
        accuracy = float(f.read())
else:
    print("metrics.txt not found. Setting accuracy to 0.")
    accuracy = 0.0

# Use file-based tracking for CI
mlflow.set_tracking_uri("file:///tmp/mlruns")  # Local filesystem-based tracking
mlflow.set_experiment("Fraud Detection")

# Log run
with mlflow.start_run():
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")
    print("✅ Model and metrics logged successfully.")


