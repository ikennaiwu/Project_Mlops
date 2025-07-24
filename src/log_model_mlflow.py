import mlflow
import mlflow.sklearn
import pickle

# Load saved model
with open("models/logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load metrics
with open("metrics.txt", "r") as f:
    accuracy = float(f.read())

# Start MLflow run
mlflow.set_tracking_uri("http://localhost:5000")  # or your remote server
mlflow.set_experiment("Fraud Detection")

with mlflow.start_run():
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")
