import joblib
import pandas as pd
import numpy as np

# Load the trained model
model = joblib.load("models/logistic_model.pkl")

# Load dataset
df = pd.read_csv("data/transactions.csv")

# Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Pick one fraud example if no custom input
fraud_sample = X[y == 1].iloc[0].values.reshape(1, -1)

# Make prediction
prediction = model.predict(fraud_sample)
proba = model.predict_proba(fraud_sample)

print("Fraud Sample Input:", fraud_sample)
print("Prediction:", "Fraud Detected!" if prediction[0] == 1 else "Not Fraud")
print("Probabilities -> Not Fraud:", proba[0][0], "| Fraud:", proba[0][1])

