import joblib
import numpy as np

model = joblib.load("models/logistic_model.pkl")
scaler = joblib.load("models/scaler.pkl")

def predict(data):
    values = np.array([list(data.dict().values())])
    scaled = scaler.transform(values)
    pred = model.predict(scaled)
    return int(pred[0])
