import joblib
import numpy as np

model = joblib.load("src/model/fraud_model.pkl")
scaler = joblib.load("src/model/scaler.pkl")

def predict(data):
    values = np.array([list(data.dict().values())])
    scaled = scaler.transform(values)
    pred = model.predict(scaled)
    return int(pred[0])
