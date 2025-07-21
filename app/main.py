from fastapi import FastAPI, Depends
from app.schema import TransactionData
from app.utils import predict
from app.auth import authenticate
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()

Instrumentator().instrument(app).expose(app, include_in_schema=False)

@app.get("/", dependencies=[Depends(authenticate)])
def root():
    return {"message": "Welcome to the Fraud Detection API"}

@app.post("/predict", dependencies=[Depends(authenticate)])
def predict_fraud(data: TransactionData):
    prediction = predict(data)
    return {"fraud": bool(prediction)}

