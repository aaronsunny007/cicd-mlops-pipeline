from fastapi import FastAPI
import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/model.pkl")

app = FastAPI()

@app.get("/")
def home():
    return {"message": "CI/CD Failure Prediction API is running"}

@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]

    return {
        "prediction": int(prediction)
    }