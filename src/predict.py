import joblib
import pandas as pd
from fastapi import FastAPI

app = FastAPI()

# Load trained model
model = joblib.load("models/model.pkl")

@app.get("/")
def home():
    return {"message": "CI/CD MLOps Model API is running"}

@app.post("/predict")
def predict(feature1: float, feature2: float, feature3: float):

    data = pd.DataFrame([[feature1, feature2, feature3]],
                        columns=["feature1", "feature2", "feature3"])

    prediction = model.predict(data)

    return {"prediction": int(prediction[0])}