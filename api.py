from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
import logging
from dotenv import load_dotenv

# Load env variables
load_dotenv()
API_KEY = os.getenv("API_KEY")
print(f"Loaded API_KEY: {API_KEY}")

# Setup FastAPI with custom title/description
app = FastAPI(
    title="Diabetes Risk Prediction API",
    description="Secure REST API for predicting diabetes risk using a trained ML model. Protected with Bearer token authentication.",
    version="1.0.0"
)

# Logging
logging.basicConfig(level=logging.INFO)

# Load model + scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# Input schema
class PatientData(BaseModel):
    pregnancies: int
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree: float
    age: int

@app.post("/predict")
def predict(data: PatientData, authorization: str = Header(...)):
    #  Verify API key
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    #  Process input
    features = np.array([[ 
        data.pregnancies, data.glucose, data.blood_pressure, data.skin_thickness,
        data.insulin, data.bmi, data.diabetes_pedigree, data.age
    ]])
    features_scaled = scaler.transform(features)

    #  Predict
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    #  Log request
    logging.info(f"Request: {data.dict()} → Prediction: {prediction}, Confidence: {probability:.4f}")

    return {
        "prediction": int(prediction),
        "confidence": round(probability * 100, 2)
    }
