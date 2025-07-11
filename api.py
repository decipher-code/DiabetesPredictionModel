from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()  # 👈 this loads the .env into your environment

API_KEY = os.getenv("API_KEY")
print(f"Loaded API_KEY: {API_KEY}")

app = FastAPI()

model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

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
def predict(data: PatientData):
    features = np.array([[
        data.pregnancies, data.glucose, data.blood_pressure, data.skin_thickness,
        data.insulin,data.bmi,data.diabetes_pedigree,data.age
    ]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    return {"prediction": int(prediction)}    