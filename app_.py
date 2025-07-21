import streamlit as st
import numpy as np
import joblib

# Load your saved model + scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")
st.image("pic_diabetes_awareness.jpg", caption="Your health matters.", width=550)

st.title("🩺 Diabetes Risk Prediction App")
st.markdown("Enter your medical details to assess your diabetes risk.")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose Level", min_value=0.0)
blood_pressure = st.number_input("Blood Pressure", min_value=0.0)
skin_thickness = st.number_input("Skin Thickness", min_value=0.0)
insulin = st.number_input("Insulin Level", min_value=0.0)
bmi = st.number_input("BMI", min_value=0.0)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=1)

if st.button("Predict Diabetes Risk"):
    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                          insulin, bmi, diabetes_pedigree, age]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1] * 100

    if prediction == 1:
        st.error(f"⚠ You might be at risk for diabetes. Confidence: {probability:.2f}%")
    else:
        st.success(f"✅ You are likely not at high risk. Confidence: {probability:.2f}%")

st.caption("⚠ This is an AI estimate. Always consult a doctor.")
