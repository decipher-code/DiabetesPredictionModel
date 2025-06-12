import streamlit as st
import numpy as np
import joblib 

# Load model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page config
st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")
st.image("pic_diabetes_awareness.jpg", caption="Your health matters.", width=550)

st.title("🩺 Diabetes Risk Prediction App")
st.markdown("Enter your medical details to assess your diabetes risk. This tool provides insights and helpful resources.")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose Level", min_value=0.0)
blood_pressure = st.number_input("Blood Pressure", min_value=0.0)
skin_thickness = st.number_input("Skin Thickness", min_value=0.0)
insulin = st.number_input("Insulin Level", min_value=0.0)
bmi = st.number_input("BMI", min_value=0.0)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=1)

# Prediction
if st.button("Predict Diabetes Risk"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("⚠️ You might be at risk for diabetes.")
        st.markdown("### 🧭 Recommended Actions")
        st.markdown("""
- 🏃‍♂️ **Exercise regularly**: Try at least 30 minutes a day, 5 times a week.  
    👉 [Visit CDC Exercise Guidelines](https://www.cdc.gov/diabetes/managing/active.html)

- 🥗 **Eat a balanced diet**: Reduce sugar, refined carbs, and saturated fats.  
    👉 [Visit Harvard Healthy Eating Plate](https://www.hsph.harvard.edu/nutritionsource/healthy-eating-plate/)

- 💧 **Stay hydrated** and avoid sugary drinks.  
- 📉 **Monitor blood sugar levels** if you have access.  
- 🩺 **See a healthcare provider** for further tests and advice.  
        """)

    else:
        st.success("✅ You are likely not at high risk. Keep maintaining a healthy lifestyle!")
        st.markdown("### 🌿 Healthy Habits to Maintain")
        st.markdown("""
- 🍎 **Continue a healthy diet**  
- 🏃‍♀️ **Stay physically active**  
- 🚭 **Avoid smoking and excessive alcohol**  
- ⏱️ **Get regular checkups**, especially as you age  
- 🧘 **Manage stress** through mindfulness or relaxation techniques  
    👉 [Visit Stress Management - WHO](https://www.who.int/news-room/fact-sheets/detail/stress)

- 📘 [Learn more about preventing diabetes (CDC)](https://www.cdc.gov/diabetes/prevention/index.html)
        """)

# Footer
st.caption("⚠️ This is an AI-based estimate. Always consult a medical professional for medical advice or diagnosis.")
