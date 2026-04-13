
import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('C:/Users/HP/heart_disease_model.pkl')
scaler = joblib.load('C:/Users/HP/scaler.pkl')  # Remove if not used

st.title("Heart Disease Prediction App")

age = st.number_input("Age", min_value=1, max_value=120)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure")
chol = st.number_input("Cholesterol")
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Rest ECG", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved")
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("Oldpeak")
slope = st.selectbox("Slope of ST segment", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thal", [0, 1, 2, 3])

if sex=="Male":
    sex=1
else:
    sex=0
    
features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal]])

features = scaler.transform(features)  # Remove if not using scaler

if st.button("Predict"):
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.error("⚠ The model predicts risk of heart disease.")
    else:
        st.success("No heart disease predicted.")
