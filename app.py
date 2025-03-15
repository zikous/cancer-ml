import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and artifacts
model = joblib.load('model_artifacts/cervical_cancer_best_model.joblib')
scaler = joblib.load('model_artifacts/cervical_cancer_scaler.joblib')
feature_names = joblib.load('model_artifacts/feature_names.joblib')

# Default values
default_values = {
    "Age": 18,
    "Number of sexual partners": 4.0,
    "First sexual intercourse": 15.0,
    "Num of pregnancies": 1.0,
    "Smokes": 0.0,
    "Smokes (years)": 0.0,
    "Smokes (packs/year)": 0.0,
    "Hormonal Contraceptives": 0.0,
    "Hormonal Contraceptives (years)": 0.0,
    "IUD": 0.0,
    "IUD (years)": 0.0,
    "STDs": 0.0,
    "STDs (number)": 0.0,
    "STDs:condylomatosis": 0.0,
    "STDs:cervical condylomatosis": 0.0,
    "STDs:vaginal condylomatosis": 0.0,
    "STDs:vulvo-perineal condylomatosis": 0.0,
    "STDs:syphilis": 0.0,
    "STDs:pelvic inflammatory disease": 0.0,
    "STDs:genital herpes": 0.0,
    "STDs:molluscum contagiosum": 0.0,
    "STDs:AIDS": 0.0,
    "STDs:HIV": 0.0,
    "STDs:Hepatitis B": 0.0,
    "STDs:HPV": 0.0,
    "STDs: Number of diagnosis": 0.0,
    "Dx:Cancer": 0.0,
    "Dx:CIN": 0.0,
    "Dx:HPV": 0.0,
    "Dx": 0.0,
    "Hinselmann": 0.0,
    "Schiller": 0.0,
    "Citology": 0.0
}

st.title("Cervical Cancer Risk Prediction")

st.write("Enter your details to predict the risk:")

# Create input fields for user data
input_data = {}
for feature in feature_names:
    input_data[feature] = st.number_input(feature, value=default_values.get(feature, 0.0))

# Prediction button
if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    
    try:
        prediction_prob = model.predict_proba(input_scaled)[:, 1]
        prediction = prediction_prob[0]
    except:
        prediction = model.predict(input_scaled)[0]

    st.subheader("Prediction Result:")
    st.write(f"Risk Score: {prediction:.2f}")

