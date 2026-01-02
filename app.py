import streamlit as st
import pandas as pd
import joblib

# Load trained model + expected columns
model = joblib.load("xgb_v1_model.pkl")
feature_cols = joblib.load("v1_feature_columns.pkl")

st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")
st.title("Diabetes Risk Predictor (XGBoost)")
st.write("Enter patient details to estimate diabetes probability.")

# ---- User inputs (keep it simple but meaningful) ----
age = st.number_input("Age", min_value=0, max_value=120, value=40)
bmi = st.number_input("BMI", min_value=0.0, max_value=80.0, value=25.0)
waist = st.number_input("Waist-to-Hip Ratio", min_value=0.0, max_value=3.0, value=0.9)

sys_bp = st.number_input("Systolic BP", min_value=50.0, max_value=250.0, value=120.0)
dia_bp = st.number_input("Diastolic BP", min_value=30.0, max_value=150.0, value=80.0)

gender = st.selectbox("Gender", ["Male", "Female"])
smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])

# ---- Build a raw row (same original feature names before one-hot) ----
raw = {
    "id": 0,
    "age": age,
    "alcohol_consumption_per_week": 0.0,
    "physical_activity_minutes_per_week": 0.0,
    "diet_score": 0.0,
    "sleep_hours_per_day": 0.0,
    "screen_time_hours_per_day": 0.0,
    "bmi": bmi,
    "waist_to_hip_ratio": waist,
    "systolic_bp": sys_bp,
    "diastolic_bp": dia_bp,
    "heart_rate": 0.0,
    "cholesterol_total": 0.0,
    "hdl_cholesterol": 0.0,
    "ldl_cholesterol": 0.0,
    "triglycerides": 0.0,
    "gender": gender,
    "ethnicity": "Unknown",
    "education_level": "Unknown",
    "income_level": "Unknown",
    "smoking_status": smoking,
    "employment_status": "Unknown",
    "family_history_diabetes": "Unknown",
    "hypertension_history": "Unknown",
    "cardiovascular_history": "Unknown",
}

raw_df = pd.DataFrame([raw])

# ---- Convert raw_df -> V1 (one-hot) ----
v1_df = pd.get_dummies(raw_df)

# Align to training feature columns
v1_df = v1_df.reindex(columns=feature_cols, fill_value=0)

if st.button("Predict"):
    proba = float(model.predict_proba(v1_df)[:, 1][0])

    st.subheader("Prediction")
    st.write(f"Estimated probability of diabetes: **{proba:.3f}**")

    if proba < 0.33:
        st.success("Risk level: Low")
    elif proba < 0.66:
        st.warning("Risk level: Medium")
    else:
        st.error("Risk level: High")
