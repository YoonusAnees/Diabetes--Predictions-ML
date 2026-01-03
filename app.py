import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
import numpy as np

# =========================
# Load feature columns + Booster model (portable)
# =========================
feature_cols = joblib.load("v1_feature_columns.pkl")

booster = xgb.Booster()
booster.load_model("xgb_v1_model.json")

st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")
st.title("Diabetes Risk Predictor (XGBoost)")
st.write("Enter patient details to estimate diabetes probability.")
st.caption("Disclaimer: This tool provides an ML-based estimate and is not a medical diagnosis.")

# =========================
# Helper for numeric input
# =========================
def num_input(label, minv=None, maxv=None, val=None, step=None):
    kwargs = {}
    if minv is not None: kwargs["min_value"] = minv
    if maxv is not None: kwargs["max_value"] = maxv
    if val is not None: kwargs["value"] = val
    if step is not None: kwargs["step"] = step
    return st.number_input(label, **kwargs)

# =========================
# UI - All features
# =========================
st.header("1) Demographics")
age = num_input("Age (years)", 0, 120, 40, 1)

gender = st.selectbox("Gender", ["Male", "Female"], index=0)

ethnicity = st.selectbox(
    "Ethnicity",
    ["Asian", "Black", "White", "Hispanic", "Middle Eastern", "Indigenous", "Mixed", "Other", "Unknown"],
    index=8
)

education_level = st.selectbox(
    "Education Level",
    ["No schooling", "Primary", "Secondary", "High School", "Diploma", "Bachelor", "Master", "PhD", "Other", "Unknown"],
    index=9
)

income_level = st.selectbox(
    "Income Level",
    ["Low", "Lower-Middle", "Middle", "Upper-Middle", "High", "Other", "Unknown"],
    index=6
)

employment_status = st.selectbox(
    "Employment Status",
    ["Employed", "Self-employed", "Unemployed", "Student", "Retired", "Other", "Unknown"],
    index=6
)

st.header("2) Lifestyle")
alcohol = num_input("Alcohol consumption per week (units)", 0.0, 200.0, 0.0, 0.5)
physical_activity = num_input("Physical activity (minutes per week)", 0.0, 10000.0, 150.0, 10.0)
diet_score = num_input("Diet score (higher = healthier)", 0.0, 100.0, 50.0, 1.0)
sleep = num_input("Sleep hours per day", 0.0, 24.0, 7.0, 0.5)
screen_time = num_input("Screen time (hours per day)", 0.0, 24.0, 4.0, 0.5)

smoking_status = st.selectbox(
    "Smoking status",
    ["Never", "Former", "Current", "Occasional", "Unknown"],
    index=0
)

st.header("3) Body Measures")
bmi = num_input("BMI", 0.0, 80.0, 25.0, 0.1)
waist_to_hip_ratio = num_input("Waist-to-hip ratio", 0.0, 3.0, 0.9, 0.01)

st.header("4) Vital Signs")
systolic_bp = num_input("Systolic BP (mmHg)", 50.0, 250.0, 120.0, 1.0)
diastolic_bp = num_input("Diastolic BP (mmHg)", 30.0, 150.0, 80.0, 1.0)
heart_rate = num_input("Heart rate (bpm)", 30.0, 220.0, 75.0, 1.0)

st.header("5) Cholesterol & Lipids")
cholesterol_total = num_input("Total cholesterol", 0.0, 1000.0, 180.0, 1.0)
hdl_cholesterol = num_input("HDL cholesterol", 0.0, 300.0, 50.0, 1.0)
ldl_cholesterol = num_input("LDL cholesterol", 0.0, 500.0, 100.0, 1.0)
triglycerides = num_input("Triglycerides", 0.0, 1500.0, 150.0, 1.0)

st.header("6) Medical History")
family_history_diabetes = st.selectbox("Family history of diabetes", ["Yes", "No", "Unknown"], index=2)
hypertension_history = st.selectbox("History of hypertension", ["Yes", "No", "Unknown"], index=2)
cardiovascular_history = st.selectbox("Cardiovascular history", ["Yes", "No", "Unknown"], index=2)

# =========================
# Build raw row (original schema)
# =========================
raw = {
    "id": 0,
    "age": float(age),
    "alcohol_consumption_per_week": float(alcohol),
    "physical_activity_minutes_per_week": float(physical_activity),
    "diet_score": float(diet_score),
    "sleep_hours_per_day": float(sleep),
    "screen_time_hours_per_day": float(screen_time),
    "bmi": float(bmi),
    "waist_to_hip_ratio": float(waist_to_hip_ratio),
    "systolic_bp": float(systolic_bp),
    "diastolic_bp": float(diastolic_bp),
    "heart_rate": float(heart_rate),
    "cholesterol_total": float(cholesterol_total),
    "hdl_cholesterol": float(hdl_cholesterol),
    "ldl_cholesterol": float(ldl_cholesterol),
    "triglycerides": float(triglycerides),
    "gender": gender,
    "ethnicity": ethnicity,
    "education_level": education_level,
    "income_level": income_level,
    "smoking_status": smoking_status,
    "employment_status": employment_status,
    "family_history_diabetes": family_history_diabetes,
    "hypertension_history": hypertension_history,
    "cardiovascular_history": cardiovascular_history,
}

raw_df = pd.DataFrame([raw])

# One-hot encode to V1 style
v1_df = pd.get_dummies(raw_df)

# Align to training columns and make numeric
v1_df = v1_df.reindex(columns=feature_cols, fill_value=0).astype(np.float32)

# =========================
# Predict using Booster + DMatrix
# =========================
st.header("Prediction")

if st.button("Predict diabetes probability"):
    dmat = xgb.DMatrix(v1_df, feature_names=feature_cols)
    proba = float(booster.predict(dmat)[0])  # already probability for binary:logistic

    st.subheader("Result")
    st.write(f"Estimated probability of diagnosed diabetes: **{proba:.3f}**")

    if proba < 0.33:
        st.success("Risk level: Low")
    elif proba < 0.66:
        st.warning("Risk level: Medium")
    else:
        st.error("Risk level: High")

    with st.expander("Show model input vector (V1)"):
        st.dataframe(v1_df)
