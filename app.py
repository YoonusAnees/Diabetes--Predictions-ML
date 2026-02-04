import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
import numpy as np

# ────────────────────────────────────────────────
# Page config + custom styling
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern dark/midnight theme with accent colors
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #e2e8f0;
    }

    .stButton > button {
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1.6rem;
        font-weight: 600;
        transition: all 0.25s;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.35);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(99, 102, 241, 0.5);
    }

    .stButton > button:active {
        transform: translateY(1px);
    }

    h1, h2, h3 {
        color: #c7d2fe;
        font-weight: 600;
    }

    .stExpander {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(99, 102, 241, 0.18);
        border-radius: 12px;
        color: #cbd5e1;
    }

    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        background: rgba(30, 41, 59, 0.7);
        color: #e2e8f0;
        border: 1px solid #475569;
        border-radius: 10px;
    }

    .stNumberInput label, .stSelectbox label {
        color: #94a3b8 !important;
    }

    hr {
        border-color: rgba(99, 102, 241, 0.25);
        margin: 2.2rem 0 1.6rem;
    }

    .stSuccess, .stWarning, .stError {
        border-radius: 12px;
        padding: 1.2rem !important;
        margin: 1rem 0;
    }

    .metric-card {
        background: rgba(30, 41, 59, 0.65);
        border-radius: 16px;
        padding: 1.6rem;
        border: 1px solid rgba(139, 92, 246, 0.2);
        text-align: center;
        margin: 1.5rem 0;
    }

    .big-prob {
        font-size: 3.8rem;
        font-weight: 700;
        color: #c084fc;
        line-height: 1;
    }
    </style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────
# Load model & features
# ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    feature_cols = joblib.load("v1_feature_columns.pkl")
    booster = xgb.Booster()
    booster.load_model("xgb_v1_model.json")
    return feature_cols, booster

feature_cols, booster = load_model()

# ────────────────────────────────────────────────
# Header
# ────────────────────────────────────────────────
st.title("🩺 Diabetes Risk Estimator")
st.caption("Machine learning–based probability estimate — **not a medical diagnosis**")
st.markdown("---")

# ────────────────────────────────────────────────
# Input layout — using columns for better UX
# ────────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Demographics & History")
    
    age = st.number_input("Age (years)", 0, 120, 42, 1, key="age")
    gender = st.selectbox("Gender", ["Male", "Female"], index=0)
    ethnicity = st.selectbox("Ethnicity", ["Asian", "Black", "White", "Hispanic", "Middle Eastern", "Indigenous", "Mixed"])
    
    with st.expander("More background (optional)"):
        education_level = st.selectbox("Education Level", ["No schooling", "Primary", "Secondary", "High School", "Diploma", "Bachelor", "Master", "PhD"])
        income_level = st.selectbox("Income Level", ["Low", "Lower-Middle", "Middle", "Upper-Middle", "High"])
        employment_status = st.selectbox("Employment Status", ["Employed", "Self-employed", "Unemployed", "Student", "Retired"])

with col2:
    st.subheader("Lifestyle & Body Measures")
    
    bmi = st.number_input("BMI", 10.0, 70.0, 25.0, 0.1, format="%.1f")
    waist_to_hip = st.number_input("Waist-to-hip ratio", 0.5, 2.0, 0.90, 0.01, format="%.2f")
    
    physical_activity = st.slider("Physical activity (min/week)", 0, 3000, 150, 30, help="150–300 min/week is usually recommended")
    sleep = st.slider("Average sleep (hours/day)", 3.0, 12.0, 7.0, 0.5)
    screen_time = st.slider("Screen time (hours/day)", 0.0, 16.0, 4.0, 0.5)

# ────────────────────────────────────────────────
# Grouped inputs — second row
# ────────────────────────────────────────────────
st.markdown("### Health Markers")

c1, c2, c3 = st.columns(3)

with c1:
    smoking = st.selectbox("Smoking status", ["Never", "Former", "Current", "Occasional"])
    alcohol = st.number_input("Alcohol (units/week)", 0.0, 120.0, 0.0, 0.5, format="%.1f")

with c2:
    systolic = st.number_input("Systolic BP (mmHg)", 70, 220, 120, 1)
    diastolic = st.number_input("Diastolic BP (mmHg)", 40, 140, 80, 1)

with c3:
    family_diabetes = st.radio("Family history of diabetes", ["Yes", "No"], horizontal=True, index=1)
    hypertension = st.radio("History of hypertension", ["Yes", "No"], horizontal=True, index=1)

# ────────────────────────────────────────────────
# Prediction section
# ────────────────────────────────────────────────
st.markdown("---")

if st.button("Calculate Diabetes Risk", type="primary", use_container_width=True):
    with st.spinner("Processing..."):
        # Build input dictionary
        raw = {
            "id": 0,
            "age": float(age),
            "alcohol_consumption_per_week": float(alcohol),
            "physical_activity_minutes_per_week": float(physical_activity),
            "diet_score": 50.0,  # ← you removed it — using neutral default
            "sleep_hours_per_day": float(sleep),
            "screen_time_hours_per_day": float(screen_time),
            "bmi": float(bmi),
            "waist_to_hip_ratio": float(waist_to_hip),
            "systolic_bp": float(systolic),
            "diastolic_bp": float(diastolic),
            "heart_rate": 75.0,          # ← missing in UI — neutral default
            "cholesterol_total": 180.0,  # ← missing — neutral
            "hdl_cholesterol": 50.0,
            "ldl_cholesterol": 100.0,
            "triglycerides": 150.0,
            "gender": gender,
            "ethnicity": ethnicity,
            "education_level": education_level,
            "income_level": income_level,
            "smoking_status": smoking,
            "employment_status": employment_status,
            "family_history_diabetes": family_diabetes,
            "hypertension_history": hypertension,
            "cardiovascular_history": "No",  # ← missing — default
        }

        df = pd.DataFrame([raw])
        df_encoded = pd.get_dummies(df)
        df_final = df_encoded.reindex(columns=feature_cols, fill_value=0).astype(np.float32)

        dmat = xgb.DMatrix(df_final, feature_names=feature_cols)
        proba = float(booster.predict(dmat)[0])

    # ── Result presentation ───────────────────────────────
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size:1.1rem; color:#94a3b8; margin-bottom:0.6rem;">
            Estimated probability of diabetes
        </div>
        <div class="big-prob">{proba:.1%}</div>
    </div>
    """, unsafe_allow_html=True)

    if proba < 0.25:
        st.success("**Low risk** — keep up healthy habits!")
    elif proba < 0.50:
        st.warning("**Moderate risk** — consider lifestyle review")
    elif proba < 0.75:
        st.error("**Elevated risk** — medical consultation recommended")
    else:
        st.error("**High risk** — please consult a doctor soon")

    with st.expander("Detailed model input (one-hot encoded)", expanded=False):
        st.dataframe(df_final.style.format("{:.3f}"))

st.caption("Model: XGBoost • Features aligned to training set • Probability output from logistic objective")