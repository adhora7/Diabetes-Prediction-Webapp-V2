
#  DIABETES PREDICTION APP - STREAMLIT

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

#Page Config
st.set_page_config(
    page_title="Diabetes Predictor",
    page_icon="🩺",
    layout="centered"
)

#Load Model, Scaler, Features
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("features.pkl", "rb") as f:
        features = pickle.load(f)
    return model, scaler, features

model, scaler, features = load_model()
all_features      = features["all_features"]
important_features = features["important_features"]

# ── Title ─────────────────────────────────────────────────────
st.title("🩺 Diabetes Prediction App")
st.markdown("**Powered by Random Forest + SHAP Feature Selection**")
st.markdown("---")

# ── Sidebar: About ────────────────────────────────────────────
with st.sidebar:
    st.header(" About This Model")
    st.markdown("""
    This model predicts diabetes using:
    - Clinical features (Glucose, BMI, etc.)
    - Morphological features (cvamp, stdamp, PowerHF)
    - SHAP-based feature selection (no redundancy)
    - Balanced class weighting (handles imbalance)
    """)
    st.markdown("---")
    st.markdown("**Important Features Used:**")
    for f in important_features:
        st.markdown(f"• `{f}`")

# ── Input Section ─────────────────────────────────────────────
st.subheader("Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    pregnancies    = st.number_input("Pregnancies",       min_value=0,   max_value=20,   value=1)
    glucose        = st.number_input("Glucose Level",     min_value=0,   max_value=300,  value=120)
    blood_pressure = st.number_input("Blood Pressure",    min_value=0,   max_value=200,  value=70)
    skin_thickness = st.number_input("Skin Thickness",    min_value=0,   max_value=100,  value=20)

with col2:
    insulin        = st.number_input("Insulin",           min_value=0,   max_value=900,  value=80)
    bmi            = st.number_input("BMI",               min_value=0.0, max_value=70.0, value=25.0)
    dpf            = st.number_input("Diabetes Pedigree", min_value=0.0, max_value=3.0,  value=0.5)
    age            = st.number_input("Age",               min_value=1,   max_value=120,  value=30)

# ── Build Feature Vector ──────────────────────────────────────
def build_features(preg, gluc, bp, skin, ins, bmi_val, dpf_val, age_val):
    # Base features
    data = {
        'Pregnancies':              preg,
        'Glucose':                  gluc,
        'BloodPressure':            bp,
        'SkinThickness':            skin,
        'Insulin':                  ins,
        'BMI':                      bmi_val,
        'DiabetesPedigreeFunction': dpf_val,
        'Age':                      age_val,
        # Morphological features (same as training)
        'cvamp':    gluc / 121.7,          # 121.7 = dataset mean glucose
        'stdamp':   bp   / 69.1,           # 69.1  = dataset mean BP
        'PowerHF':  (bmi_val * ins) / 1000,
        'age_glucose': age_val * gluc / 1000
    }
    return pd.DataFrame([data])

# ── Predict Button ────────────────────────────────────────────
if st.button("Predict Diabetes Risk", use_container_width=True):

    # Build raw input
    input_df = build_features(
        pregnancies, glucose, blood_pressure,
        skin_thickness, insulin, bmi, dpf, age
    )

    # Scale using saved scaler
    input_scaled = scaler.transform(input_df[all_features])
    input_scaled_df = pd.DataFrame(input_scaled, columns=all_features)

    # Select only important features
    input_final = input_scaled_df[important_features]

    # Predict
    prediction   = model.predict(input_final)[0]
    probability  = model.predict_proba(input_final)[0]

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠ **High Risk of Diabetes**")
        st.metric("Diabetes Probability", f"{probability[1]:.1%}")
    else:
        st.success(f" **Low Risk of Diabetes**")
        st.metric("No Diabetes Probability", f"{probability[0]:.1%}")

    # Progress bar for risk
    st.progress(float(probability[1]))
    st.caption(f"Risk Score: {probability[1]:.1%}")

    # ── SHAP Explanation ──────────────────────────────────────
    st.markdown("---")
    st.subheader(" Why this prediction? (SHAP Explanation)")
    st.markdown("This shows which features pushed the prediction toward or away from diabetes.")

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_final)

    # Show SHAP bar chart
    shap_df = pd.DataFrame({
        'Feature':    important_features,
        'SHAP Value': shap_values[1][0]
    }).sort_values('SHAP Value', key=abs, ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in shap_df['SHAP Value']]
    ax.barh(shap_df['Feature'], shap_df['SHAP Value'], color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel("SHAP Value (impact on prediction)")
    ax.set_title("Feature Impact — Red = increases risk, Green = decreases risk")
    plt.tight_layout()
    st.pyplot(fig)

    st.caption("🔴 Red bars = features pushing toward Diabetic | 🟢 Green bars = features pushing toward Non-Diabetic")

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("*Built with Random Forest + SHAP | Morphological features for robustness*")