# startup_success_app.py
# Ready-to-run Streamlit app for Startup Success Prediction
# Files required in the same folder:
#   - xgb_model_retrained.pkl  (XGBClassifier trained on 8 features)
#   - scaler.pkl               (StandardScaler fitted on the same 8 features)

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# -----------------------
# Config & constants
# -----------------------
st.set_page_config(page_title="Startup Success Predictor", layout="centered", page_icon="🚀")
FEATURES = [
    "funding_total_usd",
    "funding_rounds",
    "startup_age",
    "funding_per_round",
    "funding_per_year",
    "funding_rounds_per_year",
    "sector_popularity",
    "is_USA",
]

MODEL_FILE = "xgb_model_retrained.pkl"
SCALER_FILE = "scaler.pkl"

# -----------------------
# Load model + scaler
# -----------------------
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_FILE):
        st.stop()  # clear error after message
    if not os.path.exists(SCALER_FILE):
        st.stop()

    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_FILE, "rb") as f:
        scaler = pickle.load(f)

    # SHAP explainer (model-agnostic API; works with XGBClassifier)
    explainer = shap.Explainer(model)
    return model, scaler, explainer

try:
    model, scaler, explainer = load_artifacts()
except Exception as e:
    st.error(f"Failed to load model/scaler: {e}")
    st.stop()

# -----------------------
# UI
# -----------------------
st.title("🚀 Startup Success Predictor")
st.write(
    "Enter the base startup details below. The app will derive the rate features, "
    "scale them the same way as training, predict success likelihood, and explain the result."
)

with st.form("inputs"):
    col1, col2 = st.columns(2)

    with col1:
        funding_total_usd = st.number_input(
            "Total Funding (USD)", min_value=0.0, value=500000.0, step=10000.0, format="%.2f", key="total_funding"
        )
        funding_rounds = st.number_input(
            "Number of Funding Rounds", min_value=1, value=3, step=1, key="rounds"
        )
        sector_popularity = st.number_input(
            "Sector Popularity (no. of startups in same sector)", min_value=1, value=600, step=10, key="sector_pop"
        )

    with col2:
        startup_age = st.number_input(
            "Startup Age (Years)", min_value=0.1, value=4.5, step=0.5, format="%.2f", key="age"
        )
        is_usa_label = st.selectbox("Is the Startup based in the USA?", ["Yes", "No"], key="usa_sel")
        is_USA = 1 if is_usa_label == "Yes" else 0

    submitted = st.form_submit_button("Predict Success Probability")


# -----------------------
# Build feature row (derive consistent features)
# -----------------------
def build_feature_row(total_funding, n_rounds, age, sector_pop, usa_flag):
    # prevent division by zero while keeping meaningful ratios
    safe_rounds = max(n_rounds, 1)
    safe_age = max(age, 0.1)

    funding_per_round = total_funding / safe_rounds
    funding_per_year = total_funding / safe_age
    funding_rounds_per_year = n_rounds / safe_age

    row = {
        "funding_total_usd": total_funding,
        "funding_rounds": n_rounds,
        "startup_age": age,
        "funding_per_round": funding_per_round,
        "funding_per_year": funding_per_year,
        "funding_rounds_per_year": funding_rounds_per_year,
        "sector_popularity": sector_pop,
        "is_USA": usa_flag,
    }
    return pd.DataFrame([row], columns=FEATURES)

# --- Validation helper ---
def validate_inputs(total_funding: float, rounds: int, age: float, sector_pop: int):
    errors = []
    if total_funding < 0:
        errors.append("Total funding cannot be negative.")
    if rounds < 1:
        errors.append("Funding rounds must be ≥ 1.")
    if age <= 0:
        errors.append("Startup age must be > 0.")
    if sector_pop < 1:
        errors.append("Sector popularity must be ≥ 1.")
    # Optional: block the “all zeros/near zero” case
    if total_funding == 0 and rounds == 1 and age <= 0.1 and sector_pop == 1:
        errors.append("Inputs are too small to make a meaningful prediction.")
    return errors


# -----------------------
# Predict + Explain
# -----------------------
if submitted:
    # 1) Validate base inputs
    errs = validate_inputs(funding_total_usd, funding_rounds, startup_age, sector_popularity)
    if errs:
        st.error("Please fix the following:")
        for e in errs:
            st.write("• " + e)
        st.stop()   # <-- prevents the app from continuing to predict


    # 2) Build feature row (derived features stay consistent)
    input_df = build_feature_row(
        funding_total_usd, funding_rounds, startup_age, sector_popularity, is_USA
    )

    # 3) Optionally: block nearly-zero info (very weak signal)
    if (
        funding_total_usd == 0
        or (funding_rounds == 0 and startup_age == 0)
        or sector_popularity == 0
    ):
        st.warning("⚠️ Not enough signal in inputs to make a meaningful prediction. Please provide non‑zero values.")
        st.stop()

    # 4) Scale + predict
    scaled = scaler.transform(input_df)
    proba = float(model.predict_proba(scaled)[0][1])
    st.metric("Predicted Success Likelihood", f"{proba*100:.2f}%")

    # 5) SHAP on scaled
    scaled_df = pd.DataFrame(scaled, columns=input_df.columns)
    shap_values = explainer(scaled_df)

    # (Optional but recommended) show ORIGINAL input values in the left labels
    expl = shap_values[0]
    expl.feature_names = input_df.columns.tolist()          # use your column names
    expl.data = input_df.iloc[0].values                     # show raw (unscaled) values next to names

    st.subheader("🔍 Feature Impact Explanation")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(expl, max_display=len(input_df.columns), show=False)
    st.pyplot(fig)

    with st.expander("Debug (features & scaled values)"):
        st.write("Input features:", input_df)
        st.write("Scaled features:", scaled_df)
        st.write("Raw model proba:", proba)



st.markdown("---")
st.caption("Built for MSc Data Science project • XGBoost + SHAP • 8-feature model with consistent derived features")
