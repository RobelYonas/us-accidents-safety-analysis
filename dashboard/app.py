import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from pathlib import Path

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Traffic Safety Severity Predictor",
    page_icon="🚗",
    layout="wide"
)

# ── Load Model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent.parent / "models" / "accident_severity_model.pkl"
    return joblib.load(model_path)

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Could not load model: {e}")

# ── Feature Definitions ──────────────────────────────────────────────────────
# These must match the training features exactly
WEATHER_OPTIONS = [
    "Clear", "Cloudy", "Fair", "Fog", "Haze", "Heavy Rain", "Heavy Snow",
    "Light Drizzle", "Light Rain", "Light Snow", "Mist", "Mostly Cloudy",
    "Overcast", "Partly Cloudy", "Rain", "Scattered Clouds", "Snow",
    "Thunderstorm"
]

# The model was trained with drop_first=True, so we need to know the reference category
# Weather_Condition_Clear was the reference (dropped), so all other weather conditions
# are binary columns. Same for Sunrise_Sunset — "Day" was reference.

# ── Sidebar: User Inputs ─────────────────────────────────────────────────────
st.sidebar.header("🎛️ Accident Scenario")

with st.sidebar.form("prediction_form"):
    st.subheader("Weather Conditions")
    temperature = st.slider("Temperature (°F)", -20, 120, 65)
    humidity = st.slider("Humidity (%)", 0, 100, 50)
    pressure = st.slider("Pressure (in)", 25.0, 32.0, 29.9, step=0.1)
    visibility = st.slider("Visibility (mi)", 0.0, 10.0, 10.0, step=0.1)
    wind_speed = st.slider("Wind Speed (mph)", 0, 60, 5)
    precipitation = st.slider("Precipitation (in)", 0.0, 5.0, 0.0, step=0.1)
    weather = st.selectbox("Weather Condition", WEATHER_OPTIONS)

    st.subheader("Time & Location")
    hour = st.slider("Hour of Day", 0, 23, 12)
    month = st.selectbox("Month", list(range(1, 13)), index=6)
    day_of_week = st.selectbox("Day of Week", [
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
    ])
    is_weekend = 1 if day_of_week in ["Saturday", "Sunday"] else 0
    is_rush_hour = 1 if hour in [7, 8, 9, 16, 17, 18] else 0
    sunrise_sunset = st.selectbox("Time of Day", ["Day", "Night"])

    st.subheader("Road Features")
    distance = st.slider("Distance / Road Blockage (mi)", 0.0, 10.0, 0.5, step=0.1)
    crossing = st.checkbox("Near a Crossing", value=False)
    junction = st.checkbox("Near a Junction", value=False)
    traffic_signal = st.checkbox("Near a Traffic Signal", value=False)

    submitted = st.form_submit_button("🔮 Predict Severity", use_container_width=True)

# ── Main Area ────────────────────────────────────────────────────────────────
st.title("🚦 Traffic Safety Severity Predictor")
st.markdown("""
This app uses a machine learning model trained on **7.7 million US traffic accidents (2016–2023)**
to predict the likely severity of an accident given a specific scenario.

**Severity Scale:** 1 (minor) → 4 (severe)
""")

if submitted and model_loaded:
    # ── Build Feature Vector ───────────────────────────────────────────────
    # Base numeric features
    input_data = {
        "Distance(mi)": distance,
        "Temperature(F)": temperature,
        "Humidity(%)": humidity,
        "Pressure(in)": pressure,
        "Visibility(mi)": visibility,
        "Wind_Speed(mph)": wind_speed,
        "Precipitation(in)": precipitation,
        "Hour": hour,
        "Month": month,
        "DayOfWeek": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day_of_week),
        "IsWeekend": is_weekend,
        "IsRushHour": is_rush_hour,
        "Crossing": int(crossing),
        "Junction": int(junction),
        "Traffic_Signal": int(traffic_signal),
    }

    # One-hot encode weather (drop_first=True means Clear is reference)
    for w in WEATHER_OPTIONS:
        col_name = f"Weather_Condition_{w}"
        input_data[col_name] = 1 if weather == w else 0
    # Remove reference category
    if "Weather_Condition_Clear" in input_data:
        del input_data["Weather_Condition_Clear"]

    # One-hot encode sunrise/sunset (Day is reference)
    input_data["Sunrise_Sunset_Night"] = 1 if sunrise_sunset == "Night" else 0

    # Build DataFrame with exact column order
    feature_names = model.get_booster().feature_names
    X_input = pd.DataFrame([input_data])

    # Ensure all expected columns exist (fill missing with 0)
    for col in feature_names:
        if col not in X_input.columns:
            X_input[col] = 0
    X_input = X_input[feature_names]

    # ── Predict ──────────────────────────────────────────────────────────────
    pred_proba = model.predict_proba(X_input)[0]
    pred_class = model.predict(X_input)[0] + 1  # back to 1-4

    # ── Display Results ────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        st.metric("Predicted Severity", f"{int(pred_class)}")

    with col2:
        severity_labels = {1: "Minor", 2: "Moderate", 3: "Serious", 4: "Severe"}
        severity_colors = {1: "🟢", 2: "🟡", 3: "🟠", 4: "🔴"}
        st.metric("Category", f"{severity_colors[int(pred_class)]} {severity_labels[int(pred_class)]}")

    with col3:
        st.subheader("Probability Distribution")
        prob_df = pd.DataFrame({
            "Severity": ["1 (Minor)", "2 (Moderate)", "3 (Serious)", "4 (Severe)"],
            "Probability": pred_proba
        })
        st.bar_chart(prob_df.set_index("Severity"), use_container_width=True)

    # ── Risk Factors ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📊 What Drives This Prediction?")

    # Simple explanation based on input
    factors = []
    if precipitation > 0.2:
        factors.append("🌧️ **Precipitation** is elevated — wet roads increase risk")
    if visibility < 2:
        factors.append("🌫️ **Low visibility** — major contributor to severe accidents")
    if wind_speed > 25:
        factors.append("💨 **High wind speed** — can destabilize vehicles")
    if traffic_signal:
        factors.append("🚦 **Traffic signal present** — ranked as top predictor in the model")
    if distance > 2:
        factors.append("📏 **Long road blockage** — correlates with more severe incidents")
    if is_rush_hour:
        factors.append("⏰ **Rush hour** — higher traffic density, more complex scenarios")
    if not factors:
        factors.append("✅ Conditions appear relatively favorable based on the inputs provided.")

    for f in factors:
        st.markdown(f)

elif submitted and not model_loaded:
    st.error("Model not loaded. Please check that `models/accident_severity_model.pkl` exists.")

# ── EDA Section (Static Images) ──────────────────────────────────────────────
st.divider()
st.header("📈 Dataset Insights")
st.markdown("Key findings from the exploratory analysis on 7.7M records:")

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Weather & Time Patterns")
    img_path = Path(__file__).parent.parent / "outputs" / "eda_weather_time.png"
    if img_path.exists():
        st.image(str(img_path), use_container_width=True)
    else:
        st.info("Run the notebook to generate `eda_weather_time.png`")

with col_b:
    st.subheader("Road Infrastructure Impact")
    img_path = Path(__file__).parent.parent / "outputs" / "eda_road_features.png"
    if img_path.exists():
        st.image(str(img_path), use_container_width=True)
    else:
        st.info("Run the notebook to generate `eda_road_features.png`")

col_c, col_d = st.columns(2)

with col_c:
    st.subheader("Model Performance")
    img_path = Path(__file__).parent.parent / "outputs" / "confusion_matrix.png"
    if img_path.exists():
        st.image(str(img_path), use_container_width=True)
    else:
        st.info("Run the notebook to generate `confusion_matrix.png`")

with col_d:
    st.subheader("Feature Importance (SHAP)")
    img_path = Path(__file__).parent.parent / "outputs" / "shap_bar.png"
    if img_path.exists():
        st.image(str(img_path), use_container_width=True)
    else:
        st.info("Run the notebook to generate `shap_bar.png`")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
**About this project:** Built as a personal exploration into traffic safety data,
using Python, XGBoost, and SHAP. The model was trained on a 400K-row sample and achieves
~81% weighted accuracy. See the [GitHub repo](https://github.com/yourusername/us-accidents-safety-analysis) for the full notebook and methodology.

*Limitations: Class imbalance biases predictions toward Severity 2. Future work includes
geospatial clustering and class balancing.*
""")
