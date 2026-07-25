import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Traffic Safety Severity Predictor",
    page_icon="🚗",
    layout="wide"
)

# ── Try Load Model ───────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent.parent / "models" / "accident_severity_model.pkl"
    return joblib.load(model_path)

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False

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
    weather = st.selectbox("Weather Condition", [
        "Clear", "Cloudy", "Fair", "Fog", "Haze", "Heavy Rain", "Heavy Snow",
        "Light Drizzle", "Light Rain", "Light Snow", "Mist", "Mostly Cloudy",
        "Overcast", "Partly Cloudy", "Rain", "Scattered Clouds", "Snow", "Thunderstorm"
    ])

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
This app explores **7.7 million US traffic accidents (2016–2023)** to understand
what drives accident severity — aligning with data-driven safety research in the automotive industry.

**Severity Scale:** 1 (minor) → 4 (severe)
""")

if submitted:
    if model_loaded:
        # Build feature vector (same as before)
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

        weather_options = ["Clear", "Cloudy", "Fair", "Fog", "Haze", "Heavy Rain", "Heavy Snow",
            "Light Drizzle", "Light Rain", "Light Snow", "Mist", "Mostly Cloudy",
            "Overcast", "Partly Cloudy", "Rain", "Scattered Clouds", "Snow", "Thunderstorm"]
        
        for w in weather_options:
            input_data[f"Weather_Condition_{w}"] = 1 if weather == w else 0
        if "Weather_Condition_Clear" in input_data:
            del input_data["Weather_Condition_Clear"]

        input_data["Sunrise_Sunset_Night"] = 1 if sunrise_sunset == "Night" else 0

        feature_names = model.get_booster().feature_names
        X_input = pd.DataFrame([input_data])
        for col in feature_names:
            if col not in X_input.columns:
                X_input[col] = 0
        X_input = X_input[feature_names]

        pred_proba = model.predict_proba(X_input)[0]
        pred_class = model.predict(X_input)[0] + 1

        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            st.metric("Predicted Severity", f"{int(pred_class)}")
        with col2:
            labels = {1: "Minor", 2: "Moderate", 3: "Serious", 4: "Severe"}
            colors = {1: "🟢", 2: "🟡", 3: "🟠", 4: "🔴"}
            st.metric("Category", f"{colors[int(pred_class)]} {labels[int(pred_class)]}")
        with col3:
            prob_df = pd.DataFrame({
                "Severity": ["1 (Minor)", "2 (Moderate)", "3 (Serious)", "4 (Severe)"],
                "Probability": pred_proba
            })
            st.bar_chart(prob_df.set_index("Severity"), use_container_width=True)

        # Risk factors
        st.divider()
        st.subheader("📊 What Drives This Prediction?")
        factors = []
        if precipitation > 0.2:
            factors.append("🌧️ **Precipitation** is elevated")
        if visibility < 2:
            factors.append("🌫️ **Low visibility** — major contributor")
        if wind_speed > 25:
            factors.append("💨 **High wind speed**")
        if traffic_signal:
            factors.append("🚦 **Traffic signal present** — top predictor in model")
        if distance > 2:
            factors.append("📏 **Long road blockage** — correlates with severity")
        if is_rush_hour:
            factors.append("⏰ **Rush hour** — higher traffic density")
        if not factors:
            factors.append("✅ Conditions appear relatively favorable.")
        for f in factors:
            st.markdown(f)

    else:
        st.info("""
        ⚠️ **Model not available in cloud deployment**
        
        The trained model file is too large to host on Streamlit Cloud. 
        The prediction feature works when running locally with the model file.
        
        Below you can explore the dataset insights and methodology from the analysis.
        """)

# ── EDA Gallery (Always Visible) ───────────────────────────────────────────
st.divider()
st.header("📈 Dataset Insights")
st.markdown("Key findings from exploratory analysis on 7.7M records:")

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Weather & Time Patterns")
    st.image("https://raw.githubusercontent.com/RobelYonas/us-accidents-safety-analysis/main/outputs/eda_weather_time.png", use_container_width=True)
with col_b:
    st.subheader("Road Infrastructure Impact")
    st.image("https://raw.githubusercontent.com/RobelYonas/us-accidents-safety-analysis/main/outputs/eda_road_features.png", use_container_width=True)

col_c, col_d = st.columns(2)
with col_c:
    st.subheader("Model Performance")
    st.image("https://raw.githubusercontent.com/RobelYonas/us-accidents-safety-analysis/main/outputs/confusion_matrix.png", use_container_width=True)
with col_d:
    st.subheader("Feature Importance (SHAP)")
    st.image("https://raw.githubusercontent.com/RobelYonas/us-accidents-safety-analysis/main/outputs/shap_bar.png", use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
**About this project:** Built as a personal exploration into traffic safety data,
using Python, XGBoost, and SHAP. See the 
[GitHub repo](https://github.com/RobelYonas/us-accidents-safety-analysis) 
for the full notebook and methodology.

*Limitations: Model trained on 400K-row sample, class imbalance biases toward Severity 2.*
""")
