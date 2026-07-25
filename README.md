# US Accidents Safety Analysis

Exploratory analysis and severity prediction on the [US Accidents (2016–2023)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents) dataset.

## Motivation
This project explores real-world traffic accident data to understand what drives accident severity — aligning with data-driven safety research in the automotive industry.

## What I Did
- **EDA**: Analyzed 7.7M records to explore weather, temporal, and road infrastructure patterns
- **Feature Engineering**: Extracted time-based features (hour, rush hour, weekend) and handled missing data
- **Modeling**: Trained an XGBoost classifier to predict accident severity (1–4)
- **Interpretability**: Applied SHAP analysis to identify top risk-driving features

## Key Findings
- Distance (road blockage length) and precipitation are strong severity predictors
- Traffic signals emerged as the top feature in the model — an interesting result worth deeper investigation
- Rush hour timing has less impact on severity than expected (likely due to lower speeds)
- Severe class imbalance limits the model's ability to predict Severity 3–4 accurately

## Tech Stack
Python, pandas, XGBoost, SHAP, matplotlib, seaborn, scikit-learn

## Outputs
| File | Description |
|------|-------------|
| `eda_weather_time.png` | Severity by weather condition and hour |
| `eda_road_features.png` | Impact of crossings, junctions, signals |
| `confusion_matrix.png` | Model performance |
| `feature_importance.png` | XGBoost feature rankings |
| `shap_bar.png` | SHAP importance across severity classes |

## Limitations & Future Work
- Used 400K-row sample (5% of data) for faster iteration
- Class imbalance biases predictions toward Severity 2
- Next: geospatial clustering, class balancing, interactive Streamlit dashboard

## Run It
The analysis was performed on Kaggle. To run locally:
```bash
pip install -r requirements.txt
jupyter notebook notebook/us_accidents_analysis.ipynb