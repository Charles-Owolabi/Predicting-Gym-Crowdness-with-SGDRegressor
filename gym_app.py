import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
import datetime
import joblib
import os

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="🏋️ Gym Crowd Predictor", layout="wide")
st.title("🏋️ Gym Crowdness Predictor")
st.markdown("""
Plan your workout smarter by predicting gym crowd levels.
This ML-powered app forecasts how busy your gym will be based on the time and conditions you select.
""")

# --- Helper Functions ---
def engineer_features(df):
    """Add cyclical features and drop unnecessary columns."""
    df['date'] = pd.to_datetime(df['date'])
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)
    return df.drop(['date', 'hour', 'day_of_week', 'timestamp'], axis=1)

def build_pipeline():
    """Returns an ML pipeline for preprocessing and regression."""
    return Pipeline([
        ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler()),
        ('regressor', SGDRegressor(
            loss='huber', penalty='elasticnet', alpha=0.001,
            l1_ratio=0.7, learning_rate='adaptive', eta0=0.05,
            random_state=52, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=10
        ))
    ])

# --- Training Function ---
def train_and_save_model(path="gym_model.joblib"):
    try:
        df = pd.read_csv("crowdness_gym_data - small.csv")
    except FileNotFoundError:
        st.error("Dataset not found. Ensure 'crowdness_gym_data - small.csv' is in the app directory.")
        return None, None

    df = engineer_features(df)
    X = df.drop("number_people", axis=1)
    y = df["number_people"]
    feature_names = X.columns.tolist()

    model = build_pipeline()
    model.fit(X, y)

    joblib.dump({'model': model, 'feature_names': feature_names}, path)
    return model, feature_names

# --- Load or Train Model ---
@st.cache_resource
def load_or_train_model(path="gym_model.joblib"):
    if os.path.exists(path):
        saved = joblib.load(path)
        return saved['model'], saved['feature_names']
    else:
        st.info("Powered by AI, the app transforms your input into real-time crowd predictions—FE/23/76744852.")
        st.info("Charles Owolabi AI/ML — FE/23/76744852.")
        return train_and_save_model(path)

# Load model
model, feature_names = load_or_train_model()

# --- Sidebar: User Input ---
st.sidebar.header("Input Parameters")
now = datetime.datetime.now()

day_map = {day: idx for idx, day in enumerate(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])}
day_name = st.sidebar.selectbox("Day of Week", list(day_map), index=now.weekday())
day_index = day_map[day_name]

hour = st.sidebar.slider("Hour of Day", 0, 23, now.hour)
temperature = st.sidebar.slider("Temperature (°F)", 20, 100, 75)
month = st.sidebar.slider("Month", 1, 12, now.month)
is_weekend = 1 if day_index >= 5 else 0
is_holiday = st.sidebar.radio("Is it a holiday?", [0, 1], horizontal=True)
is_start_of_semester = st.sidebar.radio("Start of semester?", [0, 1], horizontal=True)
is_during_semester = st.sidebar.radio("During semester?", [0, 1], horizontal=True)

# --- Prediction ---
if model and feature_names and st.sidebar.button("Predict Crowdness", use_container_width=True):
    input_data = {
        'is_weekend': [is_weekend],
        'is_holiday': [is_holiday],
        'temperature': [temperature],
        'is_start_of_semester': [is_start_of_semester],
        'is_during_semester': [is_during_semester],
        'month': [month],
        'hour_sin': [np.sin(2 * np.pi * hour / 24.0)],
        'hour_cos': [np.cos(2 * np.pi * hour / 24.0)],
        'day_sin': [np.sin(2 * np.pi * day_index / 7.0)],
        'day_cos': [np.cos(2 * np.pi * day_index / 7.0)]
    }
    input_df = pd.DataFrame(input_data)[feature_names]
    prediction = int(round(model.predict(input_df)[0]))

    if prediction < 0:
        prediction = 0

    # --- Display Results ---
    st.header("📊 Prediction Result")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted People", prediction)

    with col2:
        if prediction < 20:
            st.success("✅ Not Busy")
            st.markdown("Great time to visit the gym!")
        elif 20 <= prediction < 40:
            st.warning("⚠️ Moderate Crowd")
            st.markdown("Expect some delays at peak machines.")
        else:
            st.error("🔴 Very Crowded")
            st.markdown("Consider going later to avoid the rush.")

    # --- Show Inputs ---
    with st.expander("🔎 Show Input Summary"):
        st.write(pd.DataFrame({
            "Feature": ["Day", "Hour", "Temperature (°F)", "Month", "Weekend?", "Holiday?", "Start of Semester?", "During Semester?"],
            "Value": [
                day_name, f"{hour}:00", temperature, month,
                "Yes" if is_weekend else "No",
                "Yes" if is_holiday else "No",
                "Yes" if is_start_of_semester else "No",
                "Yes" if is_during_semester else "No"
            ]
        }))
else:
    st.info("Use the sidebar to enter parameters and click 'Predict Crowdness' to get a forecast.")
