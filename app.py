import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

MODEL_PATH = "model.pkl"
FEATURE_COLUMNS = [
    "age",
    "family_income",
    "transport_time",
    "study_hours",
    "attendance_rate",
    "lms_logins",
    "stress_level",
]


def train_and_save_model():
    data = pd.read_csv("cleaned_dataset.csv")
    X = data[FEATURE_COLUMNS]
    y = data["dropout"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump((model, scaler), f)

    return model, scaler


def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return train_and_save_model()


model, scaler = load_model()

st.title("Student Dropout Predictor")
st.write("Enter student details and click Predict.")

age = st.number_input("Age", min_value=10, max_value=100, value=20)
family_income = st.number_input("Family income", min_value=0.0, max_value=10000.0, value=50.0)
transport_time = st.number_input("Transport time (minutes)", min_value=0.0, max_value=120.0, value=30.0)
study_hours = st.number_input("Study hours per day", min_value=0.0, max_value=24.0, value=3.0)
attendance_rate = st.number_input("Attendance rate (%)", min_value=0.0, max_value=100.0, value=75.0)
lms_logins = st.number_input("Weekly LMS logins", min_value=0, max_value=100, value=5)
stress_level = st.number_input("Stress level (1-10)", min_value=0.0, max_value=10.0, value=4.0)

if st.button("Predict"):
    features = np.array([
        [
            age,
            family_income,
            transport_time,
            study_hours,
            attendance_rate,
            lms_logins,
            stress_level,
        ]
    ])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)

    if prediction[0] == 1:
        st.error("Student likely to Dropout")
    else:
        st.success("Student likely to Continue")
