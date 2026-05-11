import math

import numpy as np
import streamlit as st


st.set_page_config(page_title="Student Dropout Predictor", layout="centered")

REPO_URL = "https://github.com/29KChenje/Machine-Learning-Practical-Assignment"
STREAMLIT_APP_URL = "https://machine-learning-practical-assignment.streamlit.app"
STREAMLIT_DEPLOY_URL = (
    "https://share.streamlit.io/deploy"
    "?repository=https://github.com/29KChenje/Machine-Learning-Practical-Assignment"
    "&branch=main"
    "&mainModule=app.py"
)

FEATURE_MEANS = np.array(
    [
        20.999357912475592,
        250.7803238855114,
        45.62769978904672,
        3.0714606916341562,
        74.71433638337868,
        4.99781594764003,
        5.009567047314036,
    ]
)
FEATURE_SCALES = np.array(
    [
        1.9992401081309397,
        95.58529985481992,
        20.409897706502473,
        1.5992351120646182,
        14.011155746797364,
        2.2337284545887717,
        1.9516049191411022,
    ]
)
MODEL_COEFFICIENTS = np.array(
    [
        -0.0004328230893874877,
        0.0032352145924426177,
        0.03717318371691494,
        -0.004979455357033702,
        -0.023962773255435194,
        0.0006049422232035966,
        0.012513245916764544,
    ]
)
MODEL_INTERCEPT = 0.5633504947351936


def predict_dropout_probability(student_features):
    scaled_features = (student_features - FEATURE_MEANS) / FEATURE_SCALES
    score = float(np.dot(scaled_features, MODEL_COEFFICIENTS) + MODEL_INTERCEPT)
    return 1 / (1 + math.exp(-score))


st.sidebar.title("Project links")
st.sidebar.link_button("Open live app", STREAMLIT_APP_URL, use_container_width=True)
st.sidebar.link_button("GitHub repository", REPO_URL, use_container_width=True)
st.sidebar.link_button("Deploy on Streamlit Cloud", STREAMLIT_DEPLOY_URL, use_container_width=True)

st.title("Student Dropout Predictor")
st.markdown("Predict student dropout risk based on academic factors")
st.divider()

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=10, max_value=100, value=20)
    family_income = st.number_input(
        "Family Income",
        min_value=0.0,
        max_value=10000.0,
        value=50.0,
    )
    transport_time = st.number_input(
        "Transport Time (min)",
        min_value=0.0,
        max_value=120.0,
        value=30.0,
    )
    study_hours = st.number_input(
        "Study Hours/Day",
        min_value=0.0,
        max_value=24.0,
        value=3.0,
    )

with col2:
    attendance_rate = st.number_input(
        "Attendance Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=75.0,
    )
    lms_logins = st.number_input("Weekly LMS Logins", min_value=0, max_value=100, value=5)
    stress_level = st.number_input(
        "Stress Level (1-10)",
        min_value=1.0,
        max_value=10.0,
        value=4.0,
    )

st.divider()

if st.button("Predict Outcome", use_container_width=True):
    student_features = np.array(
        [age, family_income, transport_time, study_hours, attendance_rate, lms_logins, stress_level]
    )
    dropout_probability = predict_dropout_probability(student_features)
    prediction = int(dropout_probability >= 0.5)

    st.metric("Dropout risk probability", f"{dropout_probability:.1%}")
    if prediction == 1:
        st.error("**High Risk of Dropout**")
        st.info("Consider providing additional academic support and counseling.")
    else:
        st.success("**Likely to Continue**")
        st.info("Student shows positive indicators for continuing studies.")
