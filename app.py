import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Student Dropout Predictor", layout="centered")

@st.cache_resource
def load_and_train_model():
    try:
        df = pd.read_csv("cleaned_dataset.csv")
        
        features = ["age", "family_income", "transport_time", "study_hours", 
                   "attendance_rate", "lms_logins", "stress_level"]
        X = df[features].values
        y = df["dropout"].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_scaled, y)
        
        return model, scaler
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None, None

st.title("🎓 Student Dropout Predictor")
st.markdown("Predict student dropout risk based on academic factors")
st.divider()

model, scaler = load_and_train_model()

if model is None:
    st.error("Failed to load model. Please contact support.")
else:
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=10, max_value=100, value=20)
        family_income = st.number_input("Family Income", min_value=0.0, max_value=10000.0, value=50.0)
        transport_time = st.number_input("Transport Time (min)", min_value=0.0, max_value=120.0, value=30.0)
        study_hours = st.number_input("Study Hours/Day", min_value=0.0, max_value=24.0, value=3.0)
    
    with col2:
        attendance_rate = st.number_input("Attendance Rate (%)", min_value=0.0, max_value=100.0, value=75.0)
        lms_logins = st.number_input("Weekly LMS Logins", min_value=0, max_value=100, value=5)
        stress_level = st.number_input("Stress Level (1-10)", min_value=1.0, max_value=10.0, value=4.0)
    
    st.divider()
    
    if st.button("🔮 Predict Outcome", use_container_width=True):
        features = np.array([[age, family_income, transport_time, study_hours, 
                             attendance_rate, lms_logins, stress_level]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        
        if prediction == 1:
            st.error("⚠️ **High Risk of Dropout**")
            st.info("Consider providing additional academic support and counseling.")
        else:
            st.success("✅ **Likely to Continue**")
            st.info("Student shows positive indicators for continuing studies.")
