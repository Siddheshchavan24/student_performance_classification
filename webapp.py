import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model and encoder
model = joblib.load("student_performance_model.pkl")
test_prep_encoder = joblib.load("test_prep_encoder.pkl")

# Streamlit UI
st.title("🎓 Student Performance Prediction")
st.write("Enter student details below to predict Pass/Fail")

# Input fields
test_prep = st.selectbox("Test Preparation Course", ["none", "completed"])
math_score = st.slider("Math Score", 0, 100, 50)
reading_score = st.slider("Reading Score", 0, 100, 50)
writing_score = st.slider("Writing Score", 0, 100, 50)

# Predict Button
if st.button("Predict Pass/Fail"):
    # Encode test preparation course
    test_prep_encoded = test_prep_encoder.transform([test_prep])[0]

    # Create input array
    input_data = np.array([[test_prep_encoded, math_score, reading_score, writing_score]])

    # Make prediction
    prediction = model.predict(input_data)[0]
    result = "✅ Pass" if prediction == 1 else "❌ Fail"

    # Display result
    st.subheader(f"Prediction: {result}")
