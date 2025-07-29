import streamlit as st
import pandas as pd
import joblib

# Load model and feature names
model = joblib.load("model.pkl")
feature_names = joblib.load("feature_names.pkl")

st.title("Heart Disease Prediction App")

# Input widgets in same order as feature_names
user_inputs = {}
for feature in feature_names:
    if feature in ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]:
        user_inputs[feature] = st.selectbox(f"{feature}", [0, 1, 2, 3])
    else:
        user_inputs[feature] = st.number_input(f"{feature}")

# Create DataFrame with exact matching columns
input_df = pd.DataFrame([user_inputs])

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.error("Prediction: You may have heart disease.")
    else:
        st.success("Prediction: You are unlikely to have heart disease.")
