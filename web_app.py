import streamlit as st
import numpy as np
import joblib

st.title("🌱 Crop Recommendation System (GROUP2)")

# Load model
model = joblib.load("crop_model_GROUP2.pkl")
label_encoder = joblib.load("label_encoder_GROUP2.pkl")

N = st.number_input("Nitrogen")
P = st.number_input("Phosphorus")
K = st.number_input("Potassium")
temperature = st.number_input("Temperature")
humidity = st.number_input("Humidity")
ph = st.number_input("pH")
rainfall = st.number_input("Rainfall")

if st.button("Predict Crop"):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(features)
    crop = label_encoder.inverse_transform(prediction)[0]
    st.success(f"✅ Recommended Crop: {crop}")
