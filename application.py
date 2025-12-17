import streamlit as st
import numpy as np
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Crop Recommendation System (GROUP2)",
    page_icon="🌱",
    layout="centered"
)

st.title("🌱 Crop Recommendation System (GROUP2)")

# Get the directory of this script
BASE_DIR = os.path.dirname(__file__)

# Load trained model and label encoder
try:
    model_path = os.path.join(BASE_DIR, "crop_model_GROUP2.pkl")  # your model file
    le_path = os.path.join(BASE_DIR, "label_encoder.pkl")         # your actual label encoder file

    model = joblib.load(model_path)
    label_encoder = joblib.load(le_path)
except FileNotFoundError as e:
    st.error(f"❌ Model file not found: {e}")
    st.stop()

# Sidebar inputs
st.sidebar.header("Enter Soil and Environmental Parameters")

N = st.sidebar.number_input("Nitrogen (N)")
P = st.sidebar.number_input("Phosphorus (P)")
K = st.sidebar.number_input("Potassium (K)")
temperature = st.sidebar.number_input("Temperature (°C)")
humidity = st.sidebar.number_input("Humidity (%)")
ph = st.sidebar.number_input("pH")
rainfall = st.sidebar.number_input("Rainfall (mm)")

# Predict button
if st.button("Predict Crop"):
    try:
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = model.predict(features)
        crop_name = label_encoder.inverse_transform(prediction)[0]
        st.success(f"✅ Recommended Crop: **{crop_name}**")
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")

# Footer / info
st.markdown("---")
st.markdown("This app is based on **GROUP2 Crop Recommendation Model**.")
