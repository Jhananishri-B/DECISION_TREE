import streamlit as st
import pickle
import numpy as np

with open("house_price_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🏡 House Price Category Prediction")


size = st.number_input("Enter Size (m²)", min_value=10.0, max_value=1000.0, value=120.0)
bedrooms = st.number_input("Enter Number of Bedrooms", min_value=1, max_value=10, value=3)
location_score = st.number_input("Enter Location Score (1-10)", min_value=1.0, max_value=10.0, value=7.5)


if st.button("Predict Price Category"):
    input_data = np.array([[size, bedrooms, location_score]])
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Price Category: {prediction}")
