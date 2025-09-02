# app/streamlit_app.py

import streamlit as st
import numpy as np
import json
from joblib import load

# Load model
model = load("models/model.joblib")

# Load features
with open("models/features.json", "r") as f:
    feature_names = json.load(f)

st.title("🍷 Wine Quality Classifier")
st.write("Input wine features and model will predict if wine is good!")

# Form for features
user_input = {}
for feature in feature_names:
    user_input[feature] = st.number_input(
        label=feature.replace("_", " ").capitalize(),
        min_value=0.0,
        step=0.001
        format="%.3f"
    )

# Button "Predict"
if st.button("Predict"):
    # Przygotuj dane wejściowe
    input_data = np.array([list(user_input.values())])
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]  # Prediction "good"

    # Result
    st.subheader("🔍 Prediction Result:")
    st.write("**Good wine** 🍷" if prediction == 1 else "**Not good wine** ❌")
    st.write(f"Confidence: **{proba:.2%}**")