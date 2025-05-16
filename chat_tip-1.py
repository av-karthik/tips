import streamlit as st
import joblib
import numpy as np
import pandas as pd


# Load model and features

#import pickle


model1 = joblib.load('new_model.pkl')
features = joblib.load('new_features.pkl')

st.title("Tip Predictor App")

# Input fields
total_bill = st.number_input("Total Bill", min_value=0.0, step=0.1)

sex = st.selectbox("Sex", ['Male', 'Female'])
smoker = st.selectbox("Smoker", ['Yes', 'No'])
day = st.selectbox("Day", ['Thur', 'Fri', 'Sat', 'Sun'])
time = st.selectbox("Time", ['Lunch', 'Dinner'])
size = st.number_input("Party Size", min_value=1, step=1)

# Create a DataFrame for prediction
input_data = pd.DataFrame({
    'total_bill': [total_bill],
    'size': [size],
    'sex_Male': [1 if sex == 'Male' else 0],
    'smoker_Yes': [1 if smoker == 'Yes' else 0],
    'day_Fri': [1 if day == 'Fri' else 0],
    'day_Sat': [1 if day == 'Sat' else 0],
    'day_Sun': [1 if day == 'Sun' else 0],
    'time_Lunch': [1 if time == 'Lunch' else 0]
})

# Align with model features
input_data = input_data.reindex(columns=features, fill_value=0)

# Prediction
if st.button("Predict Tip"):
    prediction = model1.predict(input_data)[0]
    st.success(f"Predicted Tip Amount: ${prediction:.2f}")