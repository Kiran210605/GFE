import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the saved model
model = joblib.load('best_model.pkl')

# Function to make predictions
def predict_order(input_data):
    # Make predictions using the trained model
    forecast = model.predict(input_data)
    return forecast

# Title of the app
st.title("Inventory Demand Prediction")

# User input
st.header("Enter the Date and Product Information")

date_input = st.date_input("Select the Date", value=pd.to_datetime('2025-02-17'))
product = st.selectbox("Select Product", ['Tenderstem', 'babycorn', 'finebeans'])

# Simulate features (use actual feature extraction in production)
# Here we will just simulate lag values for the sake of example
lag_1 = st.number_input(f"Enter lag for {product} (1 day)", value=100, min_value=0)
lag_7 = st.number_input(f"Enter lag for {product} (7 days)", value=120, min_value=0)
rolling_mean_7 = st.number_input(f"Enter rolling mean for {product} (7 days)", value=110, min_value=0)

# Prepare the input data in the same format the model expects
input_data = np.array([[lag_1, lag_7, rolling_mean_7]])

# Predict and display the result
if st.button("Predict"):
    prediction = predict_order(input_data)
    st.write(f"The predicted demand for {product} on {date_input} is: {prediction[0]:.2f} units")
