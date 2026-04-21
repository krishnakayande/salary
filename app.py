import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the trained model
try:
    with open('random_forest_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file 'random_forest_model.pkl' not found. Please ensure it's in the same directory.")
    st.stop()

# Load the LabelEncoders
try:
    with open('label_encoders_for_deployment.pkl', 'rb') as file:
        label_encoders = pickle.load(file)
except FileNotFoundError:
    st.error("LabelEncoder file 'label_encoders_for_deployment.pkl' not found. Please ensure it's in the same directory.")
    st.stop()

# Streamlit UI
st.title("Salary Prediction App")
st.write("Enter the details below to predict the salary.")

# Input fields
age = st.slider("Age", 18, 70, 30)
gender = st.selectbox("Gender", label_encoders['Gender'].classes_)
education_level = st.selectbox("Education Level", label_encoders['Education Level'].classes_)
job_title = st.selectbox("Job Title", label_encoders['Job Title'].classes_)
years_of_experience = st.slider("Years of Experience", 0.0, 30.0, 5.0)

if st.button("Predict Salary"):
    # Create input DataFrame
    input_data = pd.DataFrame([[age, gender, education_level, job_title, years_of_experience]],
                              columns=['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience'])

    # Apply Label Encoding using the loaded encoders
    for col in ['Gender', 'Education Level', 'Job Title']:
        input_data[col] = label_encoders[col].transform(input_data[col])

    # Make prediction
    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Salary: ${prediction:,.2f}")
