import pandas as pd
import requests
import streamlit as st
import os

st.title("HR Promotion Predictor")

df = pd.read_csv("Data/train.csv")

API_URL = os.getenv("API_URL", "http://localhost:8080/predict")

employee_id = st.number_input("Employee ID")
department = st.selectbox("Department", df['department'].unique())
region = st.selectbox("Region", df['region'].unique())
education = st.selectbox("Education Level", df['education'].unique())
gender = st.selectbox("Gender", df['gender'].unique())
recruitment_channel = st.selectbox("Recruitment Channel", df['recruitment_channel'].unique())
no_of_trainings = st.number_input("Number of Trainings Attended")
age = st.number_input("Age")
previous_year_rating = st.number_input("Previous Year Rating", min_value=1, max_value=5)
length_of_service = st.number_input("Length of Service (Years)")
KPIs_met = st.number_input("Number of KPIs Met", min_value=0, max_value=1)
awards_won = st.number_input("Number of Awards Won", min_value=0, max_value=1)
avg_training_score = st.number_input("Average Training Score") 


if st.button("Predict Promotion"):
    payload = {
        'employee_id': int(employee_id),
        'department': department,
        'region': region,
        'education': education,
        'gender': gender,
        'recruitment_channel': recruitment_channel,
        'no_of_trainings': int(no_of_trainings), 
        'age': int(age),
        'previous_year_rating': float(previous_year_rating),
        'length_of_service': int(length_of_service),
        'KPIs_met_80': int(KPIs_met),
        'awards_won': int(awards_won),
        'avg_training_score': avg_training_score
    }

    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            result = response.json()
            st.success(f"Result: {result['promotion_prediction_class']}")
            st.info(f"Promotion Probability: {result['probability']:.2f}")
            if result['promotion_prediction_class'] == 'Promoted':
                st.balloons()
            else:
                st.snow()
        else:
            st.error("Error in prediction API")

    except Exception as e:
        st.error(f"An error occurred: {e}")
