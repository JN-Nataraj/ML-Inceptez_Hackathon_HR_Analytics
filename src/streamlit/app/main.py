import pandas as pd
import joblib
import streamlit as st
from featureengineering import FeatureEngineering

st.title("HR Promotion Predictor")

df = pd.read_csv("Data/train.csv")

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

inputs = {
    'employee_id': employee_id,
    'department': department,
    'region': region,
    'education': education,
    'gender': gender,
    'recruitment_channel': recruitment_channel,
    'no_of_trainings': no_of_trainings, 
    'age': age,
    'previous_year_rating': previous_year_rating,
    'length_of_service': length_of_service,
    'KPIs_met >80%': KPIs_met,
    'awards_won?': awards_won,
    'avg_training_score': avg_training_score
}

if st.button("Predict Promotion"):
    model = joblib.load("Models/hr_analytics_thrsh77_model.pkl")
    input_df = pd.DataFrame([inputs])
    prediction = model.predict(input_df)
    st.write(prediction)