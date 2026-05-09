import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('heart_model.pkl')

st.set_page_config(
    page_title='Heart Disease Detection',
    page_icon='❤️',
    layout='centered'
)


st.title('❤️ Heart Disease Detection App')
st.write('Enter patient details to predict heart disease.')


Age = st.number_input('Age', min_value=1, max_value=120, value=30)

Gender = st.selectbox(
    'Gender',
    ['Male', 'Female']
)

ChestPainType = st.selectbox(
    'Chest Pain Type',
    ['ATA', 'NAP', 'ASY', 'TA']
)

RestingBP = st.number_input('Resting Blood Pressure', value=120)

Cholesterol = st.number_input('Cholesterol', value=200)

FastingBS = st.selectbox(
    'Fasting Blood Sugar',
    [0, 1]
)

RestingECG = st.selectbox(
    'Resting ECG',
    ['Normal', 'ST', 'LVH']
)

MaxHR = st.number_input('Maximum Heart Rate', value=150)

ExerciseAngina = st.selectbox(
    'Exercise Angina',
    ['N', 'Y']
)

Oldpeak = st.number_input('Oldpeak', value=1.0)

ST_Slope = st.selectbox(
    'ST Slope',
    ['Up', 'Flat', 'Down']
)


# gender encoding
if Gender == 'Male':
    gender = 1
else:
    gender = 0

# ChestPainType encoding
cp_map = {
    'ASY': 0,
    'ATA': 1,
    'NAP': 2,
    'TA': 3
}
ChestPainType = cp_map[ChestPainType]

# RestingECG encoding
restecg_map = {
    'LVH': 0,
    'Normal': 1,
    'ST': 2
}
RestingECG = restecg_map[RestingECG]

# ExerciseAngina encoding
exang_map = {
    'N': 0,
    'Y': 1
}
ExerciseAngina = exang_map[ExerciseAngina]

# ST_Slope encoding
slope_map = {
    'Down': 0,
    'Flat': 1,
    'Up': 2
}
ST_Slope = slope_map[ST_Slope]


# Prediction

if st.button('Predict'):

    input_data = np.array([[ 
        Age,
        ChestPainType,
        RestingBP,
        Cholesterol,
        FastingBS,
        RestingECG,
        MaxHR,
        ExerciseAngina,
        Oldpeak,
        ST_Slope,
        gender
    ]])

    prediction = model.predict(input_data)

    st.subheader('Prediction Result')

    if prediction[0] == 1:
        st.error('⚠️ Heart Disease Detected')
    else:
        st.success('✅ No Heart Disease Detected')


st.markdown('---')
st.caption('Machine Learning Heart Disease Detection Project')
