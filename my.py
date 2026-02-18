import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    return joblib.load("model_catboost.pkl")

model = load_model()

# ===============================
# TITLE
# ===============================
st.title("Prediksi Risiko Stroke Menggunakan Machine Learning")

st.write("Masukkan data pasien untuk memprediksi risiko stroke:")

col1, col2 = st.columns(2)

# ===============================
# INPUT FEATURES
# ===============================
with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female", "Other"]
    )

    age = st.number_input(
        "Age",
        min_value=0,
        max_value=120,
        value=30
    )

    hypertension = st.selectbox(
        "Hypertension",
        [0, 1]
    )

    heart_disease = st.selectbox(
        "Heart Disease",
        [0, 1]
    )

    ever_married = st.selectbox(
        "Ever Married",
        ["Yes", "No"]
    )


with col2:

    work_type = st.selectbox(
        "Work Type",
        ["Private", "Self-employed", "Govt_job", "children", "Never_worked"]
    )

    Residence_type = st.selectbox(
        "Residence Type",
        ["Urban", "Rural"]
    )

    avg_glucose_level = st.number_input(
        "Average Glucose Level",
        min_value=0.0,
        value=100.0
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        value=25.0
    )

    smoking_status = st.selectbox(
        "Smoking Status",
        ["formerly smoked", "never smoked", "smokes", "Unknown"]
    )

# ===============================
# ENCODING (SAMA SEPERTI LABELENCODER)
# ===============================

gender_map = {
    "Female": 0,
    "Male": 1,
    "Other": 2
}

ever_married_map = {
    "No": 0,
    "Yes": 1
}

work_type_map = {
    "Govt_job": 0,
    "Never_worked": 1,
    "Private": 2,
    "Self-employed": 3,
    "children": 4
}

Residence_type_map = {
    "Rural": 0,
    "Urban": 1
}

smoking_status_map = {
    "Unknown": 0,
    "formerly smoked": 1,
    "never smoked": 2,
    "smokes": 3
}

# encode
gender_encoded = gender_map[gender]
ever_married_encoded = ever_married_map[ever_married]
work_type_encoded = work_type_map[work_type]
Residence_type_encoded = Residence_type_map[Residence_type]
smoking_status_encoded = smoking_status_map[smoking_status]

# ===============================
# PREDICT BUTTON
# ===============================
if st.button("Prediksi Stroke"):

    features = pd.DataFrame([{
        "gender": gender_encoded,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": ever_married_encoded,
        "work_type": work_type_encoded,
        "Residence_type": Residence_type_encoded,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": smoking_status_encoded
    }])

    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("Pasien Berisiko Stroke")
    else:
        st.success("Pasien Tidak Berisiko Stroke")
