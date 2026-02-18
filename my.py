import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# ===============================
# LOAD MODEL
# ===============================
model = pickle.load(open("model_catboost.pkl", "rb"))

# ===============================
# LOAD DATASET (untuk encoder & imputer)
# ===============================
df = pd.read_csv("stroke.csv")

# simpan encoder untuk setiap kolom kategori
encoders = {}
cat_cols = df.select_dtypes(include=['object']).columns

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# imputasi median
num_cols = df.select_dtypes(include=['int64','float64']).columns
imputer = SimpleImputer(strategy="median")
df[num_cols] = imputer.fit_transform(df[num_cols])

# ===============================
# STREAMLIT UI
# ===============================

st.title("Prediksi Risiko Stroke")
st.write("Aplikasi ini memprediksi risiko stroke menggunakan model CatBoost")

st.subheader("Masukkan Data Pasien")

# input user
gender = st.selectbox("Gender", encoders['gender'].classes_)

age = st.number_input("Age", min_value=0, max_value=120, value=30)

hypertension = st.selectbox("Hypertension", [0,1])

heart_disease = st.selectbox("Heart Disease", [0,1])

ever_married = st.selectbox("Ever Married", encoders['ever_married'].classes_)

work_type = st.selectbox("Work Type", encoders['work_type'].classes_)

Residence_type = st.selectbox("Residence Type", encoders['Residence_type'].classes_)

avg_glucose_level = st.number_input("Average Glucose Level", value=100.0)

bmi = st.number_input("BMI", value=25.0)

smoking_status = st.selectbox("Smoking Status", encoders['smoking_status'].classes_)

# ===============================
# ENCODE INPUT
# ===============================

input_dict = {
    "gender": encoders['gender'].transform([gender])[0],
    "age": age,
    "hypertension": hypertension,
    "heart_disease": heart_disease,
    "ever_married": encoders['ever_married'].transform([ever_married])[0],
    "work_type": encoders['work_type'].transform([work_type])[0],
    "Residence_type": encoders['Residence_type'].transform([Residence_type])[0],
    "avg_glucose_level": avg_glucose_level,
    "bmi": bmi,
    "smoking_status": encoders['smoking_status'].transform([smoking_status])[0],
}

input_df = pd.DataFrame([input_dict])

# ===============================
# PREDICT BUTTON
# ===============================

if st.button("Predict"):

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("Pasien Berisiko Stroke")
    else:
        st.success("Pasien Tidak Berisiko Stroke")
