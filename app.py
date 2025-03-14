import streamlit as st
# import numpy as np
import pandas as pd
import joblib
import shap

# from sklearn.preprocessing import StandardScaler
# from xgboost import XGBClassifier

# import os
# print(os.path.exists("best_model.pkl"))

try:
  model = joblib.load("best_model.pkl")

except Exception as e:
    print("Failed to load model:", e)

st.set_page_config(page_title="Heart Disease Risk Estimator", layout="centered")
st.title("Heart Disease Risk Estimator")

st.markdown(
    """
    [![GitHub](https://img.shields.io/badge/GitHub-View%20Source-blue?logo=github)](https://github.com/SarahMesroua/heart-disease-risk-app)  
    View this project on GitHub
    
    ---
    This interactive machine learning app predicts the risk of heart disease based on clinical features like age, cholesterol, chest pain type, and more.  
    
    Multiple machine learning models were trained and evaluated, including SVM, Random Forest, and XGBoost.  
    After cross-validation and performance comparison, a **Logistic Regression model** was selected as the best performer.
    
    Enter health parameters below to estimate heart disease risk
    
    ---
    """,
    unsafe_allow_html=True
)

def user_inputs():
    age = st.number_input("Age", 20, 100, 50)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female (0)" if x == 0 else "Male (1)")
    cp = st.selectbox("Chest Pain Type", [1, 2, 3, 4], 
                format_func=lambda x: "typical angina (1, classic heart-related chest pain)" if x == 1 else "atypical angina (2, chest pain not clearly related to the heart)" if x == 2 else "non-anginal pain (3, chest pain not due to heart)" if x == 3 else "asymptomatic (4, no chest pain)")
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.slider("Cholesterol (mg/dl)", 100, 600, 240)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "False (0)" if x == 0 else "True (1)")
    restecg = st.selectbox("Resting ECG Result", [0, 1], format_func=lambda x: "Normal (0)" if x == 0 else "Abnormal (1)")
    thalach = st.slider("Max Heart Rate Achieved (peak heart rate during exercise)", 60, 210, 150)
    exang = st.selectbox("Exercise Induced Angina (chest pain during exercise)", [0, 1], format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")
    oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0, step=0.1)
    slope = st.selectbox("Slope of ST segment (shape of ST segment during exercise)", [1, 2, 3],
                format_func=lambda x: "upsloping (1)" if x == 1 else "flat (2)" if x == 2 else "downsloping (3)")
    ca = st.selectbox("Number of Major Vessels (count of major vessels (0–3) colored during fluoroscopy)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia (blood disorder screening during imaging)", [3, 6, 7],
                format_func=lambda x: "normal (=3)" if x == 3 else "fixed defect (=6)" if x == 6 else "reversable defect (=7)")

    return {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }

input_data = user_inputs()

# compute additional features
df_input = pd.DataFrame([input_data])
df_input['age_chol'] = df_input['age'] * df_input['chol']
df_input['bp_chol_ratio'] = df_input['trestbps'] / (df_input['chol'] + 1e-5)
df_input['chol_ratio'] = df_input['chol'] / 50

# prediction
if st.button("Predict Heart Disease Risk"):
    feature_order = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else df_input.columns
    X = df_input[feature_order]

    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]

    # display pred with sytling
    st.subheader("Prediction Result")

    risk_label = ""
    risk_style = ""

    if prediction == 1:
        if proba >= 0.8:
            risk_label = "High Risk (Confidence >= 80%)"
            risk_style = "error"
        elif proba >= 0.5:
            risk_label = "Moderate Risk (50% < Confidence < 80%)"
            risk_style = "warning"
        else:
            risk_label = "No to Low Risk (Confidence < 50%)"
            risk_style = "success"
    else:
        risk_label = "No to Low Risk (Confidence < 50%)"
        risk_style = "success"

    st.markdown(f"**Prediction:** {'At Risk (Yes)' if prediction == 1 else 'No Risk (No)'}")
    st.markdown(f"**Model Confidence:** `{proba * 100:.2f}%` likelihood of heart disease")

    getattr(st, risk_style)(f"**Risk Level: {risk_label}**")