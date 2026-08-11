import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Credit Scoring App", layout="centered")

st.title("💳 Credit Score Prediction")
st.write("Enter applicant details to check loan repayment risk.")

# Load trained model & scaler
artifacts = joblib.load("credit_artifacts.pkl")
model = artifacts["model"]
scaler = artifacts["scaler"]

# User Inputs
income = st.number_input("Monthly Income (₹):", min_value=0.0, value=30000.0, step=1000.0)
loan_amount = st.number_input("Loan Amount (₹):", min_value=0.0, value=15000.0, step=500.0)
term = st.selectbox("Loan Term (months):", (36, 60))
credit_history = st.selectbox("Credit History (1 = Good, 0 = Bad):", (1, 0))

# Predict Button
if st.button("Predict Loan Default"):
    term_binary = 1 if term == 60 else 0
    log_income = np.log1p(income)
    log_loan_amount = np.log1p(loan_amount)

    scaled = scaler.transform([[log_income, log_loan_amount]])
    features = np.hstack([scaled, [[term_binary, credit_history]]])

    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    st.subheader("🔍 Prediction Result")
    if pred == 1:
        st.error(f"⚠️ High Risk of Default (Probability = {prob:.2f})")
    else:
        st.success(f"✅ Low Risk of Default (Probability = {prob:.2f})")
