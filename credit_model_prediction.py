import joblib

# ✅ Load model (using joblib)
model = joblib.load("credit_model.pkl")

# Example: one test input [log_income, log_loan_amount, term_binary, credit_history]
sample_data = [[0.55, 0.33, 1, 1.0]]

# Predict
prediction = model.predict(sample_data)

if prediction[0] == 1:
    print("💰 Loan will be DEFAULTED (High Risk)")
else:
    print("✅ Loan will be REPAID (Low Risk)")
