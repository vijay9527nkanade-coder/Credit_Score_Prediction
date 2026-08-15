import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Dataset (loan_data.csv)
df = pd.read_csv("loan_data.csv")

# Feature engineering
df["term_binary"] = df["term"].apply(lambda x: 1 if x == 60 else 0)
df["log_income"] = np.log1p(df["income"])
df["log_loan_amount"] = np.log1p(df["loan_amount"])

features = ["log_income", "log_loan_amount", "term_binary", "credit_history"]
target = "defaulted"

X = df[features]
y = df[target]

# Scale features
scaler = StandardScaler()
X[["log_income", "log_loan_amount"]] = scaler.fit_transform(X[["log_income", "log_loan_amount"]])

# Train model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X, y)

# Save model + scaler
joblib.dump({"model": model, "scaler": scaler}, "credit_artifacts.pkl")
print("✅ Model and Scaler saved successfully as credit_artifacts.pkl")
