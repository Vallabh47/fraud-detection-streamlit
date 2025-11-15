import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("fraud_detector_final.pkl")

model = load_model()

# -----------------------------
# REQUIRED FEATURE COLUMNS
# -----------------------------
FEATURES = [
    'amount',
    'card_type',
    'location',
    'purchase_category',
    'customer_age',
    'fraud_type',
    'tx_hour',
    'tx_weekday',
    'amount_per_age',
    'is_night',
    'is_weekend',
    'location_fraud_rate',
    'purchase_category_fraud_rate'
]

# -----------------------------
# Feature Engineering Function
# -----------------------------
def engineer_features(df):
    """Recreate the engineered features from your notebook"""
    
    # Convert transaction_time to datetime
    df["transaction_time"] = pd.to_datetime(df["transaction_time"])

    df["tx_hour"] = df["transaction_time"].dt.hour
    df["tx_weekday"] = df["transaction_time"].dt.weekday

    # Feature: amount_per_age
    df["amount_per_age"] = df["amount"] / (df["customer_age"] + 1)

    # Feature: is_night (10 PM to 5 AM)
    df["is_night"] = df["tx_hour"].apply(lambda h: 1 if (h >= 22 or h <= 5) else 0)

    # Feature: is_weekend
    df["is_weekend"] = df["tx_weekday"].apply(lambda d: 1 if d >= 5 else 0)

    # Risk score placeholders (your notebook created these)
    df["location_fraud_rate"] = 0.05   # replace with actual rate if available
    df["purchase_category_fraud_rate"] = 0.05  # replace if needed

    return df

# -----------------------------
# Prediction
# -----------------------------
def predict_single(input_data):
    df = pd.DataFrame([input_data])

    df = engineer_features(df)

    # Keep only required columns
    df = df[FEATURES]

    pred = model.predict(df)[0]
    prob = None

    try:
        prob = model.predict_proba(df)[0][1]
    except:
        pass

    return pred, prob

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🔍 Online Fraud Detection System")
st.write("Enter transaction details below:")

amount = st.number_input("Transaction Amount", min_value=0.0)

card_type = st.selectbox("Card Type", ["Rupay", "MasterCard", "Visa", "Unknown"])

location = st.selectbox("Location", ["Bangalore", "Hyderabad", "Surat", "Unknown"])

purchase_category = st.selectbox("Purchase Category", ["POS", "Digital", "Unknown"])

customer_age = st.number_input("Customer Age", min_value=10, max_value=100)

fraud_type = st.selectbox("Fraud Type", ["Identity theft", "Malware", "Payment card fraud", "scam"])

transaction_time = st.text_input("Transaction Date-Time (YYYY-MM-DD HH:MM)",
                                 value="2024-01-01 12:30")

if st.button("Predict Fraud"):
    input_data = {
        "amount": amount,
        "card_type": card_type,
        "location": location,
        "purchase_category": purchase_category,
        "customer_age": customer_age,
        "fraud_type": fraud_type,
        "transaction_time": transaction_time
    }

    pred, prob = predict_single(input_data)

    st.subheader("Result:")
    if pred == 1:
        st.error("⚠ Fraudulent Transaction")
    else:
        st.success("✔ Legitimate Transaction")

    if prob is not None:
        st.write(f"Fraud Probability: {prob:.4f}")
