import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ================================
# Load model
# ================================
@st.cache_resource
def load_model():
    return joblib.load("fraud_detector_final.pkl")

model = load_model()

# ================================
# Load dataset for dropdown values
# ================================
@st.cache_resource
def load_data():
    df = pd.read_csv("FRAUD DETECTION.csv")
    return df

df_raw = load_data()

# Unique dropdown values from dataset
LOCATIONS = sorted(df_raw['location'].dropna().unique().tolist())
CARD_TYPES = sorted(df_raw['card_type'].dropna().unique().tolist())
PURCHASE_CATEGORIES = sorted(df_raw['purchase_category'].dropna().unique().tolist())
FRAUD_TYPES = sorted(df_raw['fraud_type'].dropna().unique().tolist())

# ================================
# Features required by your model
# ================================
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

# ================================
# Feature Engineering
# ================================
def engineer_features(df):
    df["transaction_time"] = pd.to_datetime(df["transaction_time"])

    df["tx_hour"] = df["transaction_time"].dt.hour
    df["tx_weekday"] = df["transaction_time"].dt.weekday

    df["amount_per_age"] = df["amount"] / (df["customer_age"] + 1)

    df["is_night"] = df["tx_hour"].apply(lambda h: 1 if (h >= 22 or h <= 5) else 0)
    df["is_weekend"] = df["tx_weekday"].apply(lambda d: 1 if d >= 5 else 0)

    # Default risk scores (your notebook created these)
    df["location_fraud_rate"] = 0.05
    df["purchase_category_fraud_rate"] = 0.05

    return df

# ================================
# Prediction
# ================================
def predict_single(data):
    df = pd.DataFrame([data])
    df = engineer_features(df)
    df = df[FEATURES]

    pred = model.predict(df)[0]

    try:
        prob = model.predict_proba(df)[0][1]
    except:
        prob = None

    return pred, prob

# ================================
# Streamlit UI
# ================================
st.title("🔍 Online Payment Fraud Detection")

st.subheader("Enter transaction details")

amount = st.number_input("Transaction Amount", min_value=0.0)

card_type = st.selectbox("Card Type", CARD_TYPES)

location = st.selectbox("Location", LOCATIONS)

purchase_category = st.selectbox("Purchase Category", PURCHASE_CATEGORIES)

customer_age = st.number_input("Customer Age", min_value=10, max_value=100)

fraud_type = st.selectbox("Fraud Type", FRAUD_TYPES)

transaction_time = st.text_input(
    "Transaction Date-Time (YYYY-MM-DD HH:MM)", 
    value="2024-01-01 12:30"
)

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

    st.subheader("Prediction Result:")
    if pred == 1:
        st.error("⚠ Fraudulent Transaction Detected")
    else:
        st.success("✔ Legitimate Transaction")

    if prob is not None:
        st.write(f"Fraud Probability: {prob:.4f}")
