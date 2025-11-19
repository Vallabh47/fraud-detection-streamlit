import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

# ----------------------------------------
# LOAD MODEL & DATA
# ----------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("fraud_detector_final.pkl")

@st.cache_resource
def load_dataset():
    return pd.read_csv("FRAUD DETECTION.csv")

model = load_model()
dataset = load_dataset()

st.set_page_config(page_title="Online Fraud Detection", layout="centered")

st.title("🔐 Online Payment Fraud Detection System")
st.write("Provide transaction details below to check if it is fraudulent.")

# ----------------------------------------
# PREPARE DROPDOWNS FROM DATASET
# ----------------------------------------
card_types = sorted(dataset["card_type"].dropna().unique())
locations = sorted(dataset["location"].dropna().unique())
purchase_categories = sorted(dataset["purchase_category"].dropna().unique())
fraud_types = sorted(dataset["fraud_type"].dropna().unique())

# ----------------------------------------
# INPUT FORM
# ----------------------------------------
amount = st.number_input("Transaction Amount", min_value=1.0, max_value=50000.0, value=500.0)
customer_age = st.number_input("Customer Age", min_value=18, max_value=100, value=30)

card_type = st.selectbox("Card Type", card_types)
location = st.selectbox("Transaction Location", locations)
purchase_category = st.selectbox("Purchase Category", purchase_categories)
fraud_type = st.selectbox("Fraud Type", fraud_types)

# -------------------------
# RESTORED : DATETIME INPUT
# -------------------------
transaction_time = st.text_input(
    "Transaction Date-Time (Format: YYYY-MM-DD HH:MM:SS)",
    "2024-05-01 13:24:00"
)

# ----------------------------------------
# FEATURE ENGINEERING (EXACT LIKE TRAINING)
# ----------------------------------------
try:
    # Convert to datetime
    dt = datetime.strptime(transaction_time, "%Y-%m-%d %H:%M:%S")

    tx_hour = dt.hour
    tx_weekday = dt.weekday()

    is_night = 1 if tx_hour >= 22 or tx_hour <= 5 else 0
    is_weekend = 1 if tx_weekday in [5, 6] else 0

except:
    st.warning("⚠ Please enter date-time in correct format: YYYY-MM-DD HH:MM:SS")
    tx_hour = 0
    tx_weekday = 0
    is_night = 0
    is_weekend = 0

amount_per_age = amount / (customer_age + 1)

location_fraud_rate = dataset.groupby("location")["is_fraudulent"].mean().get(location, dataset["is_fraudulent"].mean())
purchase_category_fraud_rate = dataset.groupby("purchase_category")["is_fraudulent"].mean().get(purchase_category, dataset["is_fraudulent"].mean())

# ----------------------------------------
# PREDICTION
# ----------------------------------------
if st.button("Predict"):
    try:
        input_data = {
            "amount": amount,
            "card_type": card_type,
            "location": location,
            "purchase_category": purchase_category,
            "customer_age": customer_age,
            "fraud_type": fraud_type,
            "tx_hour": tx_hour,
            "tx_weekday": tx_weekday,
            "amount_per_age": amount_per_age,
            "is_night": is_night,
            "is_weekend": is_weekend,
            "location_fraud_rate": location_fraud_rate,
            "purchase_category_fraud_rate": purchase_category_fraud_rate
        }

        input_df = pd.DataFrame([input_data])

        prediction = model.predict(input_df)[0]

        # FINAL CLEAN OUTPUT
        if prediction == 1:
            st.error("🚨 **Fraud Detected** — This transaction shows high fraud characteristics.")
        else:
            st.success("✔ **Legitimate Transaction** — No fraud detected.")

    except Exception as e:
        st.error(f"Error: {e}")
