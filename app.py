import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("fraud_detector_final.pkl")

model = load_model()

# -----------------------------
# Feature Columns (from your dataset)
# -----------------------------
FEATURES = [
    'amount',
    'card_type',
    'location',
    'purchase_category',
    'customer_age',
    'fraud_type'
]

# -----------------------------
# Preprocessing (same as notebook)
# -----------------------------
def preprocess(df):
    # Categorical columns
    cat_cols = ['card_type', 'location', 'purchase_category', 'fraud_type']
    # Numeric columns
    num_cols = ['amount', 'customer_age']

    # Fill missing categorical with "unknown"
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].fillna("unknown")

    # Fill numeric missing with median
    for c in num_cols:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())

    return df

# -----------------------------
# Prediction Function
# -----------------------------
def predict_single(input_data):
    df = pd.DataFrame([input_data])
    df = preprocess(df)
    pred = model.predict(df)[0]

    # probability (if supported)
    try:
        prob = model.predict_proba(df)[0][1]
    except:
        prob = None

    return pred, prob

# -----------------------------
# Streamlit App UI
# -----------------------------
st.title("🔍 Online Payment Fraud Detection")
st.write("Enter transaction details or upload a CSV to detect fraud.")

mode = st.radio("Select mode:", ["Single Transaction", "Batch Prediction (CSV)"])

# --------------------------------------
# SINGLE INPUT SECTION
# --------------------------------------
if mode == "Single Transaction":
    st.subheader("Enter Transaction Details")

    amount = st.number_input("Transaction Amount", min_value=0.0, step=0.1)

    card_type = st.selectbox("Card Type", 
                             ["Rupay", "MasterCard", "Visa", "unknown"])

    location = st.selectbox("Location", 
                            ["Bangalore", "Hyderabad", "Surat", "unknown"])

    purchase_category = st.selectbox("Purchase Category", 
                                     ["POS", "Digital", "unknown"])

    customer_age = st.number_input("Customer Age", min_value=10, max_value=100, step=1)

    fraud_type = st.selectbox("Fraud Type",
                              ["Identity theft", "Malware", "Payment card fraud", "scam", "unknown"])

    if st.button("Predict"):
        input_data = {
            'amount': amount,
            'card_type': card_type,
            'location': location,
            'purchase_category': purchase_category,
            'customer_age': customer_age,
            'fraud_type': fraud_type
        }

        pred, prob = predict_single(input_data)

        st.write("### Result:")
        st.success("Fraudulent Transaction ❌" if pred == 1 else "Legitimate Transaction ✔")

        if prob is not None:
            st.write(f"**Fraud Probability:** {prob:.4f}")

# --------------------------------------
# BATCH UPLOAD SECTION
# --------------------------------------
else:
    st.subheader("Upload CSV for Batch Prediction")

    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)

        st.write("### Preview of uploaded data:")
        st.dataframe(df.head())

        # Preprocess
        df_proc = preprocess(df.copy())

        # Predict
        try:
            preds = model.predict(df_proc)
            try:
                probs = model.predict_proba(df_proc)[:, 1]
                df['fraud_probability'] = probs
            except:
                probs = None

            df['fraud_prediction'] = preds

            st.write("### Output:")
            st.dataframe(df.head())

            # Download button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results CSV", csv, "fraud_predictions.csv")

        except Exception as e:
            st.error(f"Error while predicting: {e}")
