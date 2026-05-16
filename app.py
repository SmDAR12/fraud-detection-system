import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load("fraud_model.pkl")

# Title
st.title("💳 Fraud Detection Dashboard")

st.write("Real-Time Fraud Transaction Monitoring")

# Sample transaction counts
normal_count = 95
fraud_count = 5

# Metrics
st.subheader("📊 Transaction Summary")

col1, col2 = st.columns(2)

col1.metric("Normal Transactions", normal_count)
col2.metric("Fraud Transactions", fraud_count)

# Pie Chart
labels = ["Normal", "Fraud"]
sizes = [normal_count, fraud_count]

fig, ax = plt.subplots()

ax.pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%'
)

ax.set_title("Transaction Distribution")

st.pyplot(fig)

# Testing Section
st.subheader("🧪 Test Transactions")

# Sample normal transaction
normal_data = [1000] + [0]*28 + [50]

# Sample fraud transaction
fraud_data = [5000] + [-10]*28 + [5000]

# Normal transaction button
if st.button("Test Normal Transaction"):

    input_data = np.array([normal_data])

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)[0][1]

    st.write(f"Fraud Probability: {probability:.2f}")

    if prediction[0] == 1:
        st.error("⚠️ Fraud Detected")
    else:
        st.success("✅ Normal Transaction")

# Fraud transaction button
if st.button("Test Fraud Transaction"):

    input_data = np.array([fraud_data])

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)[0][1]

    st.write(f"Fraud Probability: {probability:.2f}")

    if prediction[0] == 1:
        st.error("⚠️ Fraud Detected")
    else:
        st.success("✅ Normal Transaction")