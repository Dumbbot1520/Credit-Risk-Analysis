import streamlit as st
import pickle
import pandas as pd

# Load model and scaler
with open('credit_risk_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Page config
st.set_page_config(page_title="Credit Risk Predictor", page_icon="🏦")

st.title("🏦 Credit Risk Predictor")
st.markdown("Predict whether a customer is likely to default on their credit card payment.")
st.divider()

# Input form
col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal Information")
    limit_bal = st.number_input("Credit Limit (₹)", min_value=0, value=50000, step=10000)
    age = st.number_input("Age", min_value=18, max_value=100, value=28)
    sex = st.selectbox("Sex", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
    education = st.selectbox("Education", options=[1, 2, 3, 4],
                             format_func=lambda x: {1: "Graduate School", 2: "University",
                                                     3: "High School", 4: "Others"}[x])
    marriage = st.selectbox("Marriage Status", options=[1, 2, 3],
                            format_func=lambda x: {1: "Married", 2: "Single", 3: "Others"}[x])

with col2:
    st.subheader("Payment History (Last 6 Months)")
    st.caption("-1 = On time, 0 = Minimum paid, 1+ = Months late")
    pay_0 = st.slider("PAY_0 (Most Recent)", min_value=-1, max_value=8, value=0)
    pay_2 = st.slider("PAY_2", min_value=-1, max_value=8, value=0)
    pay_3 = st.slider("PAY_3", min_value=-1, max_value=8, value=0)
    pay_4 = st.slider("PAY_4", min_value=-1, max_value=8, value=0)
    pay_5 = st.slider("PAY_5", min_value=-1, max_value=8, value=0)
    pay_6 = st.slider("PAY_6", min_value=-1, max_value=8, value=0)

st.divider()
st.subheader("Payment Ratios (Amount Paid ÷ Bill Amount)")
st.caption("0 = paid nothing, 1 = paid full bill")

col3, col4, col5 = st.columns(3)
with col3:
    pay_ratio1 = st.number_input("PAY_RATIO1", 0.0, 1.0, 0.5, step=0.01)
    pay_ratio2 = st.number_input("PAY_RATIO2", 0.0, 1.0, 0.5, step=0.01)
with col4:
    pay_ratio3 = st.number_input("PAY_RATIO3", 0.0, 1.0, 0.5, step=0.01)
    pay_ratio4 = st.number_input("PAY_RATIO4", 0.0, 1.0, 0.5, step=0.01)
with col5:
    pay_ratio5 = st.number_input("PAY_RATIO5", 0.0, 1.0, 0.5, step=0.01)
    pay_ratio6 = st.number_input("PAY_RATIO6", 0.0, 1.0, 0.5, step=0.01)

avg_pay_ratio = (pay_ratio1 + pay_ratio2 + pay_ratio3 +
                 pay_ratio4 + pay_ratio5 + pay_ratio6) / 6

total_delay = pay_0 + pay_2 + pay_3 + pay_4 + pay_5 + pay_6

st.divider()
col6, col7 = st.columns(2)
with col6:
    st.metric("Auto-calculated AVG_PAY_RATIO", f"{avg_pay_ratio:.2f}")
with col7:
    st.metric("Auto-calculated TOTAL_DELAY", total_delay)

st.divider()

# Predict button
if st.button("🔍 Predict Default Risk", use_container_width=True):
    input_data = pd.DataFrame([{
        'LIMIT_BAL': limit_bal, 'SEX': sex, 'EDUCATION': education,
        'MARRIAGE': marriage, 'AGE': age, 'PAY_0': pay_0, 'PAY_2': pay_2,
        'PAY_3': pay_3, 'PAY_4': pay_4, 'PAY_5': pay_5, 'PAY_6': pay_6,
        'PAY_RATIO1': pay_ratio1, 'PAY_RATIO2': pay_ratio2,
        'PAY_RATIO3': pay_ratio3, 'PAY_RATIO4': pay_ratio4,
        'PAY_RATIO5': pay_ratio5, 'PAY_RATIO6': pay_ratio6,
        'AVG_PAY_RATIO': avg_pay_ratio, 'TOTAL_DELAY': total_delay
    }])

    input_scaled = scaler.transform(input_data)
    probability = model.predict_proba(input_scaled)[:, 1][0]
    prediction = 1 if probability >= 0.3 else 0

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ HIGH RISK — Likely Defaulter")
    else:
        st.success(f"✅ LOW RISK — Likely Repayer")

    col8, col9 = st.columns(2)
    with col8:
        st.metric("Default Probability", f"{probability:.2%}")
    with col9:
        st.metric("Decision Threshold", "0.30")

    st.caption("Model: Logistic Regression | Optimised for Recall (catching defaulters)")