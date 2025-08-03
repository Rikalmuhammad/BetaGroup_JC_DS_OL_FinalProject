import streamlit as st
import pandas as pd
import pickle

st.title("Bank Marketing Term Deposit Predictor")

with st.sidebar:
    age = st.slider("Age", 17, 98, 30)
    job = st.selectbox("Job", ["admin.", "blue-collar", ...])
    marital = st.selectbox("Marital Status", ["married", "single", ...])
    education = st.selectbox("Education", ["primary", "secondary", ...])
    default = st.selectbox("Has credit default?", ["yes", "no", "unknown"])
    housing = st.selectbox("Housing loan?", ["yes", "no"])
    loan = st.selectbox("Personal loan?", ["yes", "no"])
    contact = st.selectbox("Contact type", ["cellular", "telephone"])
    month = st.selectbox("Month", ["jan", "feb", ...])
    day_of_week = st.selectbox("Day of Week", ["mon", "tue", ...])
    duration = st.number_input("Last contact duration (s)", min_value=0)
    campaign = st.number_input("Campaign contacts", min_value=1)
    pdays = st.number_input("Days since previous contact", value=999)
    previous = st.number_input("Previous contacts", min_value=0)
    poutcome = st.selectbox("Previous campaign outcome", ["unknown", "success", ...])
    emp_var_rate = st.number_input("Employment variation rate")
    cons_price_idx = st.number_input("Consumer price index")
    cons_conf_idx = st.number_input("Consumer confidence index")
    euribor3m = st.number_input("Euribor 3m rate")
    nr_employed = st.number_input("Number employed")

input_df = pd.DataFrame({
    'age': age,
    'job': job,
    'marital': marital,
    'education': education,
    'default': default,
    'housing': housing,
    'loan': loan,
    'contact': contact,
    'month': month,
    'day_of_week': day_of_week,
    'duration': duration,
    'campaign': campaign,
    'pdays': pdays,
    'previous': previous,
    'poutcome': poutcome,
    'emp.var.rate': emp_var_rate,
    'cons.price.idx': cons_price_idx,
    'cons.conf.idx': cons_conf_idx,
    'euribor3m': euribor3m,
    'nr.employed': nr_employed
}, index=[0])

st.write("### Input Data")
st.dataframe(input_df)

# Load model
with open('model.sav', 'rb') as f:
    model = pickle.load(f)

pred = model.predict(input_df)[0]
prob = model.predict_proba(input_df)[0][1]

st.write("### Prediction")
st.write(f"Prediction class: **{'Yes (will subscribe)' if pred==1 else 'No (unlikely)'}**")
st.write(f"Probability of success: **{prob:.2%}**")
