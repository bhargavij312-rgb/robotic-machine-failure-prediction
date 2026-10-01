import streamlit as st
import pandas as pd
import joblib
from features import add_features

st.set_page_config(page_title="Machine Failure Prediction - Pooja Sharma")
st.title("Robotic Machine Failure Prediction - Pooja Sharma")

model = joblib.load('models/failure_model.joblib')

with st.form("predict"):
    air = st.number_input("Air temperature [K]", 298.0)
    process = st.number_input("Process temperature [K]", 308.0)
    rpm = st.number_input("Rotational speed [rpm]", 1500)
    torque = st.number_input("Torque [Nm]", 40.0)
    wear = st.number_input("Tool wear [min]", 100)
    type_ = st.selectbox("Type", ["L", "M", "H"])
    submit = st.form_submit_button("Predict")

if submit:
    df = pd.DataFrame([{'Air temperature [K]': air, 'Process temperature [K]': process, 'Rotational speed [rpm]': rpm, 'Torque [Nm]': torque, 'Tool wear [min]': wear, 'Type': type_}])
    df = add_features(df)
    prob = model.predict_proba(df)[0][1]
    pred = model.predict(df)[0]
    st.metric("Failure Probability", f"{prob:.2%}")
    if pred == 1:
        st.error("FAILURE likely")
    else:
        st.success("No Failure")
