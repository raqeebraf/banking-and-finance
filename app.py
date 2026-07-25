import streamlit as st
import joblib
import pandas as pd

rf_model = joblib.load("random_model.pkl")
state_encoder = joblib.load("state_encoder.pkl")

st.set_page_config(
    page_title="Profit Prediction",
    page_icon="📈"
)

st.title("📈 Business Profit Prediction")

st.write("Predict Company Profit using Machine Learning (Random Forest Regression)")

st.divider()

rd = st.number_input(
    "R&D Spend",
    min_value=0.0
)

admin = st.number_input(
    "Administration Cost",
    min_value=0.0
)

marketing = st.number_input(
    "Marketing Spend",
    min_value=0.0
)

state = st.selectbox(
    "State",
    state_encoder.classes_
)

st.divider()

if st.button("Predict Profit"):

    input_data = pd.DataFrame({

        "R&D Spend":[rd],

        "Administration":[admin],

        "Marketing Spend":[marketing],

        "State":[state_encoder.transform([state])[0]]

    })

    prediction = rf_model.predict(input_data)

    st.success(f"Predicted Profit : ${prediction[0]:,.2f}")