import streamlit as st
import pandas as pd
import pickle
import os

# Load model
model_path = os.path.join("artifacts", "model.pkl")
if not os.path.exists(model_path):
    st.error("Model file not found. Please train the model first.")
    st.stop()

with open(model_path, "rb") as f:
    model = pickle.load(f)

st.title("✈️ Flight Delay Prediction App (Simplified)")

# User input for the same features used in training
def user_input():
    ArrDelayMinutes = st.number_input("Arrival Delay Minutes", value=5)
    Cancelled = st.selectbox("Cancelled", [0, 1])
    DayOfWeek = st.selectbox("Day of Week", [1, 2, 3, 4, 5, 6, 7])
    Distance = st.number_input("Distance", value=500)
    AirTime = st.number_input("AirTime", value=90)
    CRSDepTime = st.number_input("Scheduled Departure Time", value=1030)
    DepTime = st.number_input("Actual Departure Time", value=1045)
    DepDelay = st.number_input("Departure Delay", value=15)
    Diverted = st.selectbox("Diverted", [0, 1])

    data = {
        "ArrDelayMinutes": ArrDelayMinutes,
        "Cancelled": Cancelled,
        "DayOfWeek": DayOfWeek,
        "Distance": Distance,
        "AirTime": AirTime,
        "CRSDepTime": CRSDepTime,
        "DepTime": DepTime,
        "DepDelay": DepDelay,
        "Diverted": Diverted
    }

    return pd.DataFrame([data])

input_df = user_input()

if st.button("Predict Delay"):
    try:
        prediction = model.predict(input_df)
        if prediction[0] == 1:
            st.error("⚠️ Flight is likely to be delayed!")
        else:
            st.success("✅ Flight is likely to be on time.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
