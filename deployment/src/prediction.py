import pandas as pd
import numpy as np

import pickle
import streamlit as st

# load model
with open('best_model_dtc.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

def run():
    # header
    st.title("Purchase Prediction for E-Commerce Website Visitors")
    st.markdown('---')
    st.markdown("Enter visitor data to predict the likelihood of purchase")

    # Form input
    with st.form("input_form"):
        administrative = st.number_input("Number of Administrative Pages", 0, 100, 3)
        administrative__duration = st.number_input("Duration of Administrative Pages (seconds)", 0.0, 1000.0, 120.0)
        informational = st.number_input("Number of Informational Pages", 0, 100, 0)
        informational__duration = st.number_input("Duration of Informational Pages (seconds)", 0.0, 1000.0, 0.0)
        product_related = st.number_input("Number of Product Pages", 0, 100, 30)
        product_related__duration = st.number_input("Duration of Product Pages (seconds)", 0.0, 5000.0, 1200.0)
        bounce_rates = st.slider("Bounce Rates", 0.0, 1.0, 0.02)
        exit_rates = st.slider("Exit Rates", 0.0, 1.0, 0.05)
        page_values = st.number_input("Page Value", 0.0, 100.0, 5.4)
        special_day = st.selectbox("Special Day?", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        month = st.selectbox("Month of Visit", ['Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        operating_systems = st.selectbox("Operating System", [1, 2, 3, 4, 5, 6, 7, 8])
        browser = st.selectbox("Browser", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
        region = st.selectbox("Region", list(range(1, 10)))
        traffic_type = st.selectbox("Traffic Type", list(range(1, 21)))
        visitor_type = st.selectbox("Visitor Type", ['Returning_Visitor', 'New_Visitor', 'Other'])
        weekend = st.radio("Weekend?", [True, False])

        submitted = st.form_submit_button("Predict")

    if submitted:
        # Format input jadi dataframe
        user_data = pd.DataFrame([{
            'administrative': administrative,
            'administrative__duration': administrative__duration,
            'informational': informational,
            'informational__duration': informational__duration,
            'product_related': product_related,
            'product_related__duration': product_related__duration,
            'bounce_rates': bounce_rates,
            'exit_rates': exit_rates,
            'page_values': page_values,
            'special_day': special_day,
            'month': month,
            'operating_systems': operating_systems,
            'browser': browser,
            'region': region,
            'traffic_type': traffic_type,
            'visitor_type': visitor_type,
            'weekend': weekend
        }])
        
        # Prediction
        prediction = loaded_model.predict(user_data)[0]
        probability = loaded_model.predict_proba(user_data)[0][1]

        st.success(f"Prediction: {'Will Buy' if prediction else 'Will Not Buy'}")
        st.info(f"Purchase Probability: {probability:.2%}")

if __name__ == "__main__":
    run()