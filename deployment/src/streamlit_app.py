import streamlit as st
import prediction
import eda

navigation = st.sidebar.selectbox("Choose Page", ["Exploratory Data Analysis", "Prediction"])

if navigation == "Exploratory Data Analysis":
    eda.run()
else:
    prediction.run()