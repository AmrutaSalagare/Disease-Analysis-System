import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

# Load the trained model
model_path = 'C:/Users/heman/OneDrive/Desktop/Disease Prediction/saved_models/Covid19_model.sav'
classifier = pickle.load(open(model_path, 'rb'))

# Load and prepare the dataset for standardization
dia = pd.read_csv(
    'C:/Users/heman/OneDrive/Desktop/Disease Prediction/dataset/covid19-dataset.csv')
dia = dia.fillna(0)

# Fit the scaler on the dataset (without the target variable)
scaler = StandardScaler()
scaler.fit(X)

with st.sidebar:
    selected = option_menu("Multiple Disease Prediction System",

                           ["Covid-19 Prediction",
                            "Breast Cancer Prediction"],
                           icons=['virus', 'bullseye'],
                           default_index=0)

# Covid-19 Prediction Page
if selected == 'Covid-19 Prediction':

    # Page title
    st.title("Covid-19 Prediction using ML")

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Dry_Cough = st.text_input('Dry Cough')
    with col2:
        High_Fever = st.text_input('High Fever')
    with col1:
        Sore_Throat = st.text_input('Sore Throat')
    with col2:
        Difficulty_in_breathing = st.text_input('Difficulty in breathing')

# Predict diabetes on button click
if st.button('Cov_Predict'):
    input_data = (Dry_Cough, High_Fever, Sore_Throat, Difficulty_in_breathing)
    result = predict_result(input_data)

    st.write(f"Input data: {input_data}")

    if result == 0:
        st.success('The person is not affected with COVID-19')
    else:
        st.success('The person is affected with COVID-19')
        print("Treatment\n hsdvbubd")
