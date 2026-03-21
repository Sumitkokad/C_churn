# Gender femal =1 male = 0
# Churn yes = 1 no = 0
# Scaler is exported as scaler.pkl file which is used in the app.py file to 
# scale the input data before making predictions using the trained model. 
# This ensures that the input data is transformed in the same way as the training data, 
# allowing for accurate predictions.

# model is exported as model.pkl

# order of the x is -> ['Age', 'Gender', 'Tenure', 'MonthlyCharges']

# Now starts to use streamlit for interface 
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# File importing --> 

scaler=joblib.load('scaler.pkl')
model=joblib.load('best_model.pkl')


st.title("Churn Prediction App")
st.divider()

st.write("Please enter the following details to predict whether a customer will churn or not and press the Button.")
st.divider()

age=st.number_input("Age",min_value=18,max_value=100,value=30)

gender=st.selectbox('Gender',['Male','Female'])

tenure=st.number_input("Tenure",min_value=0,max_value=72,value=10)

monthly_charges=st.number_input("Monthly Charges",min_value=0,value=150)

st.divider()

PredictButton=st.button("Predict")
st.divider()

if PredictButton:
    
    gender_selected=1 if gender=='Female' else 0

    X=[age, gender_selected, tenure, monthly_charges]

    X1=np.array([X])
    
    X_array=scaler.transform(X1)
    
    predictions=model.predict(X_array)[0]
    
    predicted='Churn'

    st.balloons()
    st.write(f"The predicted result is: {predicted}")


else:
    print("Please enter the details and press the Predict button to get the prediction.")    
