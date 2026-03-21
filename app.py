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

# ✅ Load model with caching (VERY IMPORTANT)
@st.cache_resource
def load_model():
    scaler = joblib.load('scaler.pkl')
    model = joblib.load('best_model.pkl')
    return scaler, model

scaler, model = load_model()

# UI
st.title("Churn Prediction App")
st.divider()

st.write("Enter details to predict customer churn")
st.divider()

age = st.number_input("Age", min_value=18, max_value=100, value=30)
gender = st.selectbox('Gender', ['Male', 'Female'])
tenure = st.number_input("Tenure", min_value=0, max_value=72, value=10)
monthly_charges = st.number_input("Monthly Charges", min_value=0, value=150)

st.divider()

if st.button("Predict"):
    
    # Convert gender
    gender_selected = 1 if gender == 'Female' else 0

    # Prepare input
    X = np.array([[age, gender_selected, tenure, monthly_charges]])

    # Scale
    X_scaled = scaler.transform(X)

    # Predict
    prediction = model.predict(X_scaled)[0]

    # Convert output
    if prediction == 1:
        result = "Churn ❌"
    else:
        result = "No Churn ✅"

    st.success(f"Prediction: {result}")
    st.balloons()
