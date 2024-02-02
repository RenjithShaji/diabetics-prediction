import numpy as np
import pickle
import pandas as pd
import streamlit as st
from PIL import Image

# Load the model and scaler
with open("classifier.pkl", "rb") as f:
    rf = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

def welcome():
    return "Welcome All"

def predict_diabetics(inputs):
    # Scale the inputs using the loaded scaler
    scaled_inputs = scaler.transform([inputs])
    
    # Make a prediction using the loaded model
    prediction = rf.predict(scaled_inputs)
    
    return prediction

def main():
    st.title("Diabetics Prediction")
    # st.image("C:\\Users\\User\\OneDrive\\Desktop\\python\\attachment_67726655.png")  # Add your logo image file in the same directory

    html_temp = """
    <style>
    .container {
        background-color: #F63366;
        padding: 10px;
        border-radius: 10px;
    }
    .container h2 {
        color: white;
        text-align: center;
        font-family: Arial, sans-serif;
    }
    </style>
    <div class="container">
    <h2>Streamlit Diabetics Prediction ML App</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    
    # Collect inputs from the user
    inputs = []
    
    for feature in ["HighBP","HighChol", "CholCheck", "Smoker", "Stroke", "HeartDiseaseorAttack", "PhysActivity", "HvyAlcoholConsump", "DiffWalk"]:
        value = st.selectbox(feature, options=[0,1])
        inputs.append(value)

    inputs.append(st.slider("Age", 0, 100))
    inputs.append(st.slider("BMI", 0, 100))
    inputs.append(st.slider("GenHlth", 0, 5))
    inputs.append(st.slider("PhysHlth", 0, 5))
    inputs.append(st.slider("Education", 0, 6))
    inputs.append(st.slider("Income", 0, 11))
  
    if st.button("Submit"):
        inputs = [float(i) for i in inputs]
        result = predict_diabetics(inputs)
        if result == 1:
            st.success('The output: patient is diabetic')
        elif result == 0:
            st.success('The output: patient is not diabetic')

    if st.button("About"):
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()




# python -m streamlit run app.py
