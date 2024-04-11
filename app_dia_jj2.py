import numpy as np
import pandas as pd
import streamlit as st
import pickle
import joblib

model = joblib.load('random_forest_model.pkl')
cols=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']    

def main(): 
    st.title("Predictor de Modelo Diabetes")
    html_temp = """
    <div style="background-color: #800020; padding: 10px">
    <h2 style="color: gold; text-align: center;">Universidad Panamericaca MCD</h2>
    <h3 style="color: gold; text-align: center; font-size: 16px;">Equipo:</h3>
    <ul style="color: gold; text-align: center; font-size: 14px; list-style-type: none;">
    <li>Alberto Jorge Julián</li>
    <li>Javier Alberto Juarez Luna</li>
    <li>Roberto Capitaine Venegas</li>
    </ul>
    </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)
    Pregnancies = st.text_input("Pregnancies", "0") 
    Glucose = st.text_input("Glucose", "0.0")
    BloodPressure = st.text_input("Blood Pressure", "0")
    SkinThickness = st.text_input("Skin Thickness", "0")
    Insulin = st.text_input("Insulin", "0")
    BMI = st.text_input("BMI", "0")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function", "0.000")
    Age = st.text_input("Age", "0")

    if st.button("Predict"): 
        features = [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]
        data = {'Pregnancies': int(Pregnancies), 'Glucose': float(Glucose), 'BloodPressure': int(BloodPressure), 'SkinThickness': int(SkinThickness), 'Insulin': int(Insulin),
                'BMI': float(BMI), 'DiabetesPedigreeFunction': float(DiabetesPedigreeFunction), 'Age': int(Age)}
        print(data)
        df = pd.DataFrame([list(data.values())], columns=cols)
        features_list = df.values.tolist() 
        prediction = model.predict(features_list)

        output = int(prediction[0])
        if output == 1:
            text = "Diabetes"
        else:
            text = "Normal"

        st.success('Los datos indican que la persona es {}'.format(text))

if __name__ == '__main__': 
    main()
