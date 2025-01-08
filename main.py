from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import os

app = Flask(__name__)

# Loading resources: trained model and scaler
model_path = "saved_model/svm_model.pkl"
scaler_path = "saved_model/scaler.pkl"

try:
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except FileNotFoundError:
    print("Model or scaler file not found! Please ensure they are in the correct path.")
    exit(1)

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction page
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extracting the input features from the form
        pregnancies = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        blood_pressure = float(request.form['blood_pressure'])
        skin_thickness = float(request.form['skin_thickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        diabetes_pedigree_function = float(request.form['diabetes_pedigree_function'])
        age = float(request.form['age'])
    except ValueError:
        return render_template('index.html', prediction_text="Please enter valid values for all fields.")
    
    # Data preparation for prediction
    bmi_age = bmi * age
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age, bmi_age]])
    
    # Scaling input data using the saved scaler
    input_data_scaled = scaler.transform(input_data)
    
    # Making predictions using the trained model
    prediction = model.predict(input_data_scaled)
    
    # Prediction result
    if prediction == 1:
        result = "Yes, you are likely to develop diabetes."
    else:
        result = "No, you are not likely to develop diabetes."

    # Storing user input and outcome
    user_input = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "Blood Pressure": blood_pressure,
        "Skin Thickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "Diabetes Pedigree Function": diabetes_pedigree_function,
        "Age": age,
        "BMI*Age": bmi_age,
        "Prediction": result
    }

   # Save user input to Excel file (append if exists, create if not)
    try:
        if os.path.exists("user_input.xlsx"):
            existing_df = pd.read_excel("user_input.xlsx")
            new_df = pd.DataFrame([user_input])
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
            updated_df.to_excel("user_input.xlsx", index=False)
        else:
            pd.DataFrame([user_input]).to_excel("user_input.xlsx", index=False)
    except Exception as e:
        print(f"Error saving user input: {e}")

    # Save outcome to a text file (append if exists, create if not)
    try:
        with open("outcome.txt", "a") as outcome_file:
            outcome_file.write(f"Time: {datetime.now()} | Outcome: {result}\n")
    except Exception as e:
        print(f"Error saving outcome: {e}")

    return render_template('index.html', prediction_text=result)

# Launching the app
if __name__ == "__main__":
    app.run(debug=True)
