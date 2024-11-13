# src/api.py

from fastapi import FastAPI
import joblib
import pandas as pd
from data_processing import preprocess_data
import logging
import joblib
from data_processing import preprocess_new_data


app = FastAPI()
model = joblib.load('models/churn_model.pkl')

logging.basicConfig(level=logging.INFO)

@app.post('/predict')
def predict_churn(data: dict):
    # Convert input data to DataFrame
    df = pd.DataFrame([data])
    
    # Preprocess input data
    _, X = preprocess_new_data(df)
    
    # Make prediction
    prediction = model.predict(X)
    probability = model.predict_proba(X)[:, 1]
    
    logging.info(f"Input Data: {data}, Prediction: {prediction[0]}, Probability: {probability[0]}")
    return {'churn': int(prediction[0]), 'probability': float(probability[0])}

# src/api.py

def preprocess_new_data(df):
    # Implement the preprocessing steps
    # Similar to the function in src/api.py
    # Ensure that the processed features match those used during training
    # ...

    # For demonstration, here's a simplified version:
    # Handle missing values
    df.replace(' ', np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
    # Convert numerical columns to numeric types
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
    df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')
    df['SeniorCitizen'] = pd.to_numeric(df['SeniorCitizen'], errors='coerce')
    
    # Convert categorical variables to dummy variables
    categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                        'PaperlessBilling', 'PaymentMethod']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Align the new data with the training data
    X = df.reindex(columns=model.feature_names_in_, fill_value=0)
    
    return X

