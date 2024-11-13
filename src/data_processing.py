# src/data_processing.py

import pandas as pd
import numpy as np

def load_data():
    df = pd.read_csv('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    return df

def preprocess_data(df):
    # Handle missing values
    df.replace(' ', np.nan, inplace=True)
    df.dropna(inplace=True)

    # Drop customerID column
    df.drop('customerID', axis=1, inplace=True)

    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

    # Convert target variable to binary
    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

    # Convert categorical variables to dummy variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Feature and target split
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    return X, y

def preprocess_new_data(df, feature_columns):
    # Handle missing values
    df.replace(' ', np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Ensure correct data types
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
    X = df.reindex(columns=feature_columns, fill_value=0)

    return X
