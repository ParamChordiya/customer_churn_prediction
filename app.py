# frontend/app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import json
import joblib
import os

# Load the trained model
model = joblib.load('models/churn_model.pkl')
feature_columns = model.feature_names_in_

# Import the preprocessing function
from src.data_processing import preprocess_new_data

st.title('Customer Churn Prediction')

menu = ['Prediction', 'Model Evaluation']
choice = st.sidebar.selectbox('Menu', menu)

if choice == 'Prediction':
    st.subheader('Predict Customer Churn')

    # Collect user input
    gender = st.selectbox('Gender', ['Female', 'Male'])
    senior_citizen = st.selectbox('Senior Citizen', [0, 1])
    partner = st.selectbox('Partner', ['Yes', 'No'])
    dependents = st.selectbox('Dependents', ['Yes', 'No'])
    tenure = st.number_input('Tenure (months)', min_value=0, max_value=72, value=1)
    phone_service = st.selectbox('Phone Service', ['Yes', 'No'])
    multiple_lines = st.selectbox('Multiple Lines', ['Yes', 'No', 'No phone service'])
    internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
    online_backup = st.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])
    device_protection = st.selectbox('Device Protection', ['Yes', 'No', 'No internet service'])
    tech_support = st.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])
    streaming_tv = st.selectbox('Streaming TV', ['Yes', 'No', 'No internet service'])
    streaming_movies = st.selectbox('Streaming Movies', ['Yes', 'No', 'No internet service'])
    contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
    payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    monthly_charges = st.number_input('Monthly Charges', min_value=0.0, value=0.0)
    total_charges = st.number_input('Total Charges', min_value=0.0, value=0.0)

    customer_data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }

    if st.button('Predict'):
        with st.spinner('Making prediction...'):
            try:
                # Convert input data to DataFrame
                df = pd.DataFrame([customer_data])

                # Preprocess input data
                X = preprocess_new_data(df, feature_columns)

                # Make prediction
                prediction = model.predict(X)
                probability = model.predict_proba(X)[:, 1]

                if prediction[0] == 1:
                    st.error(f"The customer is likely to churn with a probability of {probability[0]:.2f}.")
                else:
                    st.success(f"The customer is not likely to churn with a probability of {probability[0]:.2f}.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

elif choice == 'Model Evaluation':
    st.subheader('Model Evaluation Metrics')

    # Display Classification Report
    st.write('### Classification Report')
    if os.path.exists('reports/classification_report.json'):
        with open('reports/classification_report.json', 'r') as f:
            report = json.load(f)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)
    else:
        st.warning('Classification report not found. Please run the training script.')

    # Display ROC AUC Score
    st.write('### ROC AUC Score')
    if os.path.exists('reports/roc_auc_score.txt'):
        with open('reports/roc_auc_score.txt', 'r') as f:
            roc_auc = f.read()
        st.write(f'ROC AUC Score: {roc_auc}')
    else:
        st.warning('ROC AUC score not found. Please run the training script.')

    # Display ROC Curve
    st.write('### ROC Curve')
    if os.path.exists('reports/roc_data.json'):
        with open('reports/roc_data.json', 'r') as f:
            roc_data = json.load(f)
        roc_fig = px.area(
            x=roc_data['fpr'],
            y=roc_data['tpr'],
            title=f'ROC Curve (AUC = {roc_data["roc_auc"]:.2f})',
            labels=dict(x='False Positive Rate', y='True Positive Rate'),
            width=700, height=500
        )
        roc_fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        st.plotly_chart(roc_fig)
    else:
        st.warning('ROC data not found. Please run the training script.')

    # Display Precision-Recall Curve
    st.write('### Precision-Recall Curve')
    if os.path.exists('reports/precision_recall_curve.png'):
        st.image('reports/precision_recall_curve.png')
    else:
        st.warning('Precision-Recall Curve not found. Please run the training script.')

    # Display Confusion Matrix
    st.write('### Confusion Matrix')
    if os.path.exists('reports/confusion_matrix.png'):
        st.image('reports/confusion_matrix.png')
    else:
        st.warning('Confusion Matrix not found. Please run the training script.')

    # Display Feature Importances
    st.write('### Feature Importances')
    if os.path.exists('reports/feature_importances.png'):
        st.image('reports/feature_importances.png')
    else:
        st.warning('Feature Importances not found. Please run the training script.')

    # Display Best Hyperparameters
    st.write('### Best Hyperparameters')
    if os.path.exists('reports/best_params.json'):
        with open('reports/best_params.json', 'r') as f:
            best_params = json.load(f)
        st.json(best_params)
    else:
        st.warning('Best hyperparameters not found. Please run the training script.')
