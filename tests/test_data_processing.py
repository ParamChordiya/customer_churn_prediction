import unittest
from src.data_processing import load_data, preprocess_data, preprocess_new_data
import pandas as pd

class TestDataProcessing(unittest.TestCase):
    def test_load_data(self):
        df = load_data()
        self.assertFalse(df.empty)

    def test_preprocess_data(self):
        df = load_data()
        X, y = preprocess_data(df)
        self.assertEqual(len(X), len(y))
        self.assertFalse(X.empty)
        self.assertFalse(y.empty)

    def test_preprocess_new_data(self):
        df = pd.DataFrame({
            'gender': ['Male'],
            'SeniorCitizen': [0],
            'Partner': ['Yes'],
            'Dependents': ['No'],
            'tenure': [12],
            'PhoneService': ['Yes'],
            'MultipleLines': ['No'],
            'InternetService': ['DSL'],
            'OnlineSecurity': ['Yes'],
            'OnlineBackup': ['No'],
            'DeviceProtection': ['No'],
            'TechSupport': ['Yes'],
            'StreamingTV': ['No'],
            'StreamingMovies': ['No'],
            'Contract': ['Month-to-month'],
            'PaperlessBilling': ['Yes'],
            'PaymentMethod': ['Electronic check'],
            'MonthlyCharges': [29.85],
            'TotalCharges': [382.2]
        })
        feature_columns = joblib.load('models/churn_model.pkl').feature_names_in_
        X_new = preprocess_new_data(df, feature_columns)
        self.assertFalse(X_new.empty)
        self.assertEqual(X_new.shape[1], len(feature_columns))

if __name__ == '__main__':
    unittest.main()