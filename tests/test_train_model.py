# tests/test_train_model.py

import unittest
from src.data_processing import load_data, preprocess_data
from src.train_model import train_model
import os

class TestTrainModel(unittest.TestCase):
    def test_train_model(self):
        df = load_data()
        X, y = preprocess_data(df)
        model, X_test, y_test, grid_search = train_model(X, y)
        self.assertIsNotNone(model)
        self.assertFalse(X_test.empty)
        self.assertFalse(y_test.empty)
        self.assertTrue(os.path.exists('models/churn_model.pkl'))

if __name__ == '__main__':
    unittest.main()
