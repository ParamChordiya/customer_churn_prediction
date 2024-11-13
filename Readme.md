# Customer Churn Prediction Project

This project is an end-to-end machine learning application that predicts customer churn for a telecom company. It includes data processing, model training with hyperparameter tuning, and a Streamlit frontend application that allows users to input customer data and get predictions.

## Project Structure

- **data/**: Contains raw data.
- **models/**: Contains the trained model.
- **reports/**: Contains evaluation metrics and plots.
- **src/**: Source code for data processing and training.
- **frontend/**: Streamlit app for the frontend.
- **tests/**: Unit tests for the project.
- **requirements.txt**: Python dependencies.
- **.github/workflows/**: CI pipeline configuration.

## Setup Instructions

### Prerequisites

- Python 3.8+ installed.
- Install the required Python packages.

### Steps

1. **Clone the repository**

   ```bash
   git clone https://github.com/paramchordiya/customer_churn_prediction.git
   cd customer_churn_prediction
```

Download the dataset

Place the WA_Fn-UseC_-Telco-Customer-Churn.csv file into data/raw/ directory.
Install dependencies

bash
Copy code
pip install -r requirements.txt
Train the model

bash
Copy code
python src/train_model.py
This will preprocess the data, train the model with hyperparameter tuning, and generate evaluation reports.
Run the Streamlit app

bash
Copy code
streamlit run frontend/app.py
Access the Application

Open your browser and navigate to http://localhost:8501 to use the Streamlit app.
Usage
Navigate to http://localhost:8501 in your browser.
Use the Prediction tab to input customer data and predict churn.
Use the Model Evaluation tab to view model performance metrics and plots.
Testing
Run unit tests using:

bash
Copy code
pytest tests/
CI Pipeline
The project includes a GitHub Actions workflow for continuous integration.
On each push to the main branch, tests are run, and the model is trained.
Future Enhancements
Implement authentication and security measures.
Deploy the application to a cloud platform.
Integrate a database for data storage.
Use MLflow or similar tool for experiment tracking.
License
This project is licensed under the MIT License.

Acknowledgements
Dataset from Kaggle: Telco Customer Churn