# Customer Churn Prediction Project

This project is an end-to-end machine learning application that predicts customer churn for a telecom company. It includes data processing, model training with hyperparameter tuning, and a Streamlit frontend application that allows users to input customer data and get predictions.

## Project Structure

- **data/**: Contains raw data.
- **models/**: Contains the trained model.
- **reports/**: Contains evaluation metrics and plots.
- **src/**: Source code for data processing and training.
- **tests/**: Unit tests for the project.
- **requirements.txt**: Python dependencies.


## Setup Instructions

### Prerequisites

- Python 3.8+ installed.
- Install the required Python packages.

### Steps

1. **Clone the repository**

   ```bash
   git clone https://github.com/paramchordiya/customer_churn_prediction.git
   cd customer_churn_prediction```

2. **Download the dataset**

```Place the WA_Fn-UseC_-Telco-Customer-Churn.csv file into data/raw/ directory.```


```bash
pip install -r requirements.txt
```
3. **Train the model**

```bash

python src/train_model.py
```
This will preprocess the data, train the model with hyperparameter tuning, and generate evaluation reports.
4. **Run the Streamlit app**

```bash

streamlit run frontend/app.py
```
## Access the Application

```bash
https://custchurnprediction.streamlit.app/
```
```
Use the Prediction tab to input customer data and predict churn.
Use the Model Evaluation tab to view model performance metrics and plots.
```

## Acknowledgements
Dataset from Kaggle: Telco Customer Churn