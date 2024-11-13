# src/train_model.py

import pandas as pd
from data_processing import load_data, preprocess_data
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, roc_auc_score, roc_curve,
                             precision_recall_curve, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10]
    }

    # Initialize Random Forest Classifier
    rf = RandomForestClassifier(random_state=42)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )

    # Fit the model
    grid_search.fit(X_train, y_train)

    # Save the best model
    best_model = grid_search.best_estimator_
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(best_model, 'models/churn_model.pkl')

    return best_model, X_test, y_test, grid_search

def evaluate_model(model, X_test, y_test, grid_search):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    if not os.path.exists('reports'):
        os.makedirs('reports')

    # Classification Report
    report = classification_report(y_test, y_pred, output_dict=True)
    with open('reports/classification_report.json', 'w') as f:
        json.dump(report, f)

    # ROC AUC Score
    roc_auc = roc_auc_score(y_test, y_proba)
    with open('reports/roc_auc_score.txt', 'w') as f:
        f.write(str(roc_auc))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title('Receiver Operating Characteristic')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig('reports/roc_curve.png')
    plt.close()

    # Save ROC data for interactive plot
    roc_data = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'roc_auc': roc_auc}
    with open('reports/roc_data.json', 'w') as f:
        json.dump(roc_data, f)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure()
    plt.plot(recall, precision)
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig('reports/precision_recall_curve.png')
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('reports/confusion_matrix.png')
    plt.close()

    # Feature Importances
    feature_importances = pd.Series(model.feature_importances_, index=X_test.columns)
    top_features = feature_importances.nlargest(10)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=top_features.values, y=top_features.index)
    plt.title('Top 10 Feature Importances')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig('reports/feature_importances.png')
    plt.close()

    # Best Hyperparameters
    best_params = grid_search.best_params_
    with open('reports/best_params.json', 'w') as f:
        json.dump(best_params, f)

if __name__ == '__main__':
    df = load_data()
    X, y = preprocess_data(df)
    model, X_test, y_test, grid_search = train_model(X, y)
    evaluate_model(model, X_test, y_test, grid_search)
