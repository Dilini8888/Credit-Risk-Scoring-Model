  # Credit Risk Scoring Model

## Project Overview
This project uses machine learning to predict whether a credit card client is likely to default on their payment next month. It compares Logistic Regression and Random Forest models using a real-world credit dataset.

## Objectives
- Predict potential credit card payment defaults.
- Compare machine learning models.
- Evaluate model performance using Accuracy, Recall, F1-score, and ROC-AUC.

## Technologies Used
- Python
- Pandas
- Scikit-learn

## Dataset
The project uses the **Default of Credit Card Clients** dataset from the UCI Machine Learning Repository.

- Total records: 30,000
- Input features: 23
- Target: Default payment next month (0 = No, 1 = Yes)

The dataset file is not included in this repository.

## Machine Learning Models
- Logistic Regression
- Balanced Logistic Regression
- Random Forest Classifier

## Current Best Model
**Random Forest Classifier**

- Accuracy: 77.02%
- Class 1 Recall: 60.18%
- Class 1 F1-score: 53.74%
- ROC-AUC: 0.77335

These results are from the current experimental train-test split and may vary if the data split or model settings change.

## How to Run
1. Install Python.
2. Install the required libraries:
   `pip install pandas scikit-learn xlrd`
3. Place the dataset file named `default of credit card clients.xls` in the project folder.
4. Run:
   `python credit_risk_model.py`

## Project Status
Ongoing — model evaluation and improvement are in progress.

## Author
Dilini Dissanayake

GitHub: [Dilini8888](https://github.com/Dilini8888)