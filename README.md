# Credit-Card-Fraud-Detection-System# Credit Card Fraud Detection System

This project detects fraudulent credit card transactions using Machine Learning techniques in Python. The project demonstrates a complete end-to-end Machine Learning workflow including data preprocessing, feature engineering, model training, hyperparameter tuning, and fraud detection model evaluation.

## Features
- Data Cleaning and Preprocessing
- Handling Missing Values
- Feature Scaling
- Exploratory Data Analysis (EDA)
- Handling Imbalanced Dataset
- Train-Test Split
- Machine Learning Model Training
- Hyperparameter Tuning using GridSearchCV
- Fraud Transaction Detection
- Model Evaluation and Visualization

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Machine Learning Concepts Used
- Pipeline
- ColumnTransformer
- SimpleImputer
- StandardScaler
- RandomForestClassifier
- LogisticRegression
- GridSearchCV
- Confusion Matrix
- Classification Report

## Dataset
The dataset contains transaction-related information such as:
- Transaction Amount
- Transaction Time
- Customer Details
- Merchant Information
- Transaction Features
- Payment Information

Target Variable:
- Fraud (0 = Genuine Transaction, 1 = Fraudulent Transaction)

## Project Workflow
1. Load Dataset
2. Perform Data Cleaning
3. Handle Missing Values
4. Perform Exploratory Data Analysis
5. Handle Imbalanced Dataset
6. Scale Numerical Features
7. Split Dataset into Training and Testing Data
8. Train Machine Learning Model
9. Tune Hyperparameters using GridSearchCV
10. Evaluate Model Performance
11. Detect Fraudulent Transactions

## Output
The model predicts whether a transaction is fraudulent and displays:
- Accuracy Score
- Confusion Matrix
- Classification Report
- Best Hyperparameters

## How to Run

Install required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
