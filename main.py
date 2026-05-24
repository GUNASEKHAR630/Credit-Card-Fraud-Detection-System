import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ====================== 1. LOAD DATA ======================
print("🔄 Loading Employee Salary Dataset...")

url = "https://raw.githubusercontent.com/krishnaik06/simple-Linear-Regression/master/Salary_Data.csv"
df = pd.read_csv(url)

print(f"Dataset Shape: {df.shape}")
print("Columns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

# ====================== 2. PREPROCESSING ======================
X = df[['YearsExperience']]   # Feature
y = df['Salary']              # Target

print("\nMissing Values:\n", df.isnull().sum())

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# ====================== 3. MODEL TRAINING ======================
print("\n🚀 Training models...")

# Linear Regression (Best for this simple dataset)
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Random Forest (for comparison)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

print("✅ Models trained successfully!")

# ====================== 4. MODEL EVALUATION ======================
def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n📊 {name} Performance:")
    print(f"Mean Absolute Error : ${mae:,.2f}")
    print(f"Root Mean Squared Error: ${rmse:,.2f}")
    print(f"R² Score: {r2:.4f} ({r2*100:.2f}%)")

print("="*60)
evaluate_model(lr_model, X_test_scaled, y_test, "Linear Regression")
evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest Regressor")

# ====================== 5. SAVE MODEL ======================
joblib.dump(lr_model, 'salary_model.pkl')
joblib.dump(scaler, 'salary_scaler.pkl')
print("\n💾 Model and Scaler saved successfully!")

# ====================== 6. PREDICTION FUNCTION ======================
def predict_salary(years_experience):
    """Predict salary based on years of experience"""
    model = joblib.load('salary_model.pkl')
    scaler = joblib.load('salary_scaler.pkl')
    
    input_data = np.array([[years_experience]])
    input_scaled = scaler.transform(input_data)
    
    predicted_salary = model.predict(input_scaled)[0]
    return round(predicted_salary, 2)

# ====================== 7. EXAMPLE USAGE ======================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("💼 SALARY PREDICTION SYSTEM READY!")
    print("="*70)
    
    # Example Prediction
    exp = 8.5
    predicted = predict_salary(exp)
    print(f"\nFor {exp} Years of Experience:")
    print(f"Predicted Annual Salary: ${predicted:,.2f}")
    
    # Interactive Prediction
    while True:
        try:
            user_exp = input("\nEnter Years of Experience (or 'q' to quit): ")
            if user_exp.lower() == 'q':
                break
            exp = float(user_exp)
            pred = predict_salary(exp)
            print(f"→ Predicted Salary: ${pred:,.2f}")
        except:
            print("Plase enter a valid number")



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
from imblearn.over_sampling import SMOTE
import joblib


print("Loading Credti card fraud detection dataset...")

try:
    df = pd.read_csv('creditcard.csv')
except FileNotFoundError:
    print("Dataset not found: please download 'creditcard.csv' fromkaggle and place it in this folder.")
    print("Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    exit()

print(f"Dataset shape: {df.shape}")
print("\nClass Distribution:")
print(df['Class'].value_counts())
print(f"Fraud Rate: {(df['Class'].value_counts()[1]/len(df))*100:.4f}% (Highly Imbalancd)")

plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df)
plt.title("Fraud vs Non-Fraud Transactions")
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(x='Class', y = 'Amount', data=df)
plt.title('Transaction Amount by Class')
plt.yscale('log')
plt.show()

x = df.drop('Class', axis=1)
y = df['Class']

scaler = StandardScaler()
x[['Time', 'Amount']] = scaler.fit_transform(x[['Time', 'Amount']])


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining samples: {x_train.shape[0]:,}")
print(f"Testing samples : {x_test.shape[0]:,}")

print("\n Applying SMOTE to balance training data...")

smote = SMOTE(sampling_strategy='auto', random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

print(f"After SMOTE - Training samples: {x_train_resampled.shape[0]:,}")
print(f"Class distribution after SMOTE: {pd.Series(y_train_resampled).value_counts()}")

print("\n Training models....")
lr_model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=500)
lr_model.fit(x_train, y_train)
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    random_state=42,
    n_jobs=1,
    class_weight='balanced'
)
rf_model.fit(x_train, y_train)
print("Models traind successfully.")
def evaluate_fraud_model(model, x_test, y_test, model_name):
    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)[:, 1]

    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall : {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC : {roc_auc_score(y_test, y_pred_proba):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

print("="*60)
evaluate_fraud_model(lr_model, x_test, y_test, "Logistic Regression")
evaluate_fraud_model(lr_model, x_test, y_test, "Random Forest")

joblib.dump(rf_model, 'credit_fraud_rf_model.pkl')
joblib.dump(scaler, 'credit_fraud_scaler.pkl')
print("\n Random Forest model and scaler saved successfully")

def predict_fraud(transaction_data):
    model = joblib.load('credit_fraud_rf_model.pkl')
    scaler = joblib.load('credit_fraud_scaler.pkl')

    df_pred = pd.DataFrame([transaction_data])

    df_pred[['Time', 'Amount']] = scaler.transform(df_pred[['Time', 'Amount']])
    probability = model.predict_proba(df_pred)[0][1]
    prediction = "Fraud" if probability >=0.5 else "Normal"

    return {
        "Prediction" : prediction,
        "Fraud_Probability" : round(probability*100, 4)
    }
if __name__ == "__main__":
    print("\n" + "="*70)
    print("CREDIT CARD FRAUD DETECTION SYSTM READY")
    print("="*70)

    example_transaction = {
        'Time': 0.0,
        'V1':-1.359807,
        'V2': -0.072781,
        'V3': 2.536347,
        'V4': 1.378155,
        'V5': -0.338321,
        'V6': 0.462388,
        'V7': 0.239599,
        'V8': 0.098698,
        'V9': 0.363787,
        'V10': 0.090794,
        'V11': -0.551600,
        'V12': -0.617801,
        'V13': -0.991390,
        'V14': -0.311169,
        'V15': 1.468177,
        'V16': -0.470401,
        'V17': 0.207971,
        'V18': 0.025791,
        'V19': 0.403993,
        'V20': 0.251412,
        'V21': -0.018307,
        'V22': 0.277838,
        'V23': -0.110474,
        'V24': 0.066928,
        'V25': 0.128539,
        'V26': -0.189115,
        'V27': 0.133558,
        'V28': -0.021053,
        'Amount':149.62
    }

    result = predict_fraud(example_transaction)
    print("\nExample Transaction Prediction:")
    print(f"Status : {result['Prediction']}")
    print(f"Fraud Probability: {result['Fraud_Probability']}%")
