"""
This Python script uses Support Vector Classifier to develop a model that predicts occurrences of diabetes.
This script does not launch the website in the localhost.
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

import pickle
import os

# Loading the dataset
file_path = "Dataset 1 _ Pima Indians diabetes dataset (PIDD).xlsx"
data = pd.read_excel(file_path)

# Data Preprocessing
# Separating features and target variable
X = data.iloc[:, :-1]  # All columns except 'Outcome'
y = data.iloc[:, -1]   # 'Outcome' column

# Handling missing values
columns_with_nan = ['Glucose', 'Blood pressure', 'Skin thickness', 'Insulin', 'Body mass index']
X[columns_with_nan] = X[columns_with_nan].fillna(X[columns_with_nan].mean())

X['BMI_Age'] = X['Body mass index'] * X['Age'] # feature engineering

# Feature Transformation: Log transforming features with skewed distributions
X['Insulin'] = np.log1p(X['Insulin'])
X['Skin thickness'] = np.log1p(X['Skin thickness'])

# Handling Class Imbalance
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Splitting
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Model Training with Hyperparameter Tuning
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1],
    'kernel': ['rbf', 'poly']
}

svm = SVC(probability=True, random_state=42)
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Evaluating the Model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("Best Parameters:", grid_search.best_params_)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Saving the Model and Scaler
model_dir = "saved_model"
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "svm_model.pkl")
with open(model_path, 'wb') as model_file:
    pickle.dump(best_model, model_file)

scaler_path = os.path.join(model_dir, "scaler.pkl")
with open(scaler_path, 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("\nModel and scaler trained and saved successfully at:", model_path, "and", scaler_path)