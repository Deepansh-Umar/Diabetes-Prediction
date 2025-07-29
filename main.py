#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

#read the csv file
df = pd.read_csv('diabetes.csv')

#cleaning data


#check existence of null values
print(df.isnull().sum())

#check existence of 0s in improper fields
columns_to_check = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for col in columns_to_check:
    print(f"{col} has {df[df[col] == 0].shape[0]} zero values")

#remove invalid 0s
for col in columns_to_check:
    median = df[df[col] != 0][col].median()
    df[col] = df[col].replace(0, median)


#Feature and Target Separation
X = df.drop("Outcome", axis=1)
y = df["Outcome"]


#Testing and Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#Model training
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

#model evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

#model deployment
joblib.dump(model, "diabetes_model.pkl")
joblib.dump(scaler, "scaler.pkl")
