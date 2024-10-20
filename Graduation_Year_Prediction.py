import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Load data
train_data = pd.read_excel('D:\\aniket data\\github folder\\cloud_counselage\\01_Train_Data.xlsx')
test_data = pd.read_excel('D:\\aniket data\\github folder\\cloud_counselage\\02_Test_Data.xlsx')
final_data = pd.read_excel('D:\\aniket data\\github folder\\cloud_counselage\\final_lead_data.xlsx')

# Data preprocessing
train_data = train_data.ffill()
test_data = test_data.ffill()
final_data = final_data.ffill()

# Drop rows with NaN in the target variable y_train
train_data = train_data.dropna(subset=['Year of Graduation'])

# Ensure 'Year of Graduation' is numeric
train_data['Year of Graduation'] = pd.to_numeric(train_data['Year of Graduation'], errors='coerce')
train_data = train_data.dropna(subset=['Year of Graduation'])  # Drop rows where conversion failed

# Feature selection
X_train = train_data[['College Name', 'CGPA']]  # Example features in training data
y_train = train_data['Year of Graduation']

# Ensure 'Year of Graduation' is numeric for the test data
test_data['Year of Graduation'] = pd.to_numeric(test_data['Year of Graduation'], errors='coerce')
test_data = test_data.dropna(subset=['Year of Graduation'])  # Drop rows where conversion failed
y_test = test_data['Year of Graduation']

# Feature selection for X_test
X_test = test_data[['College Name', 'CGPA']]
X_final = final_data[['New College Name', 'Academic Year']]

# Ensure consistent columns
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)
X_final = pd.get_dummies(X_final)

# Align columns to ensure the same structure between training and test/final data
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
X_final = X_final.reindex(columns=X_train.columns, fill_value=0)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting on the test and final data
y_pred_test = model.predict(X_test)
y_pred_final = model.predict(X_final)

# Ensure length of predictions matches test_data
if len(y_pred_test) == len(test_data):
    test_data['PredictedGraduationYear'] = y_pred_test
else:
    print("Length mismatch between predictions and test data.")

# Save predictions
test_data.to_excel('Graduation_Year_Predictions_Test.xlsx', index=False)
final_data['PredictedGraduationYear'] = y_pred_final
final_data.to_excel('Graduation_Year_Predictions_Final.xlsx', index=False)

# Calculate summary metrics
total_students_test = len(test_data)
total_colleges_test = test_data['College Name'].nunique()

total_students_final = len(final_data)
total_colleges_final = final_data['New College Name'].nunique()

# Print results
print("Predicted Graduation Years for Test Data:")
print(test_data[['College Name', 'CGPA', 'PredictedGraduationYear']].head())

print("\nTotal number of students in test data:", total_students_test)
print("Total number of colleges in test data:", total_colleges_test)

print("\nPredicted Graduation Years for Final Data:")
print(final_data[['New College Name', 'Academic Year', 'PredictedGraduationYear']].head())

print("\nTotal number of students in final data:", total_students_final)
print("Total number of colleges in final data:", total_colleges_final)

# Model evaluation
if 'Year of Graduation' in test_data.columns:
    print("\nMean Absolute Error on Test Data:", round(mean_absolute_error(y_test, y_pred_test),2))

# Plotting the results
plt.figure(figsize=(14, 7))

# Plot Test Data Predictions vs Actual Values
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_test, color='blue', alpha=0.5, edgecolor='k', linewidths=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel('Actual Graduation Year')
plt.ylabel('Predicted Graduation Year')
plt.title('Test Data: Actual vs Predicted Graduation Year')

# Plot Final Data Predictions
plt.subplot(1, 2, 2)
plt.scatter(range(len(y_pred_final)), y_pred_final, color='green', alpha=0.6, edgecolor='k', linewidths=0.5)
plt.xlabel('Sample Index')
plt.ylabel('Predicted Graduation Year')
plt.title('Final Data: Predicted Graduation Year')

plt.tight_layout()
plt.show()
