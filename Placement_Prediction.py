import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load data
train_data = pd.read_excel('D:\\aniket data\\github folder\\cloud_counselage\\01_Train_Data.xlsx')
test_data = pd.read_excel('D:\\aniket data\\github folder\\cloud_counselage\\02_Test_Data.xlsx')
final_data = pd.read_excel('D:\\aniket data\\github folder\\cloud_counselage\\final_lead_data.xlsx')

# Data preprocessing
train_data.ffill(inplace=True)
test_data.ffill(inplace=True)
final_data.ffill(inplace=True)

# Feature selection and encoding for training data
X_train = train_data[['CGPA', 'Speaking Skills', 'ML Knowledge']]  # Replace with actual columns from train_data
y_train = train_data['Placement Status']

X_train = pd.get_dummies(X_train)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prepare test and final data
# Replace with actual columns from your datasets
X_test = pd.get_dummies(test_data[['CGPA', 'Speaking Skills', 'ML Knowledge']])
X_final = pd.get_dummies(final_data[['Academic Year', 'Branch/ Specialisation']])  # Adjust as needed

# Align columns
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
X_final = X_final.reindex(columns=X_train.columns, fill_value=0)

# Predicting
y_pred_test = model.predict(X_test)
y_pred_final = model.predict(X_final)

# Save predictions
test_data['PredictedPlacementStatus'] = y_pred_test
final_data['PredictedPlacementStatus'] = y_pred_final

test_data.to_excel('Placement_Predictions_Test.xlsx', index=False)
final_data.to_excel('Placement_Predictions_Final.xlsx', index=False)

# Handle NaN values in 'Placement Status' by filling them with a default value (e.g., 0 for 'Not placed')
test_data['Placement Status'].fillna(0, inplace=True)

# Convert 'Placement Status' to integer
y_true_numerical = test_data['Placement Status'].astype(int)

# Convert predicted labels to numerical values
label_mapping = {'Placed': 1, 'Not placed': 0}
y_pred_test_numerical = np.array([label_mapping[label] for label in y_pred_test])

# Evaluate the model
accuracy = accuracy_score(y_true_numerical, y_pred_test_numerical)*100
print("\nAccuracy on Test Data:", accuracy)
print("Classification Report on Test Data:\n", classification_report(y_true_numerical, y_pred_test_numerical))

# Visualize the results
plt.figure(figsize=(14, 7))

# Plot distribution of placement predictions for test data
plt.subplot(1, 2, 1)
plt.hist(y_pred_test_numerical, bins=2, color='blue', edgecolor='k', alpha=0.7)
plt.xlabel('Placement Status (0 = Not Placed, 1 = Placed)')
plt.ylabel('Number of Students')
plt.title('Test Data Placement Predictions')

# Plot distribution of placement predictions for final data
plt.subplot(1, 2, 2)
plt.hist(y_pred_final, bins=2, color='red', edgecolor='k', alpha=0.7)
plt.xlabel('Placement Status (0 = Not Placed, 1 = Placed)')
plt.ylabel('Number of Students')
plt.title('Final Data Placement Predictions')

plt.tight_layout()
plt.show()
