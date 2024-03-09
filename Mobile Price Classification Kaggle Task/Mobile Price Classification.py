# -*- coding: utf-8 -*-
"""

Mobile Price Classification
Classify Mobile Price Range

About Dataset
Context
Bob has started his own mobile company. He wants to give tough fight to big companies like Apple,Samsung etc.

He does not know how to estimate price of mobiles his company creates. In this competitive mobile phone market you cannot simply assume things. To solve this problem he collects sales data of mobile phones of various companies.

Bob wants to find out some relation between features of a mobile phone(eg:- RAM,Internal Memory etc) and its selling price. But he is not so good at Machine Learning. So he needs your help to solve this problem.

In this problem you do not have to predict actual price but a price range indicating how high the price is.


"""
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load training and test datasets
Train_DF = pd.read_csv("C:\\Users\\rakes\\Documents\\Python Fun\\Mobile Price Classification Kaggle Task\\archive (3)\\train.csv")
Test_DF = pd.read_csv("C:\\Users\\rakes\\Documents\\Python Fun\\Mobile Price Classification Kaggle Task\\archive (3)\\test.csv")

# Handling Missing Values
# Check for missing values in the training dataset
missing_train = Train_DF.isnull().sum()
print("Missing Values in Training Dataset:")
print(missing_train)

# Check for missing values in the test dataset
missing_test = Test_DF.isnull().sum()
print("\nMissing Values in Test Dataset:")
print(missing_test)

# Since there are no missing values in both datasets, no imputation is needed

# Encoding Categorical Variables
# There are no categorical variables to encode in this dataset, as the binary variables are already in numeric format (0 and 1)

# Feature Scaling
# Extract features and target variable for both training and test datasets
X_train = Train_DF.drop(columns=['price_range'])  # Features for training
y_train = Train_DF['price_range']  # Target variable for training

# Exclude 'id' column from the test dataset
X_test = Test_DF.drop(columns=['id'])  # Features for testing (no target variable)

# Perform feature scaling on training and test features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Now X_train_scaled and X_test_scaled contain the scaled features

# Define models to evaluate
models = {
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Support Vector Machine": SVC()
}

# Evaluate models using cross-validation on the training data
for name, model in models.items():
    y_pred_train = cross_val_predict(model, X_train_scaled, y_train, cv=5)
    accuracy = accuracy_score(y_train, y_pred_train)
    precision = precision_score(y_train, y_pred_train, average='weighted')
    recall = recall_score(y_train, y_pred_train, average='weighted')
    f1 = f1_score(y_train, y_pred_train, average='weighted')
    
    print(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

# Train the best-performing model on the entire training data
best_model = GradientBoostingClassifier()  # Choose the best-performing model based on cross-validation results
best_model.fit(X_train_scaled, y_train)

# Make predictions on the test data
y_pred_test = best_model.predict(X_test_scaled)

# Add predictions to the Test_DF DataFrame
Test_DF['predicted_price_range'] = y_pred_test

# Specify the directory to save the CSV file
save_directory = "C:\\Users\\rakes\\Documents\\Python Fun\\Mobile Price Classification Kaggle Task\\archive (3)"

# Save the updated Test_DF DataFrame with predictions to a new CSV file in the specified directory
save_path = os.path.join(save_directory, "test_data_with_predictions.csv")
Test_DF.to_csv(save_path, index=False)

print(f"CSV file saved to: {save_path}")


