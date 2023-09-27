# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 17:38:59 2023

@author: rakes
"""

# Step 1: Load in the dataset
from sklearn import datasets

# Load a dataset (e.g., the Iris dataset)
iris = datasets.load_iris()

# The dataset is loaded into a dictionary-like object
# You can access data, target, feature names, and more
X = iris.data  # Features
y = iris.target  # Target variable
feature_names = iris.feature_names  # Feature names
target_names = iris.target_names  # Target class names

import pandas as pd
import numpy as np
import matplotlib as plt

# Step 2: Explore the dataset
# Let's start by looking at the features, target, and basic statistics.

# Display the feature names
print("Feature names:", iris.feature_names)

# Display the target names (species names)
print("Target names:", iris.target_names)

# Display the first few rows of the data
print("First 5 rows of data:")
print(iris.data[:5])

# Display the corresponding target values (species)
print("Target values for the first 5 rows:")
print(iris.target[:5])

# Get some basic statistics about the dataset
print("Basic statistics:")
print("Number of samples:", iris.data.shape[0])
print("Number of features:", iris.data.shape[1])

import numpy as np

print("Mean values for each feature:")
print(np.mean(iris.data, axis=0))
print("Minimum values for each feature:")
print(np.min(iris.data, axis=0))
print("Maximum values for each feature:")
print(np.max(iris.data, axis=0))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size = 0.75)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Create a logistic regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Predicting the Test set results
y_pred = model.predict(X_test)
# Predict probabilities
probs_y = model.predict_proba(X_test)


from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print("Precision (macro):", precision)
print("Recall (macro):", recall)
print("F1 Score (macro):", f1)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)