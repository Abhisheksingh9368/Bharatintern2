#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[ ]:


# Load your dataset (replace 'dataset.csv' with your actual dataset file)
data = pd.read_csv('dataset.csv')


# In[ ]:


# Define features and target variable
features = ['SocioEconomicStatus', 'Age', 'Gender']  # Add more features as needed
X = data[features]
y = data['Survived'] 


# In[ ]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)


# In[ ]:


# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)


# In[ ]:


# Display the results
print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')

