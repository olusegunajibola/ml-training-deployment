import pickle

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay,  precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

import os

dir = os.getcwd()
train = pd.read_csv(dir+"/data/train.csv")

train_new = train.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1)

mode_embarked = train_new.Embarked.mode()
mean_age = round(train_new.Age.mean(), 2)

train_new.Age.fillna(mean_age, inplace=True)
train_new.Embarked.fillna(mode_embarked[0], inplace=True)

y = train_new.Survived
X = train_new.drop(['Survived'], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def transform_to_vector(X, dv=None):
    """
    Transforms a DataFrame into a vectorized format using DictVectorizer.

    Parameters:
    X (pd.DataFrame): The input DataFrame to be transformed.
    dv (DictVectorizer, optional): A pre-fitted DictVectorizer instance.
                                   If None, a new instance will be created and fitted.

    Returns:
    sparse_matrix: The vectorized sparse matrix.
    DictVectorizer: The DictVectorizer instance (either newly fitted or provided).
    """
    # Convert DataFrame to a list of dictionaries
    dicts = X.to_dict(orient='records')

    # If no DictVectorizer is provided, fit a new one
    if dv is None:
        dv = DictVectorizer()
        vectorized_data = dv.fit_transform(dicts)
    else:
        # Use the provided DictVectorizer for transformation
        vectorized_data = dv.transform(dicts)

    return vectorized_data, dv

model  = LogisticRegression(l1_ratio=0.3, max_iter=10000,
                   penalty='elasticnet', random_state=42, solver='saga')


X_train2, dv = transform_to_vector(X_train)
X_test2, _ = transform_to_vector(X_test, dv=dv)

model.fit(X_train2, y_train)

y_pred = model.predict(X_test2)
# train_features, test_features, train_labels, test_labels

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Display a detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

output_file = f'model_train_py.bin'

# Save the model

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'the model is saved to {output_file}')



