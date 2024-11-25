#%% md
# In this notebook we show the process to train and deploy a model primarily utilizing Python and Docker.
# 
# The data of interest is the titanic data where we predict whether a passenger survive or not.
# 
# Data source: [kaggle](https://www.kaggle.com/competitions/titanic/data).
#%%
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
#%%
dir = os.getcwd()
#%%
train = pd.read_csv(dir+"/data/train.csv")
train.info()
#%% md
# # Data Cleaning and Preprocessing
# 
# We remove PassengerId, Name, Cabin, and Ticket from the data.
#%%
train_new = train.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1)
#%%
train_new.info()
#%% md
# Now, we fill the missing value in the age with the mean, and that of Embarked with the mode.
#%%
mode_embarked = train_new.Embarked.mode()
print('Mode of Embarked:' ,mode_embarked) 
#%%
mean_age = round(train_new.Age.mean(), 2)

print('Mean age:', mean_age)
print('Mode of Embarked:' ,mode_embarked[0]) 
#%%
mode_embarked[0]
#%%
# train_new.fillna({train_new['Age']:mean_age}, inplace=True)
# train_new.fillna({train_new['Embarked']:mode_embarked[0]}, inplace=True)
#%%
train_new.Age.fillna(mean_age, inplace=True)
train_new.Embarked.fillna(mode_embarked[0], inplace=True)
#%%
train_new.info()
#%% md
# # EDA
#%%
train_new.Embarked.value_counts()
#%%
train_new.Survived.value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Survived Value Counts')
plt.xlabel('Survived')
plt.ylabel('Counts')
plt.xticks(rotation=0)  # Keep labels horizontal
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()
#%%
train_new.Sex.value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Gender Value Counts')
plt.xlabel('Gender')
plt.ylabel('Counts')
plt.xticks(rotation=0)  # Keep labels horizontal
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()
#%%
# a histogram
train_new['Age'].plot.hist(bins=10, color='lightblue', edgecolor='black')

plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()
#%%
train_new.SibSp.value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('# of siblings/spouses aboard the Titanic Counts')
plt.xlabel('# of siblings/spouses')
plt.ylabel('Counts')
plt.xticks(rotation=0)  # Keep labels horizontal
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()
#%%
train_new.Parch.value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('# of parents/children aboard the Titanic Counts')
plt.xlabel('# of parents/children')
plt.ylabel('Counts')
plt.xticks(rotation=0)  # Keep labels horizontal
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()
#%%
# a histogram
train_new['Fare'].plot.hist(bins=50, color='lightblue', edgecolor='black')

plt.title('Passenger fare Distribution')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()
#%%
train_new.Embarked.value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Embarked Value Counts')
plt.xlabel('Embarked')
plt.ylabel('Counts')
plt.xticks(rotation=0)  # Keep labels horizontal
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()
#%% md
# Now, we do some bivariate plots.
#%%
# Scatter plot
plt.scatter(train_new['Age'], train_new['Survived'], color='blue', alpha=0.6)

# Customize the plot
plt.title('Age vs Survival')
plt.xlabel('Age')
plt.ylabel('Survived')
plt.yticks([0, 1], ['No', 'Yes'])
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()
#%%
# Boxplot
sns.boxplot(x='Survived', y='Age', data=train_new)

# Customize the plot
plt.title('Age Distribution by Survival')
plt.xlabel('Survived')
plt.ylabel('Age')
plt.xticks([0, 1], ['No', 'Yes'])

# Show the plot
plt.show()
#%%
plt.figure(figsize=(8, 5))
ax = sns.countplot(x='Survived', hue='Sex', data=train_new, palette='Set2')

# Annotate the bars with their values
for container in ax.containers:
    ax.bar_label(container, fmt='%d', label_type='edge', padding=3)

# Customize the plot
plt.title('Survival by Gender')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.xticks([0, 1], ['No', 'Yes'])

# Show the plot
plt.show()
#%%
# Create a count plot for Survived and Embarked
plt.figure(figsize=(8, 5))
ax = sns.countplot(x='Survived', hue='Embarked', data=train_new, palette='Set2')

# Annotate the bars with their values
for container in ax.containers:
    ax.bar_label(container, fmt='%d', label_type='edge', padding=3)

# Customize the plot
plt.title('Survival by Embarked')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.xticks([0, 1], ['No', 'Yes'])

# Show the plot
plt.show()
#%%
# Create a boxplot for Fare and Survived
plt.figure(figsize=(8, 5))
sns.boxplot(x='Survived', y='Fare', data=train_new, palette='Set2')

# Customize the plot
plt.title('Fare Distribution by Survival')
plt.xlabel('Survived')
plt.ylabel('Fare')
plt.xticks([0, 1], ['No', 'Yes'])

# Show the plot
plt.show()
#%% md
# # Models
# 
# At the end of our EDA, we build some models. We use the Logistic Regression, and Naive Bayes Model to solve the classification problem.
#%%
train_new
#%%
train_new.info()
#%%
y = train_new.Survived
X = train_new.drop(['Survived'], axis=1)
#%%
categorical_columns = list(X.dtypes[X.dtypes == 'object'].index)
numerical_columns = list(X.dtypes[X.dtypes != 'object'].index)
categorical_columns, numerical_columns
#%%
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#%%
from sklearn.feature_extraction import DictVectorizer

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
#%%
# dicts = X_train.to_dict(orient = 'records')
# dv = DictVectorizer()
# X_train2 = dv.fit_transform(dicts)
# X_train2
X_train2, dv = transform_to_vector(X_train)
#%% md
# ## Logistic Reg
#%%
lr = LogisticRegression(random_state=42, max_iter=10000)

params_lr = {
    'penalty': ['elasticnet'],
    'solver' : ['saga'],
    'l1_ratio' : np.arange(0., 1.0, 0.1),
}

# Instantiate the grid search model
grid_search_lr = GridSearchCV(estimator=lr,
                           param_grid=params_lr,
                           cv = 5,
                           n_jobs=-1, verbose=1, scoring="accuracy")
#%%
%%time
grid_search_lr.fit(X_train2, y_train)
#%%
grid_search_lr.best_score_, grid_search_lr.best_estimator_
#%%
lr_best = grid_search_lr.best_estimator_
print(lr_best)
 
lr_best.fit(X_train2, y_train)
#%%
X_test2, _ = transform_to_vector(X_test, dv=dv)
# Predict on the test set
y_pred = lr_best.predict(X_test2)
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
#%% md
# ## Naive Bayes
#%%
# Define the parameter grid
param_grid = {'var_smoothing':  [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]}
#%%
%%time
# Initialize the model
nb_model = GaussianNB()

# Perform Grid Search

grid_search = GridSearchCV(nb_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train2.toarray(), y_train)

# Get the best parameters and best score
best_alpha = grid_search.best_params_['var_smoothing']
best_score = grid_search.best_score_

print(f"Best alpha: {best_alpha}")
print(f"Best cross-validation accuracy: {best_score:.4f}")
#%%
print(grid_search.best_estimator_)
#%%
# Convert features to a dictionary format and vectorize
dicts_train = X_train.to_dict(orient='records')
dicts_test = X_test.to_dict(orient='records')

dv = DictVectorizer()
X_train_vectorized = dv.fit_transform(dicts_train)
X_test_vectorized = dv.transform(dicts_test)

# Initialize and train the Naive Bayes classifier
model = grid_search.best_estimator_
model.fit(X_train_vectorized.toarray(), y_train)

# Make predictions
y_pred = model.predict(X_test_vectorized.toarray())

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
#%% md
# From comparing the two models, we see that the LR model with accuracy of 0.79 is better than the NB model which has an accuracy of 0.77.
#%% md
# # Save best Model
# 
#%%
output_file = f'model.bin'
#%%
# Save the model

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, lr_best), f_out)

print(f'the model is saved to {output_file}')
#%%
