# Titanic Survival Prediction
This repo shows the training and deployment of an ML model on the Titanic data.

The video that shows video interaction with the deployed model is available at:
https://bit.ly/deploy-docker

This project uses machine learning techniques to predict the survival of Titanic passengers based on their attributes. It demonstrates data preprocessing, model training, hyperparameter tuning, and evaluation with algorithms like Logistic Regression and Naive Bayes.

## Project Overview
The Titanic dataset is one of the most well-known data science and machine learning datasets. The objective is to build a predictive model to determine whether a passenger survived based on features like class, age, gender, and fare paid.

This project includes:

- Data preprocessing and feature engineering.
- Exploratory Data Analysis (EDA) with plots.
- Implementation of machine learning models such as:
    - Logistic Regression
    - Naive Bayes
- Hyperparameter tuning to improve model performance.
- Predicting survival for new passenger entries.
- Deployment using Docker

## Dependencies
You can convert the Pipfile and Pipfile.lock in the project directory into a requirements.txt.

```pipenv lock -r > requirements.txt```
After that, you can install all your modules in your Python virtual environment by doing the following:

```pip install -r requirements.txt```

Doing the above enables you to run the Jupyter notebook `notebook.ipynb`.

## Docker
To deploy and run the model with Docker:
```
>>> docker build -t predictordock_survival .
>>> docker run -it --rm -p 9696:9696 predictordock_survival
```
