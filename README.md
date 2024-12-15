# Titanic Classification Project

This project involves predicting the survival of passengers aboard the Titanic using machine learning. The dataset consists of various features such as age, sex, class, and embarkation point, which are used to train a machine learning model that predicts whether a passenger survived or not.

## Overview

In this project, I utilized a Random Forest Classifier to classify Titanic passengers into survivors and non-survivors. The dataset was preprocessed to handle missing values and categorical features using custom transformers. Feature engineering, imputation, and scaling were also applied to enhance the model's performance.

## Key Steps

1. **Data Exploration**: 
   - Loaded the dataset and performed basic analysis (descriptive statistics, correlation heatmap).
   - Split the data into training and test sets using `StratifiedShuffleSplit` to ensure a consistent distribution of the target variable.

2. **Data Preprocessing**:
   - Handled missing values in the `Age` column using median imputation.
   - Categorical variables (`Embarked`, `Sex`) were encoded using one-hot encoding.
   - Dropped unnecessary features like `Name`, `Ticket`, `Cabin`, and `Sex` after encoding.

3. **Model Building**:
   - Used a `RandomForestClassifier` to build the model, applying hyperparameter tuning with `GridSearchCV`.
   - Split the data into features (X) and target (y), and standardized the features using `StandardScaler`.

4. **Model Evaluation**:
   - The final model achieved a classification accuracy score of **0.78947** on the test set.

5. **Prediction**:
   - The trained model was used to make predictions on the Titanic test data, and the results were saved in a CSV file for submission.

## Requirements

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

## Results

The final model achieved an accuracy score of **0.78947** on the Titanic test data.
