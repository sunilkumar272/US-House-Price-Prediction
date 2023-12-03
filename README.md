# US-House-Price-Prediction

## Description:
This project focuses on predicting house prices using various regression techniques including linear regression, Lasso regression, and decision tree. A crucial aspect of this project is the data preprocessing phase, which involves the removal of outliers and the utilization of one-hot encoding to handle categorical variables.

## Dataset:
The dataset is downloaded from Kaggle and comprises more than 600,000 samples with the attributes status, price, bed, bath, acre_lot, full_address, street, city, state, zip_code, house_size and sold_date.
States with over 1,000 data entries are considered for the project and remaining states are categorized as other.
State categories are Connecticut, Maine, Massachusetts, New Hampshire, New Jersey, New York, Puerto Rico, Rhode Island, Vermont and other.

Link to the Dataset: https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset/data

## Results:

The grid search cross-validation was employed to determine the best parameters for each model. The decision tree model outperformed linear and Lasso regression due to its ability to capture complex non-linear relationships within the data. Decision trees excel in learning complex patterns, especially in scenarios where linear models might struggle. This flexibility allows the decision tree to capture interactions and nonlinearities present in the dataset, leading to a more accurate prediction of house prices.
