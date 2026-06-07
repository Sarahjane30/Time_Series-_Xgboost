# Retail Product Demand Forecasting using XGBoost

## Overview

This project predicts retail product demand using machine learning and time-series feature engineering. Historical sales data from multiple stores and product categories was analyzed to forecast future sales and improve inventory planning.

## Dataset

Store Sales Time Series Forecasting Dataset

* Total Records: 3,000,888
* Multiple Stores
* Multiple Product Categories
* Sales Data
* Promotion Information
* Store Metadata

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost

## Feature Engineering

The following features were created:

* Year
* Month
* Day
* Day of Week
* Store Information
* Product Family
* Promotion Status
* Lag 7 Feature
* Lag 30 Feature
* Rolling Mean (7 Days)

## Models Implemented

### Linear Regression

RMSE: 1088.51

R² Score: 0.361

### Random Forest Regressor

RMSE: 444.59

R² Score: 0.893

### XGBoost Regressor

RMSE: 423.31

R² Score: 0.939

## Results

XGBoost achieved the best performance, reducing prediction error significantly compared to Linear Regression and Random Forest.

| Model             | RMSE    | R² Score |
| ----------------- | ------- | -------- |
| Linear Regression | 1088.51 | 0.361    |
| Random Forest     | 444.59  | 0.893    |
| XGBoost           | 423.31  | 0.939    |

## Key Learnings

* Time-Series Forecasting
* Feature Engineering
* Lag Features
* Rolling Averages
* Ensemble Learning
* Gradient Boosting
* Model Evaluation

## Future Improvements

* Holiday Feature Integration
* Transaction-Based Features
* Hyperparameter Tuning
* SHAP Explainability
* Forecast Visualization
