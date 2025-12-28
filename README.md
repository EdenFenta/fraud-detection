# Fraud Detection Project

## Overview
This project focuses on detecting fraudulent e-commerce transactions using machine learning. The objective is to improve fraud detection accuracy while balancing security and customer experience.

## Datasets
- Fraud_Data.csv: E-commerce transaction data
- IpAddress_to_Country.csv: IP address to country mapping for geolocation analysis

## Data Analysis
Exploratory data analysis revealed:
- Severe class imbalance, with fraudulent transactions representing a very small percentage of total transactions
- Fraudulent transactions tend to have higher purchase values
- Time-related patterns are important indicators of fraud

## Feature Engineering
The following features were engineered:
- Time since signup (hours)
- Hour of day
- Day of week
- IP address converted to integer for geolocation analysis

## Class Imbalance
Due to extreme class imbalance, accuracy is not an appropriate evaluation metric. Future modeling will rely on F1-score and Precision-Recall AUC, with resampling techniques applied only to training data.

## Modeling and Evaluation
Two models were trained:
- Logistic Regression as an interpretable baseline
- Random Forest as an ensemble model

Due to severe class imbalance, models were evaluated using F1-score and Precision-Recall AUC. Random Forest achieved superior performance, capturing non-linear fraud patterns more effectively.
