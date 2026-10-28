# Customer_Churn-Prediction
The main objective of this project is to develop a machine learning model that predicts whether a customer is likely to churn. The project focuses on understanding customer behavior, identifying important factors associated with churn, and generating insights that can help businesses make data-driven decisions

## Overview
Predicts which e-commerce customers are likely to churn so retention campaigns can be targeted.

## Dataset
3,941 customer records (3,269 after removing 672 duplicates). Features: Tenure, WarehouseToHome, NumberOfDeviceRegistered, PreferedOrderCat, SatisfactionScore, MaritalStatus, NumberOfAddress, Complain, DaySinceLastOrder, CashbackAmount. Target: Churn (about 15% positive). Source: [add dataset link]

## Approach
1. Cleaning: median/mode imputation for 3 columns with missing values, duplicate removal
2. EDA: distributions and outlier checks
3. Preprocessing: 80/20 train/test split, MinMax scaling, one-hot encoding, VIF check (CashbackAmount dropped)
4. Models: Logistic Regression and XGBoost, tuned with RandomizedSearchCV (F1 scoring, 3-fold CV)
5. Imbalance handling: random undersampling, SMOTETomek, class weighting (scale_pos_weight)

## Results (test set, churn class)
| Model | Precision | Recall | F1 |
|---|---|---|---|
| Logistic Regression | 0.79 | 0.44 | 0.56 |
| Logistic Regression (tuned) | 0.70 | 0.53 | 0.61 |
| XGBoost (default) | 0.78 | 0.67 | 0.72 |
| XGBoost (tuned, class weighting) | 0.74 | 0.81 | 0.77 |
| XGBoost + undersampling | 0.42 | 0.93 | 0.58 |
| XGBoost + SMOTETomek | 0.64 | 0.80 | 0.71 |

## Key findings
- Accuracy is misleading here (a "no churn" model scores about 85%), so recall and F1 were the focus.
- Class weighting outperformed resampling on F1.
- Tuned XGBoost identifies about 81% of churners with 74% precision.

## Limitations
- Imputation was done before the train/test split
- Test set was also used for model comparison

## Tools
Python, pandas, NumPy, scikit-learn, XGBoost, imbalanced-learn, seaborn, matplotlib

Built a churn prediction model on 3,269 e-commerce customers using XGBoost, tuned with RandomizedSearchCV and class weighting. Achieved 0.81 recall and 0.77 F1 on the churn class (vs 0.56 F1 baseline), helping target retention offers to at-risk customers. Tools: Python, pandas, scikit-learn, XGBoost.

