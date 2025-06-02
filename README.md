# 🏡 House Prices - Advanced Regression Techniques (STA141C Final Project)

This repository contains my work for UC Davis STA141C final project, where we applied and compared advanced regression models on the Kaggle "House Prices: Advanced Regression Techniques" dataset. The goal was to predict housing sale prices based on 81 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa.

## 📊 Project Overview

- **Dataset**: [Kaggle House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
- **Problem Type**: Supervised Regression
- **Target**: `SalePrice` (log-transformed)
- **Metric**: Root Mean Squared Error (RMSE) of log-transformed SalePrice

## 🛠️ Key Components

### 🔹 Data Cleaning & Preprocessing
- Handled missing values using median/mode and conditional logic
- Converted ordinal features to numerical scale
- One-hot encoded and mean target encoded nominal categorical variables
- Corrected skewed numeric features using log transformation
- Scaled numerical features using StandardScaler

### 🔹 Dimensionality Reduction
- Applied **Principal Component Analysis (PCA)** and **Partial Least Squares (PLS)**
- Evaluated how reduced-dimensional models performed vs. full feature models

### 🔹 Model Training & Evaluation
- Compared performance of:
  - Linear Regression, Lasso, Ridge
  - Random Forest
  - PCA/PLS with linear models
- Used 5-fold cross-validation with RMSE scoring
- Final stacked ensemble achieved **___** on Kaggle  leaderboard

## 🧪 Tech Stack
- Python, Jupyter
- `pandas`, `numpy`, `scikit-learn`, `seaborn`, `matplotlib`

