# Startup-Success-Fail-Dataset-from-Crunchbase


##  Dataset Info

- Source: [Kaggle - Crunchbase Investments](https://www.kaggle.com/datasets/yanmaksi/big-startup-secsees-fail-dataset-from-crunchbase/data)
- Format: CSV
- Fields: company name, status, country, category, funding rounds, total funding, dates, etc.

##  Data & Code Management

- Code tracked on GitHub and backed up on OneDrive
- Sensitive variables managed using `.env` files (not committed)
- Processed data stored securely

##  GitHub Repository

GitHub: (https://github.com/AshishKhandekar99/Startup-Success-Fail-Dataset-from-Crunchbase)

#  Predicting Startup Success Using Crunchbase Data

This project uses machine learning techniques to predict whether a startup will succeed or fail based on Crunchbase data. It is a full-cycle data science project involving data preprocessing, exploratory data analysis, model building, evaluation, and optional deployment.

##  Project Overview

**Objective:** Build a classification model that predicts the success of a startup using attributes such as funding details, location, sector, and more.

**Dataset:** Crunchbase startup dataset (via Kaggle) - Contains ~66,000 startup records with metadata including status, funding rounds, and sector.

##  Workflow

### 1. Data Cleaning & Preprocessing
- Remove duplicates and irrelevant fields
- Handle missing values
- Convert dates and calculate startup age
- Encode categorical variables

### 2. Exploratory Data Analysis (EDA)
- Distribution of success/failure
- Analysis of funding, sectors, and countries
- Correlation analysis

### 3. Feature Engineering
- Time to first funding
- Funding round frequency
- Startup age
- Group rare sectors

### 4. Model Building
- **Logistic Regression**
- **Random Forest**
- **XGBoost**
- Cross-validation & hyperparameter tuning

### 5. Model Evaluation
- Accuracy, Precision, Recall, F1 Score
- ROC-AUC Score
- Confusion Matrix
- Feature importance using SHAP

### 6. Bonus: Web App Deployment
- Deploy prediction model using **Streamlit** (optional)
- Users input startup details and get predicted success probability

##  References

- Ravuri & Olsen (2017) – Startup Success Prediction Using Decision Trees and Random Forests
- Singh et al. (2020) – Startup Success Prediction: A Machine Learning Approach
- Guzman & Stern (2016) – The State of American Entrepreneurship
- Rizwan et al. (2019) – Survival and Success Prediction of Startups
- Lundberg & Lee (2017) – A Unified Approach to Interpreting Model Predictions (SHAP)
