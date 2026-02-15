
# Machine Learning Assignment 2

## Problem Statement
In this assignment, I implemented multiple machine learning classification models to predict bank customer churn and deployed them using Streamlit for interactive testing.

## Dataset Description
The dataset contains customer banking information such as credit score, age, balance, geography, gender, and activity status.

Target column: Exited  
1 = customer left the bank  
0 = customer stayed  

The dataset contains more than 500 records and more than 12 features after preprocessing.

## Model Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9985 | 0.9992 | 0.9975 | 0.9951 | 0.9963 | 0.9954 |
| Decision Tree | 0.9975 | 0.9960 | 0.9951 | 0.9926 | 0.9939 | 0.9923 |
| KNN | 0.9920 | 0.9975 | 0.9975 | 0.9632 | 0.9801 | 0.9753 |
| Naive Bayes | 0.7975 | 0.8133 | 0.5205 | 0.0931 | 0.1580 | 0.1529 |
| Random Forest | 0.9985 | 0.9989 | 0.9975 | 0.9951 | 0.9963 | 0.9954 |
| XGBoost | 0.9985 | 0.9972 | 0.9975 | 0.9951 | 0.9963 | 0.9954 |

## Observations

Logistic Regression performed very well and produced stable results.

Decision Tree also gave high accuracy but may slightly overfit.

KNN worked well but recall is slightly lower than other models.

Naive Bayes performed poorly because feature independence assumption does not fit this dataset.

Random Forest achieved one of the best performances.

XGBoost performed similarly to Random Forest with very strong results.

## How to Run

1. Install dependencies:
pip install -r requirements.txt

2. Train models:
python train.py

3. Run Streamlit app:
streamlit run app.py
