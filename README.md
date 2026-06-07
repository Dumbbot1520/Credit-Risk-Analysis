# Credit Risk Prediction 
Link - https://your-credit-risk-analyzer.streamlit.app/
A machine learning project to predict credit card default risk using real-world banking data.

## Problem Statement
Banks need to identify customers likely to default on credit card payments before it happens. Missing a defaulter (false negative) is far more costly than wrongly flagging a good customer. This project builds a pipeline optimised for **Recall** — minimising missed defaulters.

## Dataset
- **Source**: UCI Machine Learning Repository — Default of Credit Card Clients
- **Size**: 30,000 records, 25 features
- **Target**: `default.payment.next.month` (0 = repaid, 1 = defaulted)
- **Class Distribution**: 78% non-defaulters, 22% defaulters

## Project Pipeline

### 1. Data Cleaning
- Identified hidden dirty data using `value_counts()` — isnull() alone was insufficient
- EDUCATION: replaced undocumented values (0, 5, 6) with 4 (Others)
- MARRIAGE: replaced undocumented value (0) with 3 (Others)

### 2. Exploratory Data Analysis
- Higher credit limits correlate strongly with lower default rates
- Customers with high school education default more frequently
- PAY columns show strongest correlation with default — non-defaulters averaged -1.98 total delay vs +2.02 for defaulters

### 3. Feature Engineering
Created 3 new features from existing columns:

| Feature | Formula | Purpose |
|---|---|---|
| PAY_RATIO1–6 | PAY_AMT / (BILL_AMT + 1) | Monthly payment ratio |
| AVG_PAY_RATIO | Mean of 6 ratios | Overall payment behaviour |
| TOTAL_DELAY | Sum of PAY_0 to PAY_6 | Cumulative payment delay |

### 4. Class Imbalance — SMOTE
- Applied SMOTE **only on training data** to preserve real-world distribution in test set
- Before: 18,691 vs 5,309 | After: 18,691 vs 18,691

### 5. Model Comparison

| Model | Recall (class 1) | ROC-AUC |
|---|---|---|
| Logistic Regression | 0.58 | 0.6462 |
| Random Forest | 0.48 | 0.6611 |
| XGBoost | 0.49 | 0.6611 |
| Tuned Random Forest | 0.70 | 0.7165 |

### 6. Threshold Tuning
- Default threshold 0.5 → Recall 0.58
- Lowered threshold to 0.3 → **Recall 0.80**
- Conscious tradeoff: higher false positives accepted to minimise missed defaulters

### 7. SHAP Explainability
Top features by importance:
1. TOTAL_DELAY — strongest predictor
2. PAY_0 — most recent payment status
3. LIMIT_BAL — credit limit
4. AVG_PAY_RATIO — payment consistency

### 8. Hyperparameter Tuning
- Used RandomizedSearchCV with cv=3, n_iter=20
- Optimised for Recall as primary metric
- Best params: n_estimators=100, max_depth=None, min_samples_split=2

## Final Model
**Logistic Regression at threshold 0.3**
- Recall: 0.80
- ROC-AUC: 0.6462
- Selected over higher ROC-AUC models because Recall is the business priority

## Streamlit Web App
Live inference app built with Streamlit — enter customer details and get instant default probability prediction.
python -m streamlit run app.py

## Tech Stack
- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- Imbalanced-learn (SMOTE)
- SHAP
- Streamlit
- Pickle

## Key Learnings
- Accuracy is a misleading metric for imbalanced datasets
- Threshold tuning is a powerful post-training technique
- SHAP reveals that engineered features (TOTAL_DELAY) outperformed raw features in importance
- Saving both model AND scaler is critical for correct inference
