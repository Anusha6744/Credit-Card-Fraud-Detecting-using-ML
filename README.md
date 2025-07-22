## Credit Card Fraud Detection using Machine Learning

This project aims to detect fraudulent credit card transactions using supervised machine learning algorithms. The dataset is highly imbalanced, so special care was taken to handle the imbalance using techniques like class weighting and appropriate evaluation metrics.

##  Dataset

- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions
- **Features:** 30 anonymized features (`V1` to `V28`), `Time`, `Amount`, and `Class`
- **Target:** `Class`  
  - `0`: Legitimate Transaction  
  - `1`: Fraudulent Transaction

##  Problem Statement

The objective is to identify fraudulent transactions from a large volume of credit card activity using:
- Logistic Regression
- Random Forest (with hyperparameter tuning)
- XGBoost (with GridSearchCV)

## Project Workflow

1. **Data Loading and Exploration**
   - Checked class distribution and percentage of fraud cases.
   - Visualized imbalance and correlations.

2. **Preprocessing**
   - Scaled the `Amount` feature using `StandardScaler`
   - Dropped the `Time` feature as it adds minimal value.
   - Stratified train-test split to maintain class distribution.

3. **Modeling**
   - Applied Logistic Regression with `class_weight='balanced'`
   - Trained Random Forest using GridSearchCV
   - Tuned XGBoost with `scale_pos_weight` and hyperparameters

4. **Evaluation Metrics**
   - Classification Report (Precision, Recall, F1-score)
   - Confusion Matrix
   - ROC-AUC Curve Comparison

## Results

| Model               | Precision (Class 1) | Recall (Class 1) | F1-score (Class 1) |
|--------------------|---------------------|------------------|--------------------|
| Logistic Regression| 0.06                | 0.92             | 0.11               |
| Random Forest      | 0.87                | 0.67             | 0.76               |
| XGBoost            | 0.88                | 0.84             | 0.86               |

>  XGBoost achieved the best balance between precision and recall.

##  Visualizations

- **Class Distribution Plot**
- **Correlation Heatmap**
- **Confusion Matrices**
- **ROC Curves with AUC Scores**

##  Libraries Used

- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `sklearn`
- `xgboost`

## Key Insights

- Class imbalance is extreme: Fraud cases ≈ 0.17%
- **Logistic Regression** has high recall but very low precision (over-predicts fraud)
- **Random Forest** balances precision and recall well
- **XGBoost** achieved the best **F1-score (0.86)** and **AUC**
- Evaluation based only on accuracy is misleading in imbalanced datasets
- ROC Curve and F1-score for Class 1 are essential for model comparison

  ## Author
   Mail id : aajayan525@gmail.com 🔗(https://github.com/Anusha6744)


## License
This project is for educational use.

## ⭐️ Show Support
If you liked this project, consider ⭐️ starring the repository or sharing it with others!
