# ML Assignment 2 — Obesity Level Classification

## Problem Statement

This project predicts a person's obesity level based on their eating habits, physical activity, and other lifestyle factors. We use the Obesity Levels dataset from Kaggle and train five different ML classification models on it. The target has 7 classes ranging from Insufficient Weight to Obesity Type III. The goal is to compare model performance using standard classification metrics.

---

## Dataset Description

| Property | Value |
|---|---|
| Name | Obesity Levels Based on Eating Habits and Physical Activity |
| Source | [Kaggle — fatemehmehrparvar/obesity-levels](https://www.kaggle.com/datasets/fatemehmehrparvar/obesity-levels) |
| Task | Multi-Class Classification (7 classes) |
| Instances | 2,111 |
| Features | 16 (raw) → 20 (after encoding) |
| Target | NObeyesdad — obesity level category |
| Data origin | 23% real survey data (Mexico, Peru, Colombia); 77% SMOTE-synthetic |

**Target classes:** Insufficient_Weight, Normal_Weight, Overweight_Level_I, Overweight_Level_II, Obesity_Type_I, Obesity_Type_II, Obesity_Type_III

**Features:** Gender, Age, Height, Weight, family history of overweight, FAVC, FCVC, NCP, CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS

**Preprocessing:** Yes/no columns and gender are label-encoded. CAEC and CALC are ordinally encoded (no=0, Sometimes=1, Frequently=2, Always=3). MTRANS is one-hot encoded. StandardScaler is applied before training Logistic Regression and kNN.

---

## GitHub Repository Link

> **https://github.com/2025da04184/obesity-level-classification**


---

## Models Used

All models are trained on an 80/20 stratified split (1,688 train / 423 test). Precision, Recall, and F1 use weighted averaging. AUC is computed using one-vs-rest macro averaging.

### Comparison Table — Evaluation Metrics (Test Set)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8629 | 0.9821 | 0.8616 | 0.8629 | 0.8609 | 0.8403 |
| Decision Tree | 0.9102 | 0.9462 | 0.9121 | 0.9102 | 0.9107 | 0.8952 |
| kNN | 0.7967 | 0.9471 | 0.8006 | 0.7967 | 0.7900 | 0.7652 |
| Naive Bayes | 0.5603 | 0.8764 | 0.6110 | 0.5603 | 0.5159 | 0.5081 |
| Random Forest (Ensemble) | **0.9527** | **0.9967** | **0.9576** | **0.9527** | **0.9535** | **0.9454** |

### Model Observations

| ML Model Name | Observation about Model Performance |
|---|---|
| Logistic Regression | Decent performance (86.3% accuracy). Works reasonably well but struggles with classes that are close to each other like Overweight I and II since it is a linear model. |
| Decision Tree | Good performance (91% accuracy). The data has natural thresholds based on weight and BMI which a tree can pick up well. May overfit slightly on training data. |
| kNN | Lower performance (79.7% accuracy). With 7 closely related classes, finding clear distance boundaries is harder. Sensitive to the value of k and feature scaling. |
| Naive Bayes | Weakest model (56% accuracy). The independence assumption does not hold well here since features like Height and Weight are correlated with each other. |
| Random Forest (Ensemble) | Best model overall (95.3% accuracy, 0.9967 AUC). Using 100 trees reduces the overfitting seen in a single Decision Tree and handles the non-linear patterns in the data well. |

---

## Live Streamlit App

> ****


### App Features

- Upload test data (CSV)
- Select model from dropdown
- View evaluation metrics table
- View confusion matrix
- View per-class classification report

---

## Repository Structure

```
ml-assignment-2/
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── test_data.csv                   # Test split (20% of dataset, 423 rows)
├── ObesityDataSet_raw_and_data_sinthetic.csv   # Raw dataset (Kaggle)
└── model/
    ├── train_models.py             # Model training script
    ├── scaler.pkl                  # Fitted StandardScaler
    ├── label_encoder.pkl           # LabelEncoder for target classes
    ├── Logistic_Regression.pkl
    ├── Decision_Tree.pkl
    ├── kNN.pkl
    ├── Naive_Bayes.pkl
    ├── Random_Forest.pkl
    └── results.csv                 # Saved evaluation metrics
```

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload `test_data.csv` in the app to see predictions and metrics.
