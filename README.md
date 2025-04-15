# Cervical Cancer Biopsy Prediction with XGBoost and SHAP

## 📋 Overview

This project predicts the likelihood of a cervical cancer biopsy using patient features from the UCI Cervical Cancer Risk Factors dataset. The focus is on handling class imbalance using SMOTE and interpreting model decisions with SHAP.

---

## 🔬 Dataset

- **Source**: [UCI Machine Learning Repository – Cervical Cancer Risk Factors](https://archive.ics.uci.edu/ml/datasets/Cervical+Cancer+%28Risk+Factors%29)
- **Target variable**: `Biopsy` (binary: 1 = Biopsy performed, 0 = not performed)
- **Features**: Demographics, behavioral, and clinical screening test results

---

## ⚙️ Methods

- **Model**: XGBoost Classifier
- **Preprocessing**:
  - Missing value imputation
  - Label encoding for binary variables
  - Train-test split (stratified)
- **Class imbalance handling**: SMOTE (Synthetic Minority Oversampling Technique)
- **Model interpretability**: SHAP (SHapley Additive exPlanations)

---

## 📈 Outputs

- Model accuracy and F1-score
- SHAP summary plot for feature importance

All analysis is in [`scripts/cervical_cancer_xgboost_shap.py`](scripts/cervical_cancer_xgboost_shap.py).

---

## 🔒 Data Privacy Note

No sensitive data is included in this repository. The dataset is open and publicly available from the UCI repository.

---

## 👤 Author

**Nadeem Bishtawi**  
