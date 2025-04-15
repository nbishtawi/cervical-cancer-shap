# Cervical Cancer Biopsy Prediction with XGBoost and SHAP

## Overview

This project aims to predict the likelihood of a cervical cancer biopsy using clinical, behavioral, and demographic data. We use an XGBoost classifier with SMOTE for class imbalance handling and SHAP for model interpretability. The dataset comes from the UCI Machine Learning Repository.

---

## Dataset

- **Source**: [UCI Cervical Cancer Risk Factors](https://archive.ics.uci.edu/ml/datasets/Cervical+Cancer+%28Risk+Factors%29)
- **Target Variable**: `Biopsy` (binary: 1 = Biopsy performed, 0 = Not performed)
- **Features**: Patient-reported behaviors, contraceptive use, STD history, and prior screenings

---

## Methods

- **Model**: XGBoost Classifier (`xgboost`)
- **Preprocessing**:
  - Excluded highly sparse or outcome-related columns (`STDs: Time since first diagnosis`, `Hinselmann`, etc.)
  - Imputed missing values using:
    - **Mode** for binary features (e.g., STDs, screenings)
    - **Mean** for continuous features (e.g., number of pregnancies)
- **Class Imbalance Handling**:  
  - Applied `SMOTE` to oversample the minority class to a 0.7 ratio
- **Train-Test Split**: 70-30 shuffle

---

## Outputs

- **Classification Metrics**: Accuracy, Precision, Recall, F1 Score, AUC
- **SHAP Summary Plot** (Beeswarm): Shows the global importance of features  
- **SHAP Waterfall Plot**: Shows how individual features influenced a single prediction

Plots are saved to the `figures/` folder:
- `figures/shap_beeswarm.png`
- `figures/shap_waterfall_sample.png`


---

## 🔒 Data Privacy Note

No sensitive data is included in this repository. The dataset is open and publicly available from the UCI repository.

---

## 👤 Author

**Nadeem Bishtawi**  
