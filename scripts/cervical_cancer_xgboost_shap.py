# cervical_cancer_xgboost_shap.py

# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import shap
import random

# Load dataset
df = pd.read_csv("risk_factors_cervical_cancer.csv")

# Drop columns with too many missing values
df.replace("?", pd.NA, inplace=True)

# Fill missing values
df = df.apply(pd.to_numeric, errors='coerce')

# Exclude irrelevant or unwanted columns
columns_to_exclude = [
    "STDs: Time since first diagnosis",
    "STDs: Time since last diagnosis",
    "Hinselmann",
    "Schiller",
    "Citology"
]
df_cleaned = df.drop(columns=columns_to_exclude)

# Identify boolean and continuous features for imputation
boolean_features_all = [
    'Smokes',
    'Hormonal Contraceptives',
    'IUD',
    'STDs',
    'STDs:condylomatosis',
    'STDs:vaginal condylomatosis',
    'STDs:vulvo-perineal condylomatosis',
    'STDs:syphilis',
    'STDs:pelvic inflammatory disease',
    'STDs:genital herpes',
    'STDs:molluscum contagiosum',
    'STDs:HIV',
    'STDs:Hepatitis B',
    'STDs:HPV',
    'Dx:Cancer',
    'Dx:CIN',
    'Dx:HPV',
    'Dx',
    'Biopsy'
]

boolean_to_impute = [col for col in boolean_features_all if col in df_cleaned.columns]

# Continuous features are all other numeric ones
continuous_to_impute = [
    col for col in df_cleaned.columns
    if col not in boolean_to_impute and df_cleaned[col].dtype in ['float64', 'int64']
]

# Impute missing values
for col in boolean_to_impute:
    mode = df_cleaned[col].mode(dropna=True)
    if not mode.empty:
        df_cleaned[col] = df_cleaned[col].fillna(mode[0])

for col in continuous_to_impute:
    mean = df_cleaned[col].mean(skipna=True)
    df_cleaned[col] = df_cleaned[col].fillna(mean)

# Split into features and target
X = df_cleaned.drop("Biopsy", axis=1)
y = df_cleaned["Biopsy"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=True
)

# Address class imbalance with SMOTE
smote = SMOTE(sampling_strategy=0.7)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train mode
model = XGBClassifier()
model.fit(X_train_resampled, y_train_resampled)

# Model Evaluation
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_proba))

# SHAP Explainability
explainer = shap.Explainer(model, X_train_resampled)
shap_values = explainer(X_test)

# Save SHAP Plots
os.makedirs("figures", exist_ok=True)

# Sample waterfall plot
sample_idx = random.randint(0, len(X_test) - 1)
shap.plots.waterfall(shap_values[sample_idx])
plt.title("SHAP Waterfall Plot")
plt.savefig("figures/shap_waterfall_sample.png", bbox_inches="tight")
plt.close()

# SHAP beeswarm plot
shap.plots.beeswarm(shap_values)
plt.title("SHAP Beeswarm Plot")
plt.savefig("figures/shap_beeswarm.png", bbox_inches="tight")
plt.close()
