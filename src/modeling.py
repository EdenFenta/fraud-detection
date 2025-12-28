import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    auc
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE

df = pd.read_csv("data/processed/fraud_features.csv")

print("Dataset shape:", df.shape)
print(df['class'].value_counts(normalize=True))

FEATURES = [
    'purchase_value',
    'time_since_signup',
    'hour_of_day',
    'day_of_week',
    'age'
]

X = df[FEATURES]
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Train fraud ratio:", y_train.mean())
print("Test fraud ratio:", y_test.mean())

smote = SMOTE(random_state=42)

X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:", y_train_res.value_counts())

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)

log_reg.fit(X_train_scaled, y_train_res)

y_pred_lr = log_reg.predict(X_test_scaled)
y_prob_lr = log_reg.predict_proba(X_test_scaled)[:, 1]

print("Logistic Regression Results")
print(classification_report(y_test, y_pred_lr))
print("F1 Score:", f1_score(y_test, y_pred_lr))

precision, recall, _ = precision_recall_curve(y_test, y_prob_lr)
pr_auc_lr = auc(recall, precision)

print("Logistic Regression PR-AUC:", pr_auc_lr)

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)

rf.fit(X_train_res, y_train_res)

y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

print("Random Forest Results")
print(classification_report(y_test, y_pred_rf))
print("F1 Score:", f1_score(y_test, y_pred_rf))

precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_prob_rf)
pr_auc_rf = auc(recall_rf, precision_rf)

print("Random Forest PR-AUC:", pr_auc_rf)

print("\nMODEL COMPARISON")
print(f"Logistic Regression - PR AUC: {pr_auc_lr:.4f}")
print(f"Random Forest      - PR AUC: {pr_auc_rf:.4f}")
