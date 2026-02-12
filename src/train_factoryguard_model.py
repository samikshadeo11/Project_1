import pandas as pd
import numpy as np
import joblib
import os 
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

# =========================
# 1. Load Dataset
# =========================
DATA_PATH = "data/predictive_maintenance_dataset.csv"
df = pd.read_csv(DATA_PATH)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(['machine_id', 'timestamp'])

# Create folder automatically
os.makedirs("reports/figures", exist_ok=True)

# Plot class distribution - FIXED COLUMN NAME
df['label'].value_counts().plot(kind='bar')  # ← CHANGED 'failure' to 'label'
plt.title("Class Distribution")
plt.savefig("reports/figures/class_distribution.png")
plt.close()

# =========================
# 2. Feature Engineering - COMPLETE
# =========================
# Existing features
df['temp_roll_mean_6'] = df.groupby('machine_id')['temperature'].rolling(6).mean().reset_index(0, drop=True)
df['vibration_std_6'] = df.groupby('machine_id')['vibration'].rolling(6).std().reset_index(0, drop=True)
df['temp_lag_1'] = df.groupby('machine_id')['temperature'].shift(1)
df['vibration_lag_1'] = df.groupby('machine_id')['vibration'].shift(1)

# MISSING FEATURES - ADD THESE
df['vibration_roll_mean_6h'] = df.groupby('machine_id')['vibration'].rolling(6).mean().reset_index(0, drop=True)
df['temperature_std_6h'] = df.groupby('machine_id')['temperature'].rolling(6).std().reset_index(0, drop=True)
df['current_ema_12h'] = df.groupby('machine_id')['current'].ewm(span=12).mean().reset_index(0, drop=True)
df['vibration_lag_2'] = df.groupby('machine_id')['vibration'].shift(2)
df['acoustic_lag_1'] = df.groupby('machine_id')['acoustic'].shift(1)

df = df.dropna()

# =========================
# 3. Select Features & Target - FIXED
# =========================
FEATURES = [
    'vibration', 'acoustic', 'temperature', 'current',
    'IMF_1', 'IMF_2', 'IMF_3',
    'temp_roll_mean_6', 'vibration_std_6', 
    'temp_lag_1', 'vibration_lag_1',
    'vibration_roll_mean_6h', 'temperature_std_6h',
    'current_ema_12h', 'vibration_lag_2', 'acoustic_lag_1'
]
TARGET = 'label'  # ← FIXED

X = df[FEATURES]
y = df[TARGET]

# ========================= 4-7 SAME AS BEFORE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, 
                     scale_pos_weight=20, eval_metric='aucpr', random_state=42)
model.fit(X_train, y_train)

from sklearn.metrics import PrecisionRecallDisplay
PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
plt.title("FactoryGuard AI – Precision Recall Curve")
plt.savefig("reports/figures/pr_curve.png")
plt.close()

importances = model.feature_importances_
pd.Series(importances, index=X.columns).sort_values().plot(kind='barh')
plt.title("Feature Importance")
plt.savefig("reports/figures/feature_importance.png")
plt.close()

y_pred_prob = model.predict_proba(X_test)[:, 1]
pr_auc = average_precision_score(y_test, y_pred_prob)
print(f"PR-AUC Score: {pr_auc:.4f}")

joblib.dump(model, "models/factoryguard_model.pkl")
print("✅ Model saved as models/factoryguard.pkl")
