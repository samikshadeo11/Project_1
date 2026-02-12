import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os

# Load trained model
model = joblib.load("models/factoryguard_model.pkl")

# Load dataset
df = pd.read_csv("data/predictive_maintenance_dataset.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(['machine_id', 'timestamp'])

# === SAME FEATURE ENGINEERING AS TRAINING ===
df['temp_roll_mean_6'] = df.groupby('machine_id')['temperature'].rolling(6).mean().reset_index(0, drop=True)
df['vibration_std_6'] = df.groupby('machine_id')['vibration'].rolling(6).std().reset_index(0, drop=True)
df['temp_lag_1'] = df.groupby('machine_id')['temperature'].shift(1)
df['vibration_lag_1'] = df.groupby('machine_id')['vibration'].shift(1)
df['vibration_roll_mean_6h'] = df.groupby('machine_id')['vibration'].rolling(6).mean().reset_index(0, drop=True)
df['temperature_std_6h'] = df.groupby('machine_id')['temperature'].rolling(6).std().reset_index(0, drop=True)
df['current_ema_12h'] = df.groupby('machine_id')['current'].ewm(span=12).mean().reset_index(0, drop=True)
df['vibration_lag_2'] = df.groupby('machine_id')['vibration'].shift(2)
df['acoustic_lag_1'] = df.groupby('machine_id')['acoustic'].shift(1)

df = df.dropna()

# Features MUST match training
FEATURES = [
    'vibration', 'acoustic', 'temperature', 'current',
    'IMF_1', 'IMF_2', 'IMF_3',
    'temp_roll_mean_6', 'vibration_std_6',
    'temp_lag_1', 'vibration_lag_1',
    'vibration_roll_mean_6h', 'temperature_std_6h',
    'current_ema_12h', 'vibration_lag_2', 'acoustic_lag_1'
]

X = df[FEATURES]

# Sample
X_sample = X.sample(500, random_state=42)

# SHAP
explainer = shap.Explainer(model, X_sample)
shap_values = explainer(X_sample)

os.makedirs("reports/figures", exist_ok=True)

shap.summary_plot(shap_values, X_sample, show=False)
plt.savefig("reports/figures/shap_summary.png")
plt.close()

print("✅ SHAP explainability graph saved successfully")
