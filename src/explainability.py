import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 1. Load trained model
model = joblib.load("models/factoryguard_model.pkl")

# 2. Load dataset
df = pd.read_csv("data/dataset.csv")

# 3. Define features (MUST match training)
FEATURES = ['vibration', 'acoustic', 'temperature', 'current',
    'IMF_1', 'IMF_2', 'IMF_3',
    'vibration_roll_mean_1h', 'vibration_roll_mean_6h', 'vibration_roll_mean_12h',
    'temperature_std_6h', 'current_ema_12h',
    'vibration_lag_1', 'vibration_lag_2', 'acoustic_lag_1']

X = df[FEATURES]

# 4. Take a sample for SHAP (faster)
X_sample = X.sample(500, random_state=42)

# 5. SHAP explainer
explainer = shap.Explainer(model, X_sample)
shap_values = explainer(X_sample)

# 6. SHAP summary plot
shap.summary_plot(shap_values, X_sample, show=False)
plt.savefig("reports/figures/shap_summary.png")
plt.close()

print("✅ SHAP explainability graph saved successfully")

