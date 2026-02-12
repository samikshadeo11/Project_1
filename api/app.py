from flask import Flask, request, render_template_string
import joblib
import numpy as np
import os

app = Flask(__name__)

model_path = os.path.join("models", "factoryguard_model.pkl")
model = joblib.load(model_path)


@app.route('/')
def home():
    return render_template_string("""
        <h2>FactoryGuard AI - Predict Failure</h2>
        <form action="/predict_browser" method="post">
            Vibration: <input name="vibration"><br>
            Acoustic: <input name="acoustic"><br>
            Temperature: <input name="temperature"><br>
            Current: <input name="current"><br>
            IMF_1: <input name="IMF_1"><br>
            IMF_2: <input name="IMF_2"><br>
            IMF_3: <input name="IMF_3"><br>
            temp_roll_mean_6: <input name="temp_roll_mean_6"><br>
            vibration_std_6: <input name="vibration_std_6"><br>
            temp_lag_1: <input name="temp_lag_1"><br>
            vibration_lag_1: <input name="vibration_lag_1"><br>
            vibration_roll_mean_6h: <input name="vibration_roll_mean_6h"><br>
            temperature_std_6h: <input name="temperature_std_6h"><br>
            current_ema_12h: <input name="current_ema_12h"><br>
            vibration_lag_2: <input name="vibration_lag_2"><br>
            acoustic_lag_1: <input name="acoustic_lag_1"><br><br>
            <button type="submit">Predict</button>
        </form>
    """)


@app.route('/predict_browser', methods=['POST'])
def predict_browser():
    values = [float(v) for v in request.form.values()]
    features = np.array(values).reshape(1, -1)
    prob = model.predict_proba(features)[0][1]

    return f"""
        <h3>Failure Probability: {prob:.4f}</h3>
        <h3>Risk Level: {'HIGH' if prob > 0.7 else 'LOW'}</h3>
        <a href="/">Go Back</a>
    """


if __name__ == '__main__':
    app.run(debug=True)
