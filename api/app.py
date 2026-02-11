from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('models/factoryguard_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(list(data.values())).reshape(1, -1)
    prob = model.predict_proba(features)[0][1]

    return jsonify({
        "failure_probability": float(prob),
        "risk_level": "HIGH" if prob > 0.7 else "LOW"
    })

if __name__ == "__main__":
    app.run(debug=True)
