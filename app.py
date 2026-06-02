"""
Credit Card Fraud Detection — Flask Web Application
====================================================
A premium web dashboard for real-time fraud prediction.

Usage:
    1. First train the model:  python train_model.py
    2. Then start the app:     python app.py
    3. Open: http://localhost:5000
"""

import os
import json
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ── Load model artifacts ───────────────────────────────────────
MODEL = None
SCALER = None
METADATA = None


def load_artifacts():
    global MODEL, SCALER, METADATA
    model_path = os.path.join("models", "best_model.pkl")
    scaler_path = os.path.join("models", "scaler.pkl")
    meta_path = os.path.join("models", "metadata.json")

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        MODEL = joblib.load(model_path)
        SCALER = joblib.load(scaler_path)
        print("[OK] Model and scaler loaded successfully.")
    else:
        print("[!!] No trained model found. Run 'python train_model.py' first.")

    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            METADATA = json.load(f)
        print("[OK] Metadata loaded.")


# -- Routes -----------------------------------------------------
@app.route("/")
def index():
    model_loaded = MODEL is not None
    metadata = METADATA if METADATA else {}
    return render_template("index.html",
                           model_loaded=model_loaded,
                           metadata=metadata)


@app.route("/predict", methods=["POST"])
def predict():
    if MODEL is None:
        return jsonify({"error": "Model not loaded. Train it first."}), 503

    try:
        data = request.get_json()
        features = data.get("features", [])

        if len(features) != len(METADATA["feature_names"]):
            return jsonify({
                "error": f"Expected {len(METADATA['feature_names'])} features, "
                         f"got {len(features)}"
            }), 400

        X = np.array(features, dtype=float).reshape(1, -1)
        X_scaled = SCALER.transform(X)

        prediction = int(MODEL.predict(X_scaled)[0])
        probability = float(MODEL.predict_proba(X_scaled)[0][1])

        return jsonify({
            "prediction": prediction,
            "probability": round(probability, 6),
            "is_fraud": prediction == 1,
            "confidence": round(max(probability, 1 - probability) * 100, 2),
            "label": "🚨 FRAUD DETECTED" if prediction == 1 else "✅ LEGITIMATE"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/metadata")
def api_metadata():
    if METADATA is None:
        return jsonify({"error": "No metadata available"}), 404
    return jsonify(METADATA)


@app.route("/api/sample")
def api_sample():
    """Return a sample transaction for testing."""
    # Typical legitimate transaction (approximate values)
    legit = [
        -1.36, -0.07, 2.54, 1.38, -0.34, 0.46, 0.24, 0.10, 0.36,
        -0.09, -0.55, -0.62, -0.99, -0.31, 0.17, -0.47, 0.21, 0.03,
        -0.02, -0.02, 0.28, -0.11, 0.07, 0.13, -0.19, 0.13, -0.02,
        np.log1p(149.62),  # log_Amount
        0,  # Hour
    ]
    # Typical fraud transaction
    fraud = [
        -2.31, 1.95, -1.61, 0.87, -0.21, 0.50, -0.22, 0.08, -0.19,
        -0.26, -1.32, 0.69, 1.09, -0.01, 0.48, 0.17, 0.46, -0.11,
        -0.58, -0.02, -0.14, -0.22, 0.06, -0.20, -0.01, 0.25, 0.04,
        np.log1p(1.00),  # log_Amount
        3,  # Hour
    ]
    return jsonify({
        "legitimate": legit,
        "fraud": fraud,
        "feature_names": METADATA["feature_names"] if METADATA else []
    })


# ── Main ───────────────────────────────────────────────────────
if __name__ == "__main__":
    load_artifacts()
    print("\n  Starting Credit Card Fraud Detection Web App...")
    print("  Open: http://localhost:5000\n")
    app.run(debug=True, port=5000)
