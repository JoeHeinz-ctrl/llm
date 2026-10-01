"""
Flask Machine Learning Application
Uses Flask, joblib, jsonify, and request to serve predictions and model retraining.
"""

import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from train import train_and_save_model, MODEL_PATH

app = Flask(__name__)

# Global model state
model_package = None

def load_model():
    """Load model artifact from disk or train if not present."""
    global model_package
    if not os.path.exists(MODEL_PATH):
        print("[INFO] Model file not found. Training default model...")
        model_package = train_and_save_model(MODEL_PATH)
    else:
        try:
            model_package = joblib.load(MODEL_PATH)
            print("[INFO] Loaded existing model artifact from disk.")
        except Exception as e:
            print(f"[WARN] Error loading model: {e}. Retraining...")
            model_package = train_and_save_model(MODEL_PATH)
    return model_package

# Initial load
load_model()

@app.route("/")
def index():
    """Serve the modern interactive web interface."""
    return render_template("index.html")

@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model_package is not None
    }), 200

@app.route("/api/info", methods=["GET"])
def get_info():
    """Return model metadata, feature names, targets, and test accuracy."""
    if not model_package:
        return jsonify({"error": "Model not loaded"}), 500

    return jsonify({
        "status": "success",
        "model_type": type(model_package["model"]).__name__,
        "features": model_package["feature_names"],
        "target_classes": model_package["target_names"],
        "accuracy": round(model_package["accuracy"] * 100, 2),
        "sample_count": model_package.get("sample_count", 150),
        "n_estimators": model_package.get("n_estimators", 100)
    }), 200

@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Predict flower class from input features.
    Accepts JSON:
    {
        "features": [sepal_length, sepal_width, petal_length, petal_width]
    }
    or dictionary:
    {
        "sepal length (cm)": 5.1,
        "sepal width (cm)": 3.5,
        "petal length (cm)": 1.4,
        "petal width (cm)": 0.2
    }
    """
    if not model_package:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400

    clf = model_package["model"]
    feature_names = model_package["feature_names"]
    target_names = model_package["target_names"]

    # Parse input features
    features = None
    if "features" in data and isinstance(data["features"], list):
        if len(data["features"]) != len(feature_names):
            return jsonify({
                "error": f"Expected {len(feature_names)} features, got {len(data['features'])}"
            }), 400
        try:
            features = [float(x) for x in data["features"]]
        except (ValueError, TypeError):
            return jsonify({"error": "All feature values must be numeric numbers"}), 400

    elif all(name in data for name in feature_names):
        try:
            features = [float(data[name]) for name in feature_names]
        except (ValueError, TypeError):
            return jsonify({"error": "Feature values must be numeric"}), 400

    elif all(k in data for k in ["sepal_length", "sepal_width", "petal_length", "petal_width"]):
        try:
            features = [
                float(data["sepal_length"]),
                float(data["sepal_width"]),
                float(data["petal_length"]),
                float(data["petal_width"])
            ]
        except (ValueError, TypeError):
            return jsonify({"error": "Feature values must be numeric"}), 400

    else:
        return jsonify({
            "error": "Invalid format. Provide 'features': [l1, w1, l2, w2] or named keys (sepal_length, etc.)",
            "expected_features": feature_names
        }), 400

    X_in = np.array([features])
    pred_idx = int(clf.predict(X_in)[0])
    pred_label = target_names[pred_idx]

    probabilities = {}
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X_in)[0]
        for name, prob in zip(target_names, probs):
            probabilities[name] = round(float(prob) * 100, 2)

    return jsonify({
        "status": "success",
        "prediction_index": pred_idx,
        "prediction_label": pred_label,
        "probabilities": probabilities,
        "input_features": dict(zip(feature_names, features))
    }), 200

@app.route("/api/train", methods=["POST"])
def retrain():
    """
    Retrain the model on demand and update the in-memory model.
    Accepts optional JSON parameters:
    {
        "test_size": 0.2,
        "random_state": 42,
        "n_estimators": 100
    }
    """
    global model_package
    data = request.get_json(force=True, silent=True) or {}

    test_size = float(data.get("test_size", 0.2))
    random_state = int(data.get("random_state", 42))
    n_estimators = int(data.get("n_estimators", 100))

    try:
        model_package = train_and_save_model(
            model_path=MODEL_PATH,
            test_size=test_size,
            random_state=random_state,
            n_estimators=n_estimators
        )
        return jsonify({
            "status": "success",
            "message": "Model retrained and saved successfully with joblib!",
            "accuracy": round(model_package["accuracy"] * 100, 2),
            "n_estimators": n_estimators,
            "test_size": test_size,
            "random_state": random_state
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    print("[INFO] Starting Flask Server on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=True)
