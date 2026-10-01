"""
Python Client using the 'requests' library to interact with the Flask ML API.
Demonstrates:
1. Health check (GET /api/health)
2. Model info & accuracy (GET /api/info)
3. Inference on flower samples (POST /api/predict)
4. Triggering on-demand model retraining (POST /api/train)
"""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:5000"

def pretty_print(title, data):
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")
    print(json.dumps(data, indent=2))

def run_client_demo():
    print(f"Connecting to Flask API at: {BASE_URL} ...\n")

    # 1. Health Check
    try:
        res = requests.get(f"{BASE_URL}/api/health")
        pretty_print("1. Health Check [GET /api/health]", res.json())
    except requests.exceptions.ConnectionError:
        print("[ERROR] Could not connect to Flask server. Please make sure 'python app.py' is running!")
        return

    # 2. Get Model Info
    res = requests.get(f"{BASE_URL}/api/info")
    pretty_print("2. Model Info & Accuracy [GET /api/info]", res.json())

    # 3. Test Predictions using 'requests.post' with json payload
    test_samples = [
        {
            "name": "Sample A (Typical Iris Setosa)",
            "payload": {"features": [5.1, 3.5, 1.4, 0.2]}
        },
        {
            "name": "Sample B (Typical Iris Versicolor)",
            "payload": {"features": [6.0, 2.9, 4.5, 1.5]}
        },
        {
            "name": "Sample C (Typical Iris Virginica)",
            "payload": {"features": [6.7, 3.1, 5.6, 2.4]}
        }
    ]

    for sample in test_samples:
        res = requests.post(f"{BASE_URL}/api/predict", json=sample["payload"])
        pretty_print(f"3. Prediction: {sample['name']} [POST /api/predict]", res.json())

    # 4. Trigger Model Retraining with custom parameters
    retrain_payload = {
        "n_estimators": 150,
        "test_size": 0.25,
        "random_state": 99
    }
    print("\nSending retraining request via requests.post() ...")
    res = requests.post(f"{BASE_URL}/api/train", json=retrain_payload)
    pretty_print("4. Model Retrained Dynamically [POST /api/train]", res.json())

    # 5. Verify updated info after retraining
    res = requests.get(f"{BASE_URL}/api/info")
    pretty_print("5. Verify Model State After Retrain [GET /api/info]", res.json())

    print("\n[SUCCESS] All API tests using 'requests' completed successfully!")

if __name__ == "__main__":
    run_client_demo()
