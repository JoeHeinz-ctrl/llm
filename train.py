"""
Model Training Script
Trains a Random Forest Classifier on the Iris dataset and saves the artifact using joblib.
"""

import os
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")

def train_and_save_model(model_path=MODEL_PATH, test_size=0.2, random_state=42, n_estimators=100):
    """
    Trains a Random Forest classifier and persists the model + metadata using joblib.
    """
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    target_names = list(iris.target_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))

    package = {
        "model": clf,
        "feature_names": feature_names,
        "target_names": target_names,
        "accuracy": accuracy,
        "sample_count": len(X),
        "n_estimators": n_estimators,
        "random_state": random_state
    }

    joblib.dump(package, model_path)
    print(f"[OK] Model successfully trained and saved to: {model_path}")
    print(f"[OK] Test Accuracy: {accuracy * 100:.2f}%")
    return package

if __name__ == "__main__":
    train_and_save_model()
