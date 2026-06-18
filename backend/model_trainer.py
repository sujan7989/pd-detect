"""
Train ML models on the UCI Parkinson's dataset.
Trains RF, XGBoost, and SVM, then ensembles them via VotingClassifier.
Saves models + feature importance JSON to backend/models/.
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def load_data():
    print("Downloading UCI Parkinson's dataset...")
    df = pd.read_csv(DATA_URL)
    print(f"  Loaded {len(df)} samples, {df.shape[1]} columns.")
    # Drop the 'name' column (non-numeric identifier)
    if "name" in df.columns:
        df = df.drop(columns=["name"])
    X = df.drop(columns=["status"])
    y = df["status"]
    print(f"  Features: {list(X.columns)}")
    print(f"  Class distribution:\n{y.value_counts()}")
    return X, y, list(X.columns)


def train_models(X, y, feature_names):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

    # ── Scaler ──
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # ── 1. Random Forest ──
    print("\n[1/3] Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_sc, y_train)
    rf_pred = rf.predict(X_test_sc)
    rf_acc = accuracy_score(y_test, rf_pred)
    print(f"  RF Accuracy: {rf_acc:.4f}")
    print(classification_report(y_test, rf_pred, target_names=["Healthy", "Parkinson's"]))

    # ── 2. XGBoost ──
    print("[2/3] Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    xgb_model.fit(X_train_sc, y_train)
    xgb_pred = xgb_model.predict(X_test_sc)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    print(f"  XGB Accuracy: {xgb_acc:.4f}")
    print(classification_report(y_test, xgb_pred, target_names=["Healthy", "Parkinson's"]))

    # ── 3. SVM with calibrated probabilities ──
    print("[3/3] Training SVM (with probability calibration)...")
    svm_base = SVC(kernel="rbf", C=10, gamma="scale", random_state=42, probability=False)
    svm = CalibratedClassifierCV(svm_base, cv=5, method="sigmoid")
    svm.fit(X_train_sc, y_train)
    svm_pred = svm.predict(X_test_sc)
    svm_acc = accuracy_score(y_test, svm_pred)
    print(f"  SVM Accuracy: {svm_acc:.4f}")
    print(classification_report(y_test, svm_pred, target_names=["Healthy", "Parkinson's"]))

    # ── 4. Voting Ensemble ──
    print("\nTraining Voting Ensemble...")
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb_model), ("svm", svm)],
        voting="soft",
    )
    ensemble.fit(X_train_sc, y_train)
    ens_pred = ensemble.predict(X_test_sc)
    ens_acc = accuracy_score(y_test, ens_pred)
    print(f"\nEnsemble Accuracy: {ens_acc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, ens_pred))
    print(classification_report(y_test, ens_pred, target_names=["Healthy", "Parkinson's"]))

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(ensemble, X_train_sc, y_train, cv=cv, scoring="accuracy")
    print(f"5-fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Feature importance (from RF + XGBoost combined) ──
    rf_imp = rf.feature_importances_
    xgb_imp = xgb_model.feature_importances_
    combined_imp = (rf_imp + xgb_imp) / 2.0
    feature_importance = {
        name: float(imp) for name, imp in zip(feature_names, combined_imp)
    }
    feature_importance_sorted = dict(
        sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    )

    print("\nTop 10 Features:")
    for i, (name, imp) in enumerate(list(feature_importance_sorted.items())[:10]):
        print(f"  {i+1:2d}. {name:<30s}  {imp:.4f}")

    return scaler, rf, xgb_model, svm, ensemble, feature_importance_sorted


def save_models(scaler, rf, xgb_model, svm, ensemble, feature_importance):
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(rf, os.path.join(MODELS_DIR, "random_forest.pkl"))
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, "xgboost.pkl"))
    joblib.dump(svm, os.path.join(MODELS_DIR, "svm.pkl"))
    joblib.dump(ensemble, os.path.join(MODELS_DIR, "ensemble.pkl"))
    with open(os.path.join(MODELS_DIR, "feature_importance.json"), "w") as f:
        json.dump(feature_importance, f, indent=2)
    print(f"\nAll models saved to: {MODELS_DIR}")


def main():
    print("=" * 60)
    print("  Parkinson's Disease Detection - Model Training")
    print("=" * 60)
    X, y, feature_names = load_data()
    scaler, rf, xgb_model, svm, ensemble, feature_importance = train_models(
        X, y, feature_names
    )
    save_models(scaler, rf, xgb_model, svm, ensemble, feature_importance)
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
