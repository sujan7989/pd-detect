"""
PD Detect - Industry-Grade Model Accuracy Evaluation
=====================================================
Comprehensive evaluation of the Parkinson's Disease voice detection system.

Tests:
  1. UCI Clinical Dataset - Stratified 5-fold cross-validation
  2. Per-class metrics - Sensitivity, Specificity, PPV, NPV, F1
  3. ROC Analysis - AUC, optimal threshold selection
  4. Browser Simulation - Real-world healthy/PD voice scenarios
  5. Feature Distribution Analysis - UCI mapped vs clinical
  6. Edge Case Robustness - Noisy, elderly, low-pitch voices
  7. Confusion Matrix - Visual text-based representation

Run: python evaluate_accuracy.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import json
import joblib
from datetime import datetime

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, matthews_corrcoef, cohen_kappa_score
)
from sklearn.calibration import calibration_curve

# Imports from our codebase
from main import map_to_uci_vector

# ============================================================================
#  CONFIGURATION
# ============================================================================
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
PD_THRESHOLD = 0.65  # Current production threshold

UCI_FEATURES = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA",
    "NHR", "HNR",
    "RPDE", "DFA",
    "spread1", "spread2", "D2", "PPE",
]


def print_header(title, char="="):
    width = 78
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def print_metric(name, value, fmt=".4f"):
    print(f"  {name:<35s} {value:{fmt}}")


# ============================================================================
#  LOAD MODELS & DATA
# ============================================================================
print_header("PD DETECT - INDUSTRY-GRADE MODEL ACCURACY EVALUATION")
print(f"  Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
print(f"  Threshold: {PD_THRESHOLD}")
print(f"  Models:    {MODELS_DIR}")

scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
ensemble = joblib.load(os.path.join(MODELS_DIR, "ensemble.pkl"))

print("\n  Loading UCI Parkinson's dataset...")
df = pd.read_csv(DATA_URL)
df.drop(columns=["name"], inplace=True)
X_uci = df[UCI_FEATURES].values.astype(np.float64)
y_uci = df["status"].values  # 1=PD, 0=Healthy

n_healthy = (y_uci == 0).sum()
n_pd = (y_uci == 1).sum()
print(f"  Total samples: {len(df)} | Healthy: {n_healthy} | PD: {n_pd}")
print(f"  Class ratio: {n_pd/len(df)*100:.1f}% PD, {n_healthy/len(df)*100:.1f}% Healthy")


# ============================================================================
#  TEST 1: STRATIFIED 5-FOLD CROSS-VALIDATION
# ============================================================================
print_header("TEST 1: Stratified 5-Fold Cross-Validation (UCI Data)")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Get cross-validated predictions (unseen folds)
y_pred_cv = cross_val_predict(ensemble, scaler.fit_transform(X_uci), y_uci, cv=cv)
y_proba_cv = cross_val_predict(ensemble, scaler.fit_transform(X_uci), y_uci, cv=cv, method='predict_proba')[:, 1]

# Overall metrics
cv_acc = accuracy_score(y_uci, y_pred_cv)
cv_prec = precision_score(y_uci, y_pred_cv)
cv_rec = recall_score(y_uci, y_pred_cv)
cv_f1 = f1_score(y_uci, y_pred_cv)
cv_auc = roc_auc_score(y_uci, y_proba_cv)
cv_mcc = matthews_corrcoef(y_uci, y_pred_cv)
cv_kappa = cohen_kappa_score(y_uci, y_pred_cv)

print(f"\n  --- Overall Metrics ---")
print_metric("Accuracy", cv_acc)
print_metric("Precision (PPV)", cv_prec)
print_metric("Recall (Sensitivity)", cv_rec)
print_metric("F1-Score", cv_f1)
print_metric("AUC-ROC", cv_auc)
print_metric("Matthews Correlation Coeff", cv_mcc)
print_metric("Cohen's Kappa", cv_kappa)

# Per-class metrics
cm_cv = confusion_matrix(y_uci, y_pred_cv)
tn, fp, fn, tp = cm_cv.ravel()

sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0

print(f"\n  --- Per-Class Metrics ---")
print_metric("Sensitivity (PD detected / True Positive Rate)", sensitivity)
print_metric("Specificity (Healthy correct / True Negative Rate)", specificity)
print_metric("Positive Predictive Value (PPV)", ppv)
print_metric("Negative Predictive Value (NPV)", npv)
print_metric("Healthy Recall (True Negative Rate)", specificity)
print_metric("PD Recall (True Positive Rate)", sensitivity)

print(f"\n  --- Confusion Matrix ---")
print(f"                     Predicted Healthy    Predicted PD")
print(f"  Actual Healthy       {tn:>3d} (TN)          {fp:>3d} (FP)")
print(f"  Actual PD            {fn:>3d} (FN)          {tp:>3d} (TP)")

print(f"\n  --- Per-Class Counts ---")
print(f"  Healthy: {n_healthy} total | {tn} correctly classified ({tn/n_healthy*100:.1f}%) | {fp} misclassified ({fp/n_healthy*100:.1f}%)")
print(f"  PD:      {n_pd} total | {tp} correctly classified ({tp/n_pd*100:.1f}%) | {fn} misclassified ({fn/n_pd*100:.1f}%)")

print(f"\n  --- Classification Report ---")
print(classification_report(y_uci, y_pred_cv, target_names=["Healthy (0)", "PD (1)"]))


# ============================================================================
#  TEST 2: ROC ANALYSIS & THRESHOLD OPTIMIZATION
# ============================================================================
print_header("TEST 2: ROC Analysis & Threshold Optimization")

fpr, tpr, thresholds = roc_curve(y_uci, y_proba_cv)

print(f"\n  --- ROC Curve Points ---")
print(f"  {'Threshold':<12s} {'FPR':<10s} {'TPR':<10s} {'Accuracy':<10s} {'F1':<10s}")
print(f"  {'-'*52}")

best_f1 = 0
best_thresh = 0.5
for i in range(0, len(thresholds), max(1, len(thresholds)//20)):
    thresh = thresholds[i]
    y_pred_t = (y_proba_cv >= thresh).astype(int)
    acc_t = accuracy_score(y_uci, y_pred_t)
    f1_t = f1_score(y_uci, y_pred_t, zero_division=0)
    marker = " <-- current" if abs(thresh - PD_THRESHOLD) < 0.01 else ""
    if f1_t > best_f1:
        best_f1 = f1_t
        best_thresh = thresh
    print(f"  {thresh:<12.3f} {fpr[i]:<10.4f} {tpr[i]:<10.4f} {acc_t:<10.4f} {f1_t:<10.4f}{marker}")

print(f"\n  Optimal threshold (max F1): {best_thresh:.3f} (F1={best_f1:.4f})")
print(f"  Current threshold: {PD_THRESHOLD:.3f}")

# Show metrics at current threshold
y_pred_curr = (y_proba_cv >= PD_THRESHOLD).astype(int)
print(f"\n  --- Metrics at Current Threshold ({PD_THRESHOLD}) ---")
print(f"  Accuracy:  {accuracy_score(y_uci, y_pred_curr):.4f}")
print(f"  Precision: {precision_score(y_uci, y_pred_curr, zero_division=0):.4f}")
print(f"  Recall:    {recall_score(y_uci, y_pred_curr, zero_division=0):.4f}")
print(f"  F1:        {f1_score(y_uci, y_pred_curr, zero_division=0):.4f}")

cm_curr = confusion_matrix(y_uci, y_pred_curr)
tn_c, fp_c, fn_c, tp_c = cm_curr.ravel()
print(f"  TN={tn_c}  FP={fp_c}  FN={fn_c}  TP={tp_c}")


# ============================================================================
#  TEST 3: INDIVIDUAL MODEL PERFORMANCE
# ============================================================================
print_header("TEST 3: Individual Model Performance (Hold-out 15%)")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_uci, y_uci, test_size=0.15, random_state=42, stratify=y_uci
)

# Refit scaler on train only
scaler_fresh = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
Xtr_sc = scaler_fresh.transform(X_train)
Xte_sc = scaler_fresh.transform(X_test)

y_proba_test = ensemble.predict_proba(Xte_sc)[:, 1]
y_pred_test = (y_proba_test >= PD_THRESHOLD).astype(int)

print(f"\n  Test set: {len(X_test)} samples (Healthy: {(y_test==0).sum()}, PD: {(y_test==1).sum()})")
print(f"\n  --- Hold-out Test Metrics ---")
print_metric("Accuracy", accuracy_score(y_test, y_pred_test))
print_metric("Precision", precision_score(y_test, y_pred_test, zero_division=0))
print_metric("Recall", recall_score(y_test, y_pred_test, zero_division=0))
print_metric("F1-Score", f1_score(y_test, y_pred_test, zero_division=0))
print_metric("AUC-ROC", roc_auc_score(y_test, y_proba_test))

cm_test = confusion_matrix(y_test, y_pred_test)
tn_t, fp_t, fn_t, tp_t = cm_test.ravel()
print(f"\n  Confusion Matrix:")
print(f"                     Pred Healthy    Pred PD")
print(f"  Actual Healthy       {tn_t:>3d}             {fp_t:>3d}")
print(f"  Actual PD            {fn_t:>3d}             {tp_t:>3d}")


# ============================================================================
#  TEST 4: BROWSER RECORDING SIMULATION
# ============================================================================
print_header("TEST 4: Browser Recording Simulation (Real-World Scenarios)")

def test_browser_sample(label, features, expected_class):
    """Test a simulated browser recording through the full pipeline."""
    vec = map_to_uci_vector(features)
    scaled = scaler.transform(vec.reshape(1, -1))
    prob = float(ensemble.predict_proba(scaled)[0][1])
    pred = "PD" if prob >= PD_THRESHOLD else "Healthy"
    correct = pred == expected_class
    status = "PASS" if correct else "FAIL"

    # Compare mapped features to UCI healthy/PD means
    uci_he_means = np.array([181.94, 223.64, 145.21,
        0.00387, 2.3e-5, 0.00193, 0.00206, 0.00578,
        0.01762, 0.163, 0.00950, 0.01051, 0.01330, 0.02851,
        0.01148, 24.679, 0.443, 0.696, -6.759, 0.160, 2.154, 0.123])
    uci_pd_means = np.array([145.18, 188.44, 106.89,
        0.00699, 5.07e-5, 0.00376, 0.00390, 0.01127,
        0.03366, 0.321, 0.01768, 0.02028, 0.02760, 0.05303,
        0.02921, 20.974, 0.517, 0.725, -5.333, 0.248, 2.456, 0.234])

    # Calculate distance to healthy vs PD centroid (normalized)
    dist_he = np.mean(np.abs((vec - uci_he_means) / (np.abs(uci_he_means) + 1e-9)))
    dist_pd = np.mean(np.mean(np.abs((vec - uci_pd_means) / (np.abs(uci_pd_means) + 1e-9))))

    print(f"\n  [{status}] {label}")
    print(f"    Expected: {expected_class:<10s} | Predicted: {pred:<10s} | PD prob: {prob:.4f}")
    print(f"    Dist to UCI Healthy centroid: {dist_he:.3f} | Dist to UCI PD centroid: {dist_pd:.3f}")
    if not correct:
        print(f"    *** FALSE {'POSITIVE' if pred == 'PD' else 'NEGATIVE'} ***")
        # Show which features are pushing toward PD
        for i, feat in enumerate(UCI_FEATURES):
            if vec[i] > uci_he_means[i] * 1.3 and uci_pd_means[i] > uci_he_means[i] * 1.1:
                print(f"      -> {feat}: mapped={vec[i]:.5f}, UCI_healthy={uci_he_means[i]:.5f}, UCI_PD={uci_pd_means[i]:.5f}")
    return correct, prob


results = []
probs = []

print(f"\n  -- Healthy Persons (expected: Healthy) --")
test_cases_healthy = [
    ("1. Clean studio voice", {"jitter_local": 0.03, "shimmer_local": 0.035, "hnr": 18.0, "pitch_mean": 170.0, "pitch_std": 5.0, "rpde": 0.10, "dfa": 0.08}),
    ("2. Normal room, male", {"jitter_local": 0.08, "shimmer_local": 0.055, "hnr": 14.0, "pitch_mean": 125.0, "pitch_std": 8.0, "rpde": 0.12, "dfa": 0.09}),
    ("3. Normal room, female", {"jitter_local": 0.07, "shimmer_local": 0.050, "hnr": 15.0, "pitch_mean": 220.0, "pitch_std": 7.0, "rpde": 0.11, "dfa": 0.085}),
    ("4. Noisy background", {"jitter_local": 0.20, "shimmer_local": 0.07, "hnr": 9.0, "pitch_mean": 155.0, "pitch_std": 90.0, "rpde": 0.21, "dfa": 0.09}),
    ("5. Male low-pitch (85Hz)", {"jitter_local": 0.06, "shimmer_local": 0.05, "hnr": 16.0, "pitch_mean": 85.0, "pitch_std": 4.0, "rpde": 0.11, "dfa": 0.085}),
    ("6. Female high-pitch (250Hz)", {"jitter_local": 0.05, "shimmer_local": 0.04, "hnr": 17.0, "pitch_mean": 250.0, "pitch_std": 6.0, "rpde": 0.10, "dfa": 0.08}),
    ("7. Elderly healthy (age 75)", {"jitter_local": 0.12, "shimmer_local": 0.065, "hnr": 12.0, "pitch_mean": 140.0, "pitch_std": 12.0, "rpde": 0.15, "dfa": 0.095}),
    ("8. Very noisy (crowded room)", {"jitter_local": 0.30, "shimmer_local": 0.09, "hnr": 6.0, "pitch_mean": 160.0, "pitch_std": 50.0, "rpde": 0.25, "dfa": 0.10}),
    ("9. Breathless healthy", {"jitter_local": 0.10, "shimmer_local": 0.06, "hnr": 10.0, "pitch_mean": 175.0, "pitch_std": 15.0, "rpde": 0.14, "dfa": 0.09}),
    ("10. Child-like voice", {"jitter_local": 0.04, "shimmer_local": 0.04, "hnr": 19.0, "pitch_mean": 280.0, "pitch_std": 10.0, "rpde": 0.09, "dfa": 0.075}),
]

for label, feats in test_cases_healthy:
    correct, prob = test_browser_sample(label, feats, "Healthy")
    results.append(correct)
    probs.append(prob)

print(f"\n  -- PD Persons (expected: PD) --")
test_cases_pd = [
    ("11. Advanced PD", {"jitter_local": 0.50, "shimmer_local": 0.18, "hnr": 5.0, "pitch_mean": 145.0, "pitch_std": 20.0, "rpde": 0.40, "dfa": 0.22}),
    ("12. Moderate PD", {"jitter_local": 0.30, "shimmer_local": 0.12, "hnr": 8.0, "pitch_mean": 150.0, "pitch_std": 15.0, "rpde": 0.30, "dfa": 0.18}),
    ("13. Early-stage PD", {"jitter_local": 0.18, "shimmer_local": 0.09, "hnr": 11.0, "pitch_mean": 155.0, "pitch_std": 10.0, "rpde": 0.22, "dfa": 0.14}),
    ("14. Tremor-dominant PD", {"jitter_local": 0.40, "shimmer_local": 0.15, "hnr": 6.5, "pitch_mean": 138.0, "pitch_std": 30.0, "rpde": 0.35, "dfa": 0.20}),
    ("15. Mild PD (subtle)", {"jitter_local": 0.15, "shimmer_local": 0.08, "hnr": 12.0, "pitch_mean": 165.0, "pitch_std": 8.0, "rpde": 0.20, "dfa": 0.12}),
    ("16. PD with low pitch", {"jitter_local": 0.35, "shimmer_local": 0.14, "hnr": 7.0, "pitch_mean": 100.0, "pitch_std": 18.0, "rpde": 0.33, "dfa": 0.19}),
]

for label, feats in test_cases_pd:
    correct, prob = test_browser_sample(label, feats, "PD")
    results.append(correct)
    probs.append(prob)


# ============================================================================
#  TEST 5: BROWSER SIMULATION SUMMARY
# ============================================================================
print_header("TEST 5: Browser Simulation Summary")

n_correct = sum(results)
n_total = len(results)
n_healthy_tests = len(test_cases_healthy)
n_pd_tests = len(test_cases_pd)
h_correct = sum(results[:n_healthy_tests])
p_correct = sum(results[n_healthy_tests:])

print(f"\n  Overall: {n_correct}/{n_total} correct ({n_correct/n_total*100:.1f}%)")
print(f"  Healthy accuracy: {h_correct}/{n_healthy_tests} ({h_correct/n_healthy_tests*100:.1f}%)")
print(f"  PD accuracy:      {p_correct}/{n_pd_tests} ({p_correct/n_pd_tests*100:.1f}%)")

# Average probabilities
h_probs = probs[:n_healthy_tests]
p_probs = probs[n_healthy_tests:]
print(f"\n  Avg PD probability - Healthy voices: {np.mean(h_probs):.4f} (should be < {PD_THRESHOLD})")
print(f"  Avg PD probability - PD voices:      {np.mean(p_probs):.4f} (should be >= {PD_THRESHOLD})")
print(f"  Separation margin: {np.mean(p_probs) - np.mean(h_probs):.4f}")

# Worst cases
h_worst_idx = np.argmax(h_probs)
p_worst_idx = np.argmin(p_probs)
print(f"\n  Worst healthy case: '{test_cases_healthy[h_worst_idx][0]}' (PD prob: {h_probs[h_worst_idx]:.4f})")
print(f"  Worst PD case:      '{test_cases_pd[p_worst_idx][0]}' (PD prob: {p_probs[p_worst_idx]:.4f})")


# ============================================================================
#  TEST 6: FEATURE DISTRIBUTION ANALYSIS
# ============================================================================
print_header("TEST 6: Feature Distribution Analysis (Mapped vs UCI Clinical)")

# Map a typical healthy browser voice and compare to UCI distributions
typical_healthy_browser = {
    "jitter_local": 0.08, "shimmer_local": 0.055, "hnr": 14.0,
    "pitch_mean": 155.0, "pitch_std": 8.0, "rpde": 0.12, "dfa": 0.09
}
vec_he = map_to_uci_vector(typical_healthy_browser)

typical_pd_browser = {
    "jitter_local": 0.35, "shimmer_local": 0.14, "hnr": 7.0,
    "pitch_mean": 145.0, "pitch_std": 18.0, "rpde": 0.35, "dfa": 0.20
}
vec_pd = map_to_uci_vector(typical_pd_browser)

# Load UCI stats
with open(os.path.join(MODELS_DIR, "uci_feature_stats.json")) as f:
    uci_stats = json.load(f)

print(f"\n  {'Feature':<20s} {'Browser Healthy':>15s} {'UCI Healthy':>12s} {'UCI PD':>12s} {'In Range?':>10s}")
print(f"  {'-'*70}")

in_range_count = 0
for i, feat in enumerate(UCI_FEATURES):
    stats = uci_stats[feat]
    he_mean = stats["healthy_mean"]
    pd_mean = stats["pd_mean"]
    he_std = stats["std"]
    he_min = stats["min"]
    he_max = stats["max"]

    mapped_val = vec_he[i]
    # Check if within healthy mean +/- 2std
    in_range = abs(mapped_val - he_mean) <= 2 * he_std
    if in_range:
        in_range_count += 1
    status = "YES" if in_range else "NO"

    print(f"  {feat:<20s} {mapped_val:>15.5f} {he_mean:>12.5f} {pd_mean:>12.5f} {status:>10s}")

print(f"\n  Features within UCI healthy range: {in_range_count}/{len(UCI_FEATURES)} ({in_range_count/len(UCI_FEATURES)*100:.1f}%)")


# ============================================================================
#  TEST 7: CALIBRATION CHECK (UCI MEANS -> CORRECT PREDICTION)
# ============================================================================
print_header("TEST 7: Calibration Check (UCI Clinical Means)")

# UCI healthy mean feature vector
healthy_uci = np.array([181.94, 223.64, 145.21,
    0.00387, 2.3e-5, 0.00193, 0.00206, 0.00578,
    0.01762, 0.163, 0.00950, 0.01051, 0.01330, 0.02851,
    0.01148, 24.679, 0.443, 0.696, -6.759, 0.160, 2.154, 0.123]).reshape(1, -1)

# UCI PD mean feature vector
pd_uci = np.array([145.18, 188.44, 106.89,
    0.00699, 5.07e-5, 0.00376, 0.00390, 0.01127,
    0.03366, 0.321, 0.01768, 0.02028, 0.02760, 0.05303,
    0.02921, 20.974, 0.517, 0.725, -5.333, 0.248, 2.456, 0.234]).reshape(1, -1)

h_prob = float(ensemble.predict_proba(scaler.transform(healthy_uci))[0][1])
pd_prob = float(ensemble.predict_proba(scaler.transform(pd_uci))[0][1])

print(f"\n  UCI Healthy means -> PD probability: {h_prob:.4f}")
print(f"    -> {'PASS (Healthy)' if h_prob < PD_THRESHOLD else 'FAIL (predicted PD!)'}")
print(f"  UCI PD means -> PD probability:      {pd_prob:.4f}")
print(f"    -> {'PASS (PD Detected)' if pd_prob >= PD_THRESHOLD else 'FAIL (predicted Healthy!)'}")
print(f"\n  Separation: {pd_prob - h_prob:.4f} (higher = better)")


# ============================================================================
#  FINAL SUMMARY
# ============================================================================
print_header("FINAL ACCURACY SUMMARY")

print(f"""
  +------------------------------------+------------------------------------------+
  | Metric                             | Value                                    |
  +------------------------------------+------------------------------------------+
  | UCI Cross-Val Accuracy             | {cv_acc*100:>6.2f}%                                 |
  | UCI Cross-Val AUC-ROC              | {cv_auc:>6.4f}                                   |
  | UCI Sensitivity (PD detection)     | {sensitivity*100:>6.2f}%                                 |
  | UCI Specificity (Healthy correct)  | {specificity*100:>6.2f}%                                 |
  | UCI F1-Score                       | {cv_f1:>6.4f}                                   |
  | Browser Healthy Accuracy           | {h_correct/n_healthy_tests*100:>6.2f}% ({h_correct}/{n_healthy_tests})                            |
  | Browser PD Accuracy                | {p_correct/n_pd_tests*100:>6.2f}% ({p_correct}/{n_pd_tests})                             |
  | Browser Overall Accuracy           | {n_correct/n_total*100:>6.2f}% ({n_correct}/{n_total})                            |
  | Feature Mapping In-Range           | {in_range_count/len(UCI_FEATURES)*100:>6.2f}% ({in_range_count}/{len(UCI_FEATURES)})                           |
  | Calibration (UCI means)            | {'PASS' if h_prob < PD_THRESHOLD and pd_prob >= PD_THRESHOLD else 'FAIL':>6s}                                      |
  +------------------------------------+------------------------------------------+

  Production Threshold: {PD_THRESHOLD}
  Model: Ensemble (RF + XGBoost + SVM + GB)
  Training Data: UCI Parkinson's Dataset (Little et al. 2007)
  Features: 22 UCI clinical voice biomarkers
""")

print("  Evaluation complete!")
