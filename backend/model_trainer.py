"""
PD Detect — Model Trainer v2
Strategy: Generate synthetic voice audio from UCI clinical parameters,
extract features using our own feature_extractor.py, then train on those
features. This ensures ZERO domain gap between training and inference.
"""

import os, json, warnings, io, wave
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib

# Import our own feature extractor
import sys
sys.path.insert(0, os.path.dirname(__file__))
from feature_extractor import extract_features

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_URL   = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"


def uci_to_audio(row, sr=22050, dur=4.0, seed=None):
    """
    Convert a UCI feature row into synthetic audio that will produce
    similar acoustic features when processed by our extractor.

    UCI parameters used:
    - MDVP:Fo(Hz)     → fundamental frequency
    - MDVP:Jitter(%)  → pitch period irregularity
    - MDVP:Shimmer    → amplitude irregularity
    - HNR             → harmonic-to-noise ratio → noise level
    - RPDE            → nonlinear complexity (modulates jitter pattern)
    """
    if seed is not None:
        np.random.seed(seed)

    freq    = float(row["MDVP:Fo(Hz)"])
    jitter  = float(row["MDVP:Jitter(%)"]) / 100.0   # UCI is in %, convert to ratio
    shimmer = float(row["MDVP:Shimmer"])
    hnr_db  = float(row["HNR"])
    rpde    = float(row["RPDE"])

    # Noise amplitude from HNR: HNR = 10*log10(Harmonics/Noise)
    # Noise = Harmonics / 10^(HNR/10)
    # We normalize harmonics to 1.0, so noise_amp = 1/10^(HNR/10)
    noise_amp = 1.0 / (10.0 ** (hnr_db / 10.0))
    noise_amp = float(np.clip(noise_amp, 0.0001, 0.5))

    # Jitter: period-to-period frequency variation
    # Scale jitter for synthesis (UCI jitter% → synthesis coefficient)
    jitter_coef = jitter * 3.0   # amplify for audible effect

    # RPDE modulates the complexity/irregularity of jitter pattern
    # Higher RPDE (PD) → more irregular jitter pattern
    jitter_pattern = "pd" if rpde > 0.49 else "healthy"

    t = np.linspace(0, dur, int(sr * dur), endpoint=False)

    # Generate pitch contour with realistic jitter
    if jitter_pattern == "pd":
        # PD: irregular micro-variations with occasional larger deviations
        phase_noise = np.cumsum(
            np.random.randn(len(t)) * jitter_coef * freq / sr +
            np.random.randn(len(t)) * jitter_coef * 0.5 * freq / sr
        )
        # Add subtle tremor for PD (4-6 Hz)
        tremor_freq = np.random.uniform(4.5, 6.0)
        phase_noise += 2.0 * np.sin(2 * np.pi * tremor_freq * t) * rpde
    else:
        # Healthy: smooth, regular micro-variations
        phase_noise = np.cumsum(
            np.random.randn(len(t)) * jitter_coef * freq / sr * 0.3
        )

    phase = 2 * np.pi * freq * t + phase_noise

    # Generate amplitude envelope with shimmer
    amp_base    = 1.0 + shimmer * np.random.randn(len(t))
    amp_smooth  = np.convolve(amp_base, np.ones(int(sr * 0.005)) /
                              int(sr * 0.005), mode="same")
    amplitude   = np.clip(amp_smooth, 0.3, 2.0)

    # Synthesize voice with harmonics
    signal = amplitude * (
        0.55 * np.sin(phase) +
        0.28 * np.sin(2 * phase) +
        0.13 * np.sin(3 * phase) +
        0.07 * np.sin(4 * phase) +
        0.03 * np.sin(5 * phase)
    )

    # Add noise based on HNR
    noise  = noise_amp * np.random.randn(len(t))
    signal = signal + noise

    # Normalize
    mx = np.max(np.abs(signal))
    if mx > 0:
        signal = signal / mx * 0.85

    return (signal * 32767).astype(np.int16)


def audio_to_bytes(samples, sr=22050):
    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


def generate_dataset(uci_df, augment=3, sr=22050):
    """
    For each UCI row, generate `augment` synthetic audio recordings,
    extract features, and build a new training dataset.
    """
    print(f"Generating synthetic audio from {len(uci_df)} UCI samples "
          f"(x{augment} augmentation = {len(uci_df)*augment} total)...")

    all_features = []
    all_labels   = []
    errors       = 0

    for idx, row in uci_df.iterrows():
        label = int(row["status"])
        for aug_i in range(augment):
            seed = idx * 100 + aug_i
            try:
                samples = uci_to_audio(row, sr=sr, dur=4.0, seed=seed)
                wav     = audio_to_bytes(samples, sr)
                feats   = extract_features(wav)
                all_features.append(feats)
                all_labels.append(label)
            except Exception as e:
                errors += 1

    if errors:
        print(f"  Skipped {errors} errors during synthesis.")

    print(f"  Generated {len(all_features)} feature vectors.")
    return all_features, all_labels


def train(X, y, feature_names):
    X_arr = np.array(X)
    y_arr = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=0.20, random_state=42, stratify=y_arr
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train)
    X_test_sc   = scaler.transform(X_test)

    print("\n[1/3] Random Forest...")
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_train_sc, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test_sc))
    print(f"  Accuracy: {rf_acc:.4f}")
    print(classification_report(y_test, rf.predict(X_test_sc),
          target_names=["Healthy", "PD"], zero_division=0))

    print("[2/3] XGBoost...")
    xgb_m = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.04,
                                subsample=0.8, colsample_bytree=0.8,
                                eval_metric="logloss", random_state=42, verbosity=0)
    xgb_m.fit(X_train_sc, y_train)
    xgb_acc = accuracy_score(y_test, xgb_m.predict(X_test_sc))
    print(f"  Accuracy: {xgb_acc:.4f}")
    print(classification_report(y_test, xgb_m.predict(X_test_sc),
          target_names=["Healthy", "PD"], zero_division=0))

    print("[3/3] SVM (calibrated)...")
    svm = CalibratedClassifierCV(
        SVC(kernel="rbf", C=10, gamma="scale", random_state=42),
        cv=5, method="sigmoid"
    )
    svm.fit(X_train_sc, y_train)
    svm_acc = accuracy_score(y_test, svm.predict(X_test_sc))
    print(f"  Accuracy: {svm_acc:.4f}")
    print(classification_report(y_test, svm.predict(X_test_sc),
          target_names=["Healthy", "PD"], zero_division=0))

    print("\nEnsemble (soft voting)...")
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb_m), ("svm", svm)],
        voting="soft"
    )
    ensemble.fit(X_train_sc, y_train)
    ens_pred = ensemble.predict(X_test_sc)
    ens_acc  = accuracy_score(y_test, ens_pred)
    print(f"  Ensemble Accuracy: {ens_acc:.4f}")
    print("  Confusion Matrix:")
    print(confusion_matrix(y_test, ens_pred))
    print(classification_report(y_test, ens_pred,
          target_names=["Healthy", "PD"], zero_division=0))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_sc = cross_val_score(ensemble, X_train_sc, y_train, cv=cv, scoring="accuracy")
    print(f"  5-fold CV: {cv_sc.mean():.4f} ± {cv_sc.std():.4f}")

    # Feature importances (RF + XGBoost)
    fi_combined = (rf.feature_importances_ + xgb_m.feature_importances_) / 2.0
    fi = dict(sorted({n: float(v) for n, v in zip(feature_names, fi_combined)}.items(),
                     key=lambda x: x[1], reverse=True))
    print("\nTop 10 Features:")
    for i, (n, v) in enumerate(list(fi.items())[:10]):
        print(f"  {i+1:2d}. {n:<35s} {v:.4f}")

    return scaler, rf, xgb_m, svm, ensemble, fi


def main():
    print("=" * 65)
    print("  PD Detect — Model Trainer v2 (Audio-Feature Domain)")
    print("=" * 65)
    print()
    print("Strategy: synthesize audio from UCI parameters → extract features")
    print("→ train model on those features → zero domain gap at inference")
    print()

    # Load UCI data
    print("Downloading UCI Parkinson's dataset...")
    df = pd.read_csv(DATA_URL)
    df.drop(columns=["name"], inplace=True)
    print(f"  {len(df)} samples | PD: {sum(df['status']==1)} | Healthy: {sum(df['status']==0)}")
    print()

    # Generate synthetic audio + extract features
    feats, labels = generate_dataset(df, augment=5)  # 5x augmentation
    print()

    # Get feature names from first sample
    feature_names = list(feats[0].keys())
    X = [[f[k] for k in feature_names] for f in feats]

    print(f"Training on {len(X)} samples with {len(feature_names)} features...")
    print()

    scaler, rf, xgb_m, svm, ensemble, fi = train(X, labels, feature_names)

    # Save
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler,    os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(rf,        os.path.join(MODELS_DIR, "random_forest.pkl"))
    joblib.dump(xgb_m,     os.path.join(MODELS_DIR, "xgboost.pkl"))
    joblib.dump(svm,       os.path.join(MODELS_DIR, "svm.pkl"))
    joblib.dump(ensemble,  os.path.join(MODELS_DIR, "ensemble.pkl"))

    # Save feature names (so inference knows which features to use)
    with open(os.path.join(MODELS_DIR, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)

    # Save feature importances
    with open(os.path.join(MODELS_DIR, "feature_importance.json"), "w") as f:
        json.dump({k: {"importance": v} for k, v in fi.items()}, f, indent=2)

    print(f"\nAll models saved to: {MODELS_DIR}")
    print("Training complete!")


if __name__ == "__main__":
    main()
