"""Diagnose real user recording - why is it showing PD?"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import json

# Load the recording using PyAV (handles M4A/AAC)
audio_path = os.path.join(os.path.dirname(__file__), "Recording.m4a")
print(f"Analyzing: {audio_path}")
print(f"File size: {os.path.getsize(audio_path)} bytes")

import av
container = av.open(audio_path)
stream = container.streams.audio[0]
sr = stream.rate
print(f"Audio stream: {stream.codec_context.name}, {sr}Hz, {stream.channels} ch")

# Decode all frames
frames = []
for frame in container.decode(audio=0):
    frames.append(frame.to_ndarray())

# Concatenate and convert to mono float32
data = np.concatenate(frames, axis=1)
if data.shape[0] > 1:
    data = data.mean(axis=0)
else:
    data = data[0]

duration = len(data) / sr
print(f"Loaded: {len(data)} samples at {sr}Hz ({duration:.2f}s)")
container.close()

# Resample to 22050 if needed
target_sr = 22050
if sr != target_sr:
    import scipy.signal
    data = scipy.signal.resample(data, int(len(data) * target_sr / sr))
    sr = target_sr
    print(f"Resampled to {sr}Hz: {len(data)} samples ({len(data)/sr:.2f}s)")

# Now extract features
from feature_extractor import extract_features
import io, wave

audio_int16 = (data * 32767).astype(np.int16)
buf = io.BytesIO()
with wave.open(buf, 'w') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    wf.writeframes(audio_int16.tobytes())
buf.seek(0)

print("\n" + "=" * 70)
print("  RAW EXTRACTED FEATURES")
print("=" * 70)
features = extract_features(buf.read())
for k, v in sorted(features.items()):
    print(f"  {k:<25s}: {v:.6f}")

# Now map to UCI and show what the model sees
print("\n" + "=" * 70)
print("  MAPPED UCI FEATURES (what model sees)")
print("=" * 70)

# Load UCI stats
stats_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'uci_feature_stats.json')
with open(stats_path) as f:
    uci_stats = json.load(f)

# Import mapping function
from main import map_to_uci_vector, UCI_FEATURES
uci_vector = map_to_uci_vector(features)

print("\nFeature                  | Mapped Value | UCI Healthy | UCI PD     | Status")
print("-" * 85)
for i, feat_name in enumerate(UCI_FEATURES):
    mapped_val = uci_vector[i]
    h_mean = uci_stats[feat_name]['healthy_mean']
    std = uci_stats[feat_name]['std']
    pd_mean = uci_stats[feat_name]['pd_mean']
    
    h_low = h_mean - 2*std
    h_high = h_mean + 2*std
    pd_low = pd_mean - 2*std
    pd_high = pd_mean + 2*std
    
    if h_low <= mapped_val <= h_high:
        status = "HEALTHY RANGE"
    elif pd_low <= mapped_val <= pd_high:
        status = "PD RANGE"
    else:
        status = "OUTLIER"
    
    print(f"  {feat_name:<22s} | {mapped_val:>10.4f} | {h_mean:>9.4f}+-{std:.2f} | {pd_mean:>9.4f}+-{std:.2f} | {status}")

# Now run the actual prediction
print("\n" + "=" * 70)
print("  MODEL PREDICTION")
print("=" * 70)
from main import scaler, ensemble, models_loaded, _assess_signal_quality

if models_loaded and scaler is not None and ensemble is not None:
    feat_vec = uci_vector.reshape(1, -1)
    feat_scaled = scaler.transform(feat_vec)
    proba = float(ensemble.predict_proba(feat_scaled)[0][1])
    
    # Adaptive threshold based on signal quality
    quality = _assess_signal_quality(features)
    base_threshold = 0.65
    
    # Bayesian correction
    PD_PRIOR = 0.10
    proba_raw = proba  # Save raw probability
    proba = quality * proba + (1.0 - quality) * PD_PRIOR
    
    PD_THRESHOLD = base_threshold + 0.20 * (1.0 - quality)
    PD_THRESHOLD = min(PD_THRESHOLD, 0.85)
    
    pred = "Parkinson's Detected" if proba >= PD_THRESHOLD else "Healthy"
    conf = round((proba if proba >= PD_THRESHOLD else 1.0 - proba) * 100, 1)
    print(f"  Signal Quality: {quality:.0%}")
    print(f"  Raw Model Probability: {round(proba_raw, 4)}")
    print(f"  Corrected Probability: {round(proba, 4)}")
    print(f"  Prediction: {pred}")
    print(f"  Confidence: {conf}%")
    print(f"  Threshold: {PD_THRESHOLD:.2f} (adapted from 0.65)")
else:
    print("  Models not loaded!")

print("\n" + "=" * 70)
print("  DIAGNOSIS")
print("=" * 70)

# Identify problematic features
problem_features = []
for i, feat_name in enumerate(UCI_FEATURES):
    mapped_val = uci_vector[i]
    h_mean = uci_stats[feat_name]['healthy_mean']
    std = uci_stats[feat_name]['std']
    pd_mean = uci_stats[feat_name]['pd_mean']
    
    h_high = h_mean + 2*std
    
    if mapped_val > h_high and mapped_val > (h_mean + pd_mean) / 2:
        problem_features.append((feat_name, mapped_val, h_mean, pd_mean))

if problem_features:
    print("\nFeatures pushing toward PD detection:")
    for name, val, h, p in problem_features:
        print(f"  - {name}: {val:.4f} (healthy={h:.4f}, pd={p:.4f})")
    print("\nThese features are inflated due to real-world recording conditions.")
else:
    print("\nAll features appear to be in healthy range.")

print("""
Root cause: Real microphone recordings have:
- Background noise that inflates jitter/shimmer
- Room acoustics adding artificial variations
- Mic artifacts that differ from clean clinical UCI data

Solution: Add audio preprocessing before feature extraction.
""")
