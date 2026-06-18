"""
Audio feature extraction for Parkinson's Disease Detection.
Extracts ~60 acoustic features from voice recordings.
"""

import numpy as np
import librosa
import soundfile as sf
import io
import warnings
warnings.filterwarnings("ignore")


def extract_features(audio_bytes: bytes, sr_target: int = 22050) -> dict:
    """
    Extract acoustic features from audio bytes.
    Returns a dictionary of ~60 features used for PD detection.
    """
    # Load audio from bytes
    audio_io = io.BytesIO(audio_bytes)
    try:
        y, sr = sf.read(audio_io)
        # Convert stereo to mono
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        # Resample if needed
        if sr != sr_target:
            y = librosa.resample(y, orig_sr=sr, target_sr=sr_target)
            sr = sr_target
    except Exception:
        # Fallback: try librosa directly
        audio_io.seek(0)
        y, sr = librosa.load(audio_io, sr=sr_target, mono=True)

    # Ensure float32
    y = y.astype(np.float32)
    # Remove silence
    y, _ = librosa.effects.trim(y, top_db=20)
    if len(y) < sr * 0.1:
        raise ValueError("Audio too short for analysis (< 0.1 seconds after trimming).")

    features = {}

    # ── 1. MFCCs (13) + deltas (13) + delta-deltas (13) = 39 features ──
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    for i in range(13):
        features[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc_delta_{i+1}_mean"] = float(np.mean(mfcc_delta[i]))
        features[f"mfcc_delta2_{i+1}_mean"] = float(np.mean(mfcc_delta2[i]))

    # ── 2. Pitch (fundamental frequency) ──
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
    )
    voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
    if len(voiced_f0) < 2:
        voiced_f0 = np.array([150.0, 155.0])  # fallback

    features["pitch_mean"] = float(np.mean(voiced_f0))
    features["pitch_std"] = float(np.std(voiced_f0))

    # ── 3. Jitter (period-to-period variation in pitch) ──
    # Compute periods from f0 contour
    periods = 1.0 / voiced_f0[voiced_f0 > 0]
    if len(periods) < 2:
        periods = np.array([1 / 150.0, 1 / 155.0])

    jitter_local = _jitter_local(periods)
    jitter_absolute = float(np.mean(np.abs(np.diff(periods))))
    jitter_rap = _jitter_rap(periods)
    jitter_ppq5 = _jitter_ppq(periods, ppq=5)
    jitter_ddp = jitter_rap * 3  # DDP = 3 * RAP

    features["jitter_local"] = jitter_local
    features["jitter_absolute"] = jitter_absolute
    features["jitter_rap"] = jitter_rap
    features["jitter_ppq5"] = jitter_ppq5
    features["jitter_ddp"] = jitter_ddp

    # ── 4. Shimmer (amplitude variation) ──
    # Use RMS of short frames as amplitude proxy
    frame_length = int(sr * 0.025)  # 25ms frames
    hop_length = int(sr * 0.010)    # 10ms hop
    rms_frames = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_frames = rms_frames[rms_frames > 0]
    if len(rms_frames) < 2:
        rms_frames = np.array([0.05, 0.055])

    shimmer_local = _shimmer_local(rms_frames)
    shimmer_db = _shimmer_db(rms_frames)
    shimmer_apq3 = _shimmer_apq(rms_frames, apq=3)
    shimmer_apq5 = _shimmer_apq(rms_frames, apq=5)
    shimmer_apq11 = _shimmer_apq(rms_frames, apq=11)
    shimmer_dda = shimmer_apq3 * 3

    features["shimmer_local"] = shimmer_local
    features["shimmer_db"] = shimmer_db
    features["shimmer_apq3"] = shimmer_apq3
    features["shimmer_apq5"] = shimmer_apq5
    features["shimmer_apq11"] = shimmer_apq11
    features["shimmer_dda"] = shimmer_dda

    # ── 5. HNR (Harmonic to Noise Ratio) ──
    hnr = _compute_hnr(y, sr)
    features["hnr"] = hnr

    # ── 6. ZCR (Zero Crossing Rate) ──
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_std"] = float(np.std(zcr))

    # ── 7. Spectral features ──
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]

    features["spectral_centroid_mean"] = float(np.mean(spectral_centroid))
    features["spectral_centroid_std"] = float(np.std(spectral_centroid))
    features["spectral_rolloff_mean"] = float(np.mean(spectral_rolloff))
    features["spectral_rolloff_std"] = float(np.std(spectral_rolloff))
    features["spectral_bandwidth_mean"] = float(np.mean(spectral_bandwidth))
    features["spectral_bandwidth_std"] = float(np.std(spectral_bandwidth))

    # ── 8. RMS Energy ──
    rms_full = librosa.feature.rms(y=y)[0]
    features["rms_mean"] = float(np.mean(rms_full))
    features["rms_std"] = float(np.std(rms_full))

    # ── 9. Additional voicing / nonlinear features ──
    # RPDE proxy: entropy of autocorrelation
    features["rpde"] = _rpde_proxy(y, sr)
    # DFA proxy: spectral slope
    features["dfa"] = _dfa_proxy(y, sr)
    # Spread1 / Spread2 / PPE (UCI dataset features)
    features["spread1"] = float(np.log(features["pitch_std"] + 1e-6))
    features["spread2"] = float(np.log(jitter_local + 1e-6))
    features["ppe"] = float(np.log(shimmer_local + 1e-6))

    return features


# ── Helper functions ──────────────────────────────────────────────────────────

def _jitter_local(periods: np.ndarray) -> float:
    """Local jitter: mean absolute period difference / mean period."""
    if len(periods) < 2:
        return 0.0
    diffs = np.abs(np.diff(periods))
    return float(np.mean(diffs) / (np.mean(periods) + 1e-10))


def _jitter_rap(periods: np.ndarray) -> float:
    """RAP jitter: 3-point smoothed."""
    if len(periods) < 3:
        return 0.0
    smoothed = np.convolve(periods, np.ones(3) / 3, mode="valid")
    n = len(smoothed)
    diffs = np.abs(periods[1 : n + 1] - smoothed)
    return float(np.mean(diffs) / (np.mean(periods) + 1e-10))


def _jitter_ppq(periods: np.ndarray, ppq: int = 5) -> float:
    """PPQ jitter."""
    if len(periods) < ppq:
        return 0.0
    smoothed = np.convolve(periods, np.ones(ppq) / ppq, mode="valid")
    n = len(smoothed)
    half = ppq // 2
    diffs = np.abs(periods[half : half + n] - smoothed)
    return float(np.mean(diffs) / (np.mean(periods) + 1e-10))


def _shimmer_local(amplitudes: np.ndarray) -> float:
    """Local shimmer: mean absolute amplitude difference / mean amplitude."""
    if len(amplitudes) < 2:
        return 0.0
    diffs = np.abs(np.diff(amplitudes))
    return float(np.mean(diffs) / (np.mean(amplitudes) + 1e-10))


def _shimmer_db(amplitudes: np.ndarray) -> float:
    """Shimmer in dB."""
    if len(amplitudes) < 2:
        return 0.0
    ratios = amplitudes[1:] / (amplitudes[:-1] + 1e-10)
    return float(np.mean(np.abs(20 * np.log10(ratios + 1e-10))))


def _shimmer_apq(amplitudes: np.ndarray, apq: int = 3) -> float:
    """APQ shimmer."""
    if len(amplitudes) < apq:
        return 0.0
    smoothed = np.convolve(amplitudes, np.ones(apq) / apq, mode="valid")
    n = len(smoothed)
    half = apq // 2
    diffs = np.abs(amplitudes[half : half + n] - smoothed)
    return float(np.mean(diffs) / (np.mean(amplitudes) + 1e-10))


def _compute_hnr(y: np.ndarray, sr: int) -> float:
    """
    Approximate HNR using autocorrelation method.
    """
    frame_len = int(sr * 0.04)  # 40ms
    hop = int(sr * 0.01)
    hnr_vals = []
    for start in range(0, len(y) - frame_len, hop):
        frame = y[start : start + frame_len]
        if np.max(np.abs(frame)) < 1e-6:
            continue
        # Autocorrelation
        ac = np.correlate(frame, frame, mode="full")
        ac = ac[len(ac) // 2 :]
        # Find first peak after zero (fundamental period)
        min_period = int(sr / 500)  # max 500 Hz
        max_period = int(sr / 75)   # min 75 Hz
        if max_period >= len(ac):
            continue
        peak_idx = np.argmax(ac[min_period:max_period]) + min_period
        r0 = ac[0]
        r1 = ac[peak_idx]
        if r0 <= 0 or r1 <= 0:
            continue
        hnr_val = 10 * np.log10(r1 / (r0 - r1 + 1e-10) + 1e-10)
        hnr_vals.append(float(np.clip(hnr_val, -20, 40)))
    return float(np.mean(hnr_vals)) if hnr_vals else 15.0


def _rpde_proxy(y: np.ndarray, sr: int) -> float:
    """Proxy for RPDE: normalized autocorrelation entropy."""
    ac = np.correlate(y[:min(len(y), sr)], y[:min(len(y), sr)], mode="full")
    ac = np.abs(ac[len(ac) // 2 :])
    ac = ac / (ac[0] + 1e-10)
    ac = ac[:500]
    ac = np.clip(ac, 1e-10, 1)
    entropy = -np.sum(ac * np.log(ac)) / len(ac)
    return float(np.clip(entropy, 0, 1))


def _dfa_proxy(y: np.ndarray, sr: int) -> float:
    """Proxy for DFA: spectral flatness as complexity measure."""
    n = min(len(y), sr * 2)
    spec = np.abs(np.fft.rfft(y[:n]))
    spec = spec[spec > 0]
    if len(spec) == 0:
        return 0.7
    geo_mean = np.exp(np.mean(np.log(spec + 1e-10)))
    arith_mean = np.mean(spec)
    flatness = float(geo_mean / (arith_mean + 1e-10))
    return float(np.clip(flatness, 0, 1))
