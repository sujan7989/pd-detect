"""
Lightweight audio feature extraction — no librosa/numba dependency.
Uses only scipy + numpy for <200MB RAM footprint on free hosting.
"""

import numpy as np
import soundfile as sf
import io
import warnings
warnings.filterwarnings("ignore")

# ── scipy imports (lazy, to keep startup fast) ─────────────────────────────
from scipy.signal import resample_poly
from scipy.fft import rfft, rfftfreq


# ═══════════════════════════════════════════════════════════════════════════
#  PUBLIC ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def extract_features(audio_bytes: bytes, sr_target: int = 22050) -> dict:
    """
    Extract ~60 acoustic features from raw audio bytes.
    Returns a flat dict suitable for feeding into the UCI-mapped ML model.
    """
    y, sr = _load_audio(audio_bytes, sr_target)

    features: dict = {}

    # 1. MFCCs (13 coefficients + deltas + delta-deltas = 39 features)
    mfcc, mfcc_d, mfcc_d2 = _compute_mfcc(y, sr, n_mfcc=13)
    for i in range(13):
        features[f"mfcc_{i+1}_mean"]       = float(np.mean(mfcc[i]))
        features[f"mfcc_delta_{i+1}_mean"] = float(np.mean(mfcc_d[i]))
        features[f"mfcc_delta2_{i+1}_mean"]= float(np.mean(mfcc_d2[i]))

    # 2. Pitch (fundamental frequency via autocorrelation)
    f0 = _extract_pitch(y, sr)
    voiced = f0[f0 > 0]
    if len(voiced) < 2:
        voiced = np.array([150.0, 152.0])
    features["pitch_mean"] = float(np.mean(voiced))
    features["pitch_std"]  = float(np.std(voiced))

    # 3. Jitter (period irregularity from F0 contour)
    periods = 1.0 / voiced[voiced > 0]
    if len(periods) < 2:
        periods = np.array([1/150.0, 1/152.0])
    features["jitter_local"]    = _jitter_local(periods)
    features["jitter_absolute"] = float(np.mean(np.abs(np.diff(periods))))
    features["jitter_rap"]      = _jitter_rap(periods)
    features["jitter_ppq5"]     = _jitter_ppq(periods, 5)
    features["jitter_ddp"]      = features["jitter_rap"] * 3.0

    # 4. Shimmer (amplitude irregularity from RMS frames)
    frame_len = int(sr * 0.025)
    hop_len   = int(sr * 0.010)
    rms = _frame_rms(y, frame_len, hop_len)
    rms = rms[rms > 1e-8]
    if len(rms) < 2:
        rms = np.array([0.05, 0.052])
    features["shimmer_local"] = _shimmer_local(rms)
    features["shimmer_db"]    = _shimmer_db(rms)
    features["shimmer_apq3"]  = _shimmer_apq(rms, 3)
    features["shimmer_apq5"]  = _shimmer_apq(rms, 5)
    features["shimmer_apq11"] = _shimmer_apq(rms, 11)
    features["shimmer_dda"]   = features["shimmer_apq3"] * 3.0

    # 5. HNR
    features["hnr"] = _compute_hnr(y, sr)

    # 6. ZCR
    zcr = _zero_crossing_rate(y, frame_len, hop_len)
    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_std"]  = float(np.std(zcr))

    # 7. Spectral features
    spec_c, spec_r, spec_b = _spectral_features(y, sr)
    features["spectral_centroid_mean"]  = float(np.mean(spec_c))
    features["spectral_centroid_std"]   = float(np.std(spec_c))
    features["spectral_rolloff_mean"]   = float(np.mean(spec_r))
    features["spectral_rolloff_std"]    = float(np.std(spec_r))
    features["spectral_bandwidth_mean"] = float(np.mean(spec_b))
    features["spectral_bandwidth_std"]  = float(np.std(spec_b))

    # 8. RMS energy
    rms_full = _frame_rms(y, 2048, 512)
    features["rms_mean"] = float(np.mean(rms_full))
    features["rms_std"]  = float(np.std(rms_full))

    # 9. Nonlinear / derived features
    features["rpde"]    = _rpde_proxy(y, sr)
    features["dfa"]     = _dfa_proxy(y, sr)
    features["spread1"] = float(np.log(features["pitch_std"] + 1e-8))
    features["spread2"] = float(np.log(features["jitter_local"] + 1e-8))
    features["ppe"]     = float(np.log(features["shimmer_local"] + 1e-8))

    return features


# ═══════════════════════════════════════════════════════════════════════════
#  AUDIO LOADING
# ═══════════════════════════════════════════════════════════════════════════

def _load_audio(audio_bytes: bytes, sr_target: int) -> tuple:
    """
    Load audio from bytes supporting WAV, WebM, OGG, MP3, FLAC.
    Browser recordings come as WebM/OGG — handled via ffmpeg subprocess.
    """
    import subprocess, tempfile, os

    buf = io.BytesIO(audio_bytes)

    # ── First try soundfile (handles WAV, FLAC, OGG Vorbis, AIFF) ──
    try:
        buf.seek(0)
        y, sr = sf.read(buf)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        y = y.astype(np.float32)
        if sr != sr_target:
            gcd = int(np.gcd(sr_target, sr))
            y   = resample_poly(y, sr_target // gcd, sr // gcd).astype(np.float32)
            sr  = sr_target
        return _trim_and_validate(y, sr)
    except Exception:
        pass

    # ── Fallback: use ffmpeg to convert WebM/MP3/M4A → WAV ──
    try:
        with tempfile.NamedTemporaryFile(suffix=".input", delete=False) as tmp_in:
            tmp_in.write(audio_bytes)
            tmp_in_path = tmp_in.name

        tmp_out_path = tmp_in_path + ".wav"

        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", tmp_in_path,
                "-ar", str(sr_target),
                "-ac", "1",
                "-f", "wav",
                tmp_out_path,
            ],
            capture_output=True,
            timeout=30,
        )

        if result.returncode == 0 and os.path.exists(tmp_out_path):
            y, sr = sf.read(tmp_out_path)
            os.unlink(tmp_in_path)
            os.unlink(tmp_out_path)
            if y.ndim > 1:
                y = np.mean(y, axis=1)
            y = y.astype(np.float32)
            return _trim_and_validate(y, sr)
        else:
            os.unlink(tmp_in_path)
            if os.path.exists(tmp_out_path):
                os.unlink(tmp_out_path)
    except Exception:
        pass

    # ── Last resort: assume raw PCM int16 ──
    try:
        raw = np.frombuffer(audio_bytes, dtype=np.int16)
        if len(raw) < 100:
            raise ValueError("Too short")
        y = (raw / 32768.0).astype(np.float32)
        return _trim_and_validate(y, sr_target)
    except Exception as e:
        raise ValueError(f"Could not decode audio: {e}")


def _trim_and_validate(y: np.ndarray, sr: int) -> tuple:
    """Trim silence and validate minimum length."""
    if len(y) == 0:
        raise ValueError("Audio is empty.")
    energy = y ** 2
    mask   = energy > 1e-6
    if mask.any():
        start = int(np.argmax(mask))
        end   = int(len(mask) - np.argmax(mask[::-1]))
        if end > start:
            y = y[start:end]
    if len(y) < sr * 0.1:
        raise ValueError("Audio too short for analysis (< 0.1 s after trimming).")
    return y, sr


# ═══════════════════════════════════════════════════════════════════════════
#  MFCC  (pure numpy, no numba/librosa required)
# ═══════════════════════════════════════════════════════════════════════════

def _mel_filterbank(sr: int, n_fft: int, n_mels: int = 40,
                    f_min: float = 20.0, f_max: float = None) -> np.ndarray:
    if f_max is None:
        f_max = sr / 2.0

    def hz_to_mel(hz):   return 2595 * np.log10(1 + hz / 700)
    def mel_to_hz(mel):  return 700 * (10 ** (mel / 2595) - 1)

    mel_min  = hz_to_mel(f_min)
    mel_max  = hz_to_mel(f_max)
    mel_pts  = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts   = mel_to_hz(mel_pts)
    bin_pts  = np.floor((n_fft + 1) * hz_pts / sr).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        lo, ctr, hi = bin_pts[m-1], bin_pts[m], bin_pts[m+1]
        for k in range(lo, ctr):
            if ctr != lo:
                fb[m-1, k] = (k - lo) / (ctr - lo)
        for k in range(ctr, hi):
            if hi != ctr:
                fb[m-1, k] = (hi - k) / (hi - ctr)
    return fb


def _compute_mfcc(y: np.ndarray, sr: int, n_mfcc: int = 13):
    n_fft    = 2048
    hop_len  = 512
    n_mels   = 40
    window   = np.hanning(n_fft)
    fb       = _mel_filterbank(sr, n_fft, n_mels)

    # STFT power spectrum
    n_frames = (len(y) - n_fft) // hop_len + 1
    if n_frames < 1:
        n_frames = 1
    pow_spec = np.zeros((n_fft // 2 + 1, n_frames))
    for i in range(n_frames):
        start = i * hop_len
        frame = y[start: start + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        windowed = frame * window
        fft_out  = rfft(windowed)
        pow_spec[:, i] = np.abs(fft_out) ** 2

    # Mel spectrum → log
    mel_spec = fb @ pow_spec
    log_mel  = np.log(mel_spec + 1e-9)

    # DCT-II
    n_m = log_mel.shape[0]
    dct_mat = np.zeros((n_mfcc, n_m))
    for k in range(n_mfcc):
        dct_mat[k] = np.cos(np.pi * k / n_m * (np.arange(n_m) + 0.5))
    mfcc = dct_mat @ log_mel  # shape (n_mfcc, n_frames)

    # Delta & delta-delta
    def delta(x, w=9):
        pad = w // 2
        x_p = np.pad(x, ((0, 0), (pad, pad)), mode="edge")
        denom = 2 * sum(t**2 for t in range(1, pad+1))
        d = np.zeros_like(x)
        for t in range(x.shape[1]):
            d[:, t] = sum(n * (x_p[:, t+pad+n] - x_p[:, t+pad-n])
                          for n in range(1, pad+1)) / (denom + 1e-9)
        return d

    mfcc_d  = delta(mfcc)
    mfcc_d2 = delta(mfcc_d)
    return mfcc, mfcc_d, mfcc_d2


# ═══════════════════════════════════════════════════════════════════════════
#  PITCH EXTRACTION  (autocorrelation method)
# ═══════════════════════════════════════════════════════════════════════════

def _extract_pitch(y: np.ndarray, sr: int,
                   f_min: float = 75.0, f_max: float = 500.0) -> np.ndarray:
    frame_len = int(sr * 0.04)   # 40 ms
    hop_len   = int(sr * 0.01)   # 10 ms
    lag_min   = int(sr / f_max)
    lag_max   = int(sr / f_min)
    n_frames  = (len(y) - frame_len) // hop_len + 1
    f0 = np.zeros(max(n_frames, 1))

    for i in range(n_frames):
        frame = y[i * hop_len: i * hop_len + frame_len]
        if len(frame) < frame_len:
            break
        frame = frame - np.mean(frame)
        rms = np.sqrt(np.mean(frame ** 2))
        if rms < 1e-5:
            continue
        # Normalize
        frame = frame / (rms + 1e-10)
        # Autocorrelation via FFT
        n   = len(frame)
        fa  = rfft(frame, n=n * 2)
        ac  = np.real(np.fft.irfft(fa * np.conj(fa)))[:n]
        if ac[0] <= 0:
            continue
        ac /= ac[0]
        # Find peak in valid lag range
        lmax = min(lag_max, len(ac) - 1)
        if lmax <= lag_min:
            continue
        segment  = ac[lag_min: lmax]
        if len(segment) == 0:
            continue
        peak_idx = int(np.argmax(segment)) + lag_min
        # Lower threshold to 0.15 to detect more voiced frames
        if ac[peak_idx] > 0.15:
            f0[i] = sr / peak_idx

    return f0


# ═══════════════════════════════════════════════════════════════════════════
#  JITTER helpers
# ═══════════════════════════════════════════════════════════════════════════

def _jitter_local(p: np.ndarray) -> float:
    if len(p) < 2: return 0.0
    return float(np.mean(np.abs(np.diff(p))) / (np.mean(p) + 1e-10))

def _jitter_rap(p: np.ndarray) -> float:
    if len(p) < 3: return 0.0
    sm = np.convolve(p, np.ones(3)/3, "valid")
    n  = len(sm)
    return float(np.mean(np.abs(p[1:n+1] - sm)) / (np.mean(p) + 1e-10))

def _jitter_ppq(p: np.ndarray, q: int) -> float:
    if len(p) < q: return 0.0
    sm   = np.convolve(p, np.ones(q)/q, "valid")
    half = q // 2
    n    = len(sm)
    return float(np.mean(np.abs(p[half:half+n] - sm)) / (np.mean(p) + 1e-10))


# ═══════════════════════════════════════════════════════════════════════════
#  SHIMMER helpers
# ═══════════════════════════════════════════════════════════════════════════

def _shimmer_local(a: np.ndarray) -> float:
    if len(a) < 2: return 0.0
    return float(np.mean(np.abs(np.diff(a))) / (np.mean(a) + 1e-10))

def _shimmer_db(a: np.ndarray) -> float:
    if len(a) < 2: return 0.0
    r = a[1:] / (a[:-1] + 1e-10)
    return float(np.mean(np.abs(20 * np.log10(r + 1e-10))))

def _shimmer_apq(a: np.ndarray, q: int) -> float:
    if len(a) < q: return 0.0
    sm   = np.convolve(a, np.ones(q)/q, "valid")
    half = q // 2
    n    = len(sm)
    return float(np.mean(np.abs(a[half:half+n] - sm)) / (np.mean(a) + 1e-10))


# ═══════════════════════════════════════════════════════════════════════════
#  FRAME RMS
# ═══════════════════════════════════════════════════════════════════════════

def _frame_rms(y: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    n_frames = (len(y) - frame_len) // hop_len + 1
    if n_frames < 1: return np.array([np.sqrt(np.mean(y**2))])
    out = np.zeros(n_frames)
    for i in range(n_frames):
        frame = y[i*hop_len: i*hop_len + frame_len]
        out[i] = np.sqrt(np.mean(frame**2) + 1e-10)
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  HNR
# ═══════════════════════════════════════════════════════════════════════════

def _compute_hnr(y: np.ndarray, sr: int) -> float:
    """
    Compute HNR using the cepstrum-based method.
    More accurate than pure autocorrelation for synthetic/real speech.
    Healthy voices: 20-33 dB. PD voices: 8-25 dB.
    """
    frame_len = int(sr * 0.04)
    hop_len   = int(sr * 0.02)
    hnr_vals  = []

    for i in range(0, len(y) - frame_len, hop_len):
        frame = y[i: i + frame_len].copy()
        frame -= np.mean(frame)
        rms = np.sqrt(np.mean(frame ** 2))
        if rms < 1e-5:
            continue

        # Window
        frame *= np.hanning(len(frame))
        frame /= (np.max(np.abs(frame)) + 1e-10)

        n    = len(frame)
        # Power spectrum
        spec = np.abs(np.fft.rfft(frame, n=n)) ** 2
        spec = np.maximum(spec, 1e-10)

        # Cepstrum
        log_spec = np.log(spec)
        cepst    = np.real(np.fft.irfft(log_spec))

        # Find fundamental period in cepstrum (between 2ms and 13ms)
        t_min = int(sr * 0.002)   # 2ms
        t_max = int(sr * 0.013)   # 13ms
        if t_max >= len(cepst):
            t_max = len(cepst) - 1
        if t_max <= t_min:
            continue

        seg      = np.abs(cepst[t_min:t_max])
        peak_val = np.max(seg)
        # Estimate HNR from cepstral peak strength
        # Strong peak = high harmonic content = high HNR
        noise_floor = np.median(np.abs(cepst[5: t_min]))
        if noise_floor < 1e-8:
            continue
        ratio = peak_val / (noise_floor + 1e-10)
        # Convert to dB scale (empirically calibrated)
        hnr_est = 20.0 * np.log10(ratio + 1e-6)
        # Clamp to realistic speech range
        hnr_vals.append(float(np.clip(hnr_est, 0.0, 40.0)))

    if not hnr_vals:
        return 20.0  # default healthy-ish value

    return float(np.median(hnr_vals))


# ═══════════════════════════════════════════════════════════════════════════
#  ZCR
# ═══════════════════════════════════════════════════════════════════════════

def _zero_crossing_rate(y: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    n_frames = (len(y) - frame_len) // hop_len + 1
    if n_frames < 1: return np.array([0.0])
    out = np.zeros(n_frames)
    for i in range(n_frames):
        frame   = y[i*hop_len: i*hop_len + frame_len]
        out[i]  = np.sum(np.abs(np.diff(np.sign(frame)))) / (2.0 * frame_len)
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  SPECTRAL FEATURES
# ═══════════════════════════════════════════════════════════════════════════

def _spectral_features(y: np.ndarray, sr: int):
    n_fft    = 2048
    hop_len  = 512
    window   = np.hanning(n_fft)
    n_frames = (len(y) - n_fft) // hop_len + 1
    if n_frames < 1: n_frames = 1
    freqs = rfftfreq(n_fft, 1.0 / sr)

    centroids  = []
    rolloffs   = []
    bandwidths = []

    for i in range(n_frames):
        start = i * hop_len
        frame = y[start: start + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        mag = np.abs(rfft(frame * window))
        mag_sum = np.sum(mag) + 1e-10

        # Centroid
        centroid = float(np.sum(freqs * mag) / mag_sum)
        centroids.append(centroid)

        # Rolloff (85%)
        cumsum = np.cumsum(mag)
        thresh = 0.85 * cumsum[-1]
        idx    = np.searchsorted(cumsum, thresh)
        rolloffs.append(float(freqs[min(idx, len(freqs)-1)]))

        # Bandwidth
        bw = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * mag) / mag_sum))
        bandwidths.append(bw)

    return np.array(centroids), np.array(rolloffs), np.array(bandwidths)


# ═══════════════════════════════════════════════════════════════════════════
#  NONLINEAR PROXIES
# ═══════════════════════════════════════════════════════════════════════════

def _rpde_proxy(y: np.ndarray, sr: int) -> float:
    """Entropy of normalized autocorrelation — proxy for RPDE."""
    n   = min(len(y), sr)
    fa  = rfft(y[:n])
    ac  = np.abs(np.fft.irfft(fa * np.conj(fa)))[:500]
    ac  = np.clip(ac / (ac[0] + 1e-10), 1e-10, 1)
    ent = -np.sum(ac * np.log(ac)) / len(ac)
    return float(np.clip(ent, 0, 1))

def _dfa_proxy(y: np.ndarray, sr: int) -> float:
    """Spectral flatness — proxy for DFA."""
    n   = min(len(y), sr * 2)
    mag = np.abs(rfft(y[:n])) + 1e-10
    geo = np.exp(np.mean(np.log(mag)))
    arith = np.mean(mag)
    return float(np.clip(geo / (arith + 1e-10), 0, 1))
