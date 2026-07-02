"""Generate demo WAV files for testing PD detection."""
import numpy as np
import wave
import os

OUTPUT_DIR = os.path.dirname(__file__)
SR = 22050  # Sample rate

def save_wav(filename, signal, sr=SR):
    """Save signal as 16-bit WAV."""
    signal = signal / (np.max(np.abs(signal)) + 1e-9) * 0.8
    signal_int16 = (signal * 32767).astype(np.int16)
    path = os.path.join(OUTPUT_DIR, filename)
    with wave.open(path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(signal_int16.tobytes())
    print(f"  Created: {filename} ({len(signal)/sr:.1f}s)")
    return path

def make_healthy_voice(f0=165, duration=4.0, seed=42):
    """Clean sustained vowel - healthy voice simulation."""
    np.random.seed(seed)
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    
    # Clean harmonic signal
    signal = 0.5 * np.sin(2 * np.pi * f0 * t)
    signal += 0.2 * np.sin(2 * np.pi * 2 * f0 * t)
    signal += 0.1 * np.sin(2 * np.pi * 3 * f0 * t)
    signal += 0.05 * np.sin(2 * np.pi * 4 * f0 * t)
    
    # Very low noise floor (healthy)
    signal += 0.015 * np.random.randn(len(t))
    
    # Smooth envelope
    envelope = np.ones_like(t)
    fade = int(0.05 * SR)
    envelope[:fade] = np.linspace(0, 1, fade)
    envelope[-fade:] = np.linspace(1, 0, fade)
    signal *= envelope
    return signal

def make_pd_voice(f0=145, duration=4.0, severity='moderate', seed=42):
    """Voice with PD-like characteristics (jitter, shimmer, pitch tremor, noise).
    Uses clinically realistic values: jitter 0.5-3%, pitch_std 10-30Hz.
    """
    np.random.seed(seed)
    n_samples = int(SR * duration)
    t = np.arange(n_samples) / SR
    
    # PD parameters by severity (clinically realistic ranges)
    params = {
        'mild':     {'jitter': 0.012, 'shimmer': 0.06, 'noise': 0.04, 'tremor': 8.0},
        'moderate': {'jitter': 0.022, 'shimmer': 0.10, 'noise': 0.08, 'tremor': 15.0},
        'severe':   {'jitter': 0.035, 'shimmer': 0.15, 'noise': 0.12, 'tremor': 25.0},
    }
    p = params.get(severity, params['moderate'])
    
    # Pre-generate slow pitch tremor (random walk, clipped)
    # tremor parameter = target pitch_std in Hz
    n_cycles_est = int(n_samples / (SR / f0)) + 10
    tremor_raw = np.cumsum(np.random.randn(n_cycles_est))
    tremor_raw = tremor_raw - np.mean(tremor_raw)
    # Scale to target pitch_std (in Hz), then convert to period offset
    tremor_std = np.std(tremor_raw) + 1e-9
    tremor_hz = tremor_raw / tremor_std * p['tremor']  # Hz variation
    tremor_hz = np.clip(tremor_hz, -p['tremor']*2, p['tremor']*2)
    # Convert Hz variation to period variation: delta_T = -delta_f / f0^2
    tremor_period = -tremor_hz / (f0 ** 2)
    
    # Build phase with per-cycle jitter AND pitch tremor
    base_period = 1.0 / f0
    phase = np.zeros(n_samples)
    current_phase = 0.0
    i = 0
    cycle_idx = 0
    while i < n_samples:
        # Per-cycle jitter (cycle-to-cycle variation)
        delta_T = np.random.randn() * base_period * p['jitter']
        delta_T = np.clip(delta_T, -3*base_period*p['jitter'], 3*base_period*p['jitter'])
        # Add pitch tremor (slow F0 variation)
        tremor_offset = tremor_period[min(cycle_idx, len(tremor_period)-1)]
        this_period = base_period + delta_T + tremor_offset
        this_period = max(1.0/500, min(this_period, 1.0/50))  # Clamp to valid range
        samples_in_cycle = min(max(1, int(this_period * SR)), n_samples - i)
        cycle_phase = 2 * np.pi * (np.arange(samples_in_cycle) / samples_in_cycle)
        phase[i:i+samples_in_cycle] = current_phase + cycle_phase
        current_phase += 2 * np.pi
        i += samples_in_cycle
        cycle_idx += 1
    
    # Build harmonic signal
    signal = 0.5 * np.sin(phase)
    signal += 0.2 * np.sin(2 * phase)
    signal += 0.1 * np.sin(3 * phase)
    signal += 0.05 * np.sin(4 * phase)
    
    # Amplitude shimmer (cycle-to-cycle amplitude variation)
    amp_noise = np.random.randn(n_samples)
    kernel_size = max(1, int(SR / f0))
    amp_noise = np.convolve(amp_noise, np.ones(kernel_size)/kernel_size, 'same')
    amp_mod = 1.0 + p['shimmer'] * amp_noise / (np.std(amp_noise) + 1e-9)
    signal *= amp_mod
    
    # Add noise (breathiness)
    signal += p['noise'] * np.random.randn(n_samples)
    
    # Envelope
    envelope = np.ones_like(t)
    fade = int(0.05 * SR)
    envelope[:fade] = np.linspace(0, 1, fade)
    envelope[-fade:] = np.linspace(1, 0, fade)
    signal *= envelope
    return signal

print("=" * 60)
print("  PD DETECT - DEMO AUDIO FILES")
print("=" * 60)

# Healthy voices
print("\n--- Healthy Voices ---")
make_healthy_voice(f0=165, duration=4.0, seed=42)
save_wav("demo_healthy_male_165hz.wav", make_healthy_voice(165, 4.0, 42))
save_wav("demo_healthy_female_220hz.wav", make_healthy_voice(220, 4.0, 43))
save_wav("demo_healthy_low_pitch_120hz.wav", make_healthy_voice(120, 4.0, 44))

# PD-like voices
print("\n--- PD-Like Voices ---")
save_wav("demo_pd_mild.wav", make_pd_voice(150, 4.0, 'mild', 45))
save_wav("demo_pd_moderate.wav", make_pd_voice(145, 4.0, 'moderate', 46))
save_wav("demo_pd_severe.wav", make_pd_voice(140, 4.0, 'severe', 47))

print("\n" + "=" * 60)
print(f"  Generated 6 demo files in: {OUTPUT_DIR}")
print("=" * 60)
print("\nTest with curl:")
print("  curl -X POST https://pd-detect-api.onrender.com/analyze -F 'file=@demo_healthy_male_165hz.wav'")
print("  curl -X POST https://pd-detect-api.onrender.com/analyze -F 'file=@demo_pd_severe.wav'")
