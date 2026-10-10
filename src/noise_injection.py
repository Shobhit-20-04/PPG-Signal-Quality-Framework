"""
Physiologically and environmentally grounded noise injection module for PPG signals.
Generates realistic artifacts on clean baseline recordings.
"""

from typing import Dict, Callable, Tuple
import numpy as np

FS_DEFAULT = 125.0
SEVERITIES_DEFAULT = [0.5, 1.0, 1.5, 2.0]


def inject_motion_artifact(sig: np.ndarray, severity: float = 1.0, rng=None, fs: float = FS_DEFAULT) -> np.ndarray:
    """Simulates transient motion artifacts caused by sensor movement or posture change."""
    if rng is None:
        rng = np.random.default_rng()
    sig = sig.copy()
    n = len(sig)
    for _ in range(rng.integers(1, 3)):
        start = rng.integers(0, max(1, n - int(0.5 * fs)))
        dur = rng.integers(int(0.15 * fs), int(0.5 * fs))
        end = min(start + dur, n)
        amp = severity * rng.uniform(1.5, 3.0) * sig.std()
        burst = amp * np.sin(np.linspace(0, rng.uniform(2, 5) * np.pi, end - start))
        burst += rng.normal(0, 0.3 * amp, end - start)
        sig[start:end] += burst
    return sig


def inject_baseline_wander(sig: np.ndarray, severity: float = 1.0, rng=None, fs: float = FS_DEFAULT) -> np.ndarray:
    """Simulates slow respiration and vasomotor baseline drift (0.1 - 0.4 Hz)."""
    if rng is None:
        rng = np.random.default_rng()
    n = len(sig)
    t = np.arange(n) / fs
    freq = rng.uniform(0.1, 0.4)
    amp = severity * rng.uniform(1.0, 2.5) * sig.std()
    phase = rng.uniform(0, 2 * np.pi)
    return sig + amp * np.sin(2 * np.pi * freq * t + phase)


def inject_powerline_interference(sig: np.ndarray, severity: float = 1.0, rng=None, fs: float = FS_DEFAULT) -> np.ndarray:
    """Simulates 50 Hz powerline electromagnetic coupling."""
    if rng is None:
        rng = np.random.default_rng()
    n = len(sig)
    t = np.arange(n) / fs
    amp = severity * rng.uniform(0.15, 0.4) * sig.std()
    return sig + amp * np.sin(2 * np.pi * 50.0 * t + rng.uniform(0, 2 * np.pi))


def inject_muscle_noise(sig: np.ndarray, severity: float = 1.0, rng=None) -> np.ndarray:
    """Simulates electromyographic (EMG) high-frequency broadband noise."""
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0, severity * rng.uniform(0.3, 0.6) * sig.std(), len(sig))
    return sig + noise


def inject_ambient_light(sig: np.ndarray, severity: float = 1.0, rng=None) -> np.ndarray:
    """Simulates sudden environmental lighting changes or optical path shifts."""
    if rng is None:
        rng = np.random.default_rng()
    sig = sig.copy()
    n = len(sig)
    for _ in range(rng.integers(1, 3)):
        idx = rng.integers(int(0.1 * n), int(0.9 * n))
        shift = severity * rng.uniform(1.0, 2.0) * sig.std() * rng.choice([-1, 1])
        sig[idx:] += shift
    return sig


def inject_clean_jitter(sig: np.ndarray, severity: float = 1.0, rng=None) -> np.ndarray:
    """Simulates realistic tiny photodetector thermal noise floor."""
    if rng is None:
        rng = np.random.default_rng()
    return sig + rng.normal(0, 0.01 * sig.std(), len(sig))


INJECTORS: Dict[str, Callable] = {
    "Motion_Artifact": inject_motion_artifact,
    "Baseline_Wander": inject_baseline_wander,
    "Powerline_Interference": inject_powerline_interference,
    "Muscle_Noise": inject_muscle_noise,
    "Ambient_Light": inject_ambient_light,
}


def build_labeled_dataset(
    clean_signals: np.ndarray,
    rng_seed: int = 42,
    severities: list = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Constructs a balanced labeled dataset of clean signals and 5 synthetic artifact types.
    Returns:
        X_out: np.ndarray of synthesized signals
        y_out: np.ndarray of string labels
        base_out: np.ndarray of indices pointing to the original clean signal
    """
    if severities is None:
        severities = SEVERITIES_DEFAULT

    local_rng = np.random.default_rng(rng_seed)
    X_out, y_out, base_out = [], [], []

    for i, sig in enumerate(clean_signals):
        # Clean with sensor jitter
        for _ in range(len(severities)):
            X_out.append(inject_clean_jitter(sig, rng=local_rng))
            y_out.append("Clean")
            base_out.append(i)

        # Artifact classes across severity levels
        for label, fn in INJECTORS.items():
            for sev in severities:
                X_out.append(fn(sig, severity=sev, rng=local_rng))
                y_out.append(label)
                base_out.append(i)

    return np.array(X_out), np.array(y_out), np.array(base_out)
