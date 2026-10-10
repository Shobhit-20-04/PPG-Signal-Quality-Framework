"""
Feature extraction module for PPG Signal Quality Index (SQI).
Extracts statistical and frequency-domain morphological markers.
"""

import numpy as np
from scipy.stats import kurtosis, skew
from scipy.signal import welch

FS_DEFAULT = 125.0

FEATURE_NAMES = [
    "Kurtosis",
    "Skewness",
    "AmplitudeEntropy",
    "ZeroCrossingRate",
    "AutocorrLag1",
    "LowFreqEnergyRatio",
    "StdDev"
]


def zero_crossing_rate(sig: np.ndarray) -> float:
    """Calculate the zero-crossing rate of the mean-centered signal."""
    signs = np.sign(sig - np.mean(sig))
    signs[signs == 0] = 1
    return float(np.sum(np.abs(np.diff(signs)) > 0) / len(sig))


def autocorr_lag1(sig: np.ndarray) -> float:
    """Calculate the normalized autocorrelation at lag 1."""
    centered = sig - np.mean(sig)
    denom = np.sum(centered ** 2)
    return 0.0 if denom < 1e-12 else float(np.sum(centered[:-1] * centered[1:]) / denom)


def low_freq_energy_ratio(sig: np.ndarray, cutoff: float = 0.5, fs: float = FS_DEFAULT) -> float:
    """Calculate the ratio of power spectral density below cutoff (Hz) to total power."""
    nperseg = min(512, len(sig))
    freqs, psd = welch(sig, fs=fs, nperseg=nperseg)
    total = np.sum(psd)
    return 0.0 if total < 1e-12 else float(np.sum(psd[freqs <= cutoff]) / total)


def amplitude_entropy(sig: np.ndarray, n_bins: int = 32) -> float:
    """Compute the Shannon entropy of the amplitude distribution."""
    hist, _ = np.histogram(sig, bins=n_bins, density=False)
    total = hist.sum()
    if total == 0:
        return 0.0
    probs = hist / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def extract_sqi_features(sig: np.ndarray, fs: float = FS_DEFAULT) -> list:
    """
    Extract the 7 Signal Quality Index (SQI) features:
    1. Kurtosis: Peakedness of the distribution
    2. Skewness: Asymmetry of the PPG pulse wave
    3. AmplitudeEntropy: Complexity / predictability of amplitude values
    4. ZeroCrossingRate: Indicator of high-frequency noise or oscillation
    5. AutocorrLag1: Periodicity and continuity of the cardiac rhythm
    6. LowFreqEnergyRatio: Dominance of respiratory / drift baseline fluctuations
    7. StdDev: Amplitude dispersion
    """
    sig = np.asarray(sig, dtype=float)
    return [
        float(kurtosis(sig)),
        float(skew(sig)),
        amplitude_entropy(sig),
        zero_crossing_rate(sig),
        autocorr_lag1(sig),
        low_freq_energy_ratio(sig, cutoff=0.5, fs=fs),
        float(np.std(sig))
    ]
