"""
Quantitative restoration metrics for biomedical signal quality.
"""

from typing import Dict
import numpy as np


def corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation coefficient between two 1-D signals."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Root Mean Square Error between two 1-D signals."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def snr_db(clean_ref: np.ndarray, signal: np.ndarray) -> float:
    """Signal-to-Noise Ratio (dB) taking clean_ref as the reference signal."""
    clean_ref = np.asarray(clean_ref, dtype=float)
    signal = np.asarray(signal, dtype=float)
    noise = signal - clean_ref
    p_sig = np.mean(clean_ref ** 2)
    p_noise = np.mean(noise ** 2)
    if p_noise < 1e-12:
        return 100.0
    return float(10.0 * np.log10(p_sig / p_noise))


def evaluate_signal_pair(clean_ref: np.ndarray, noisy_sig: np.ndarray, cleaned_sig: np.ndarray) -> Dict[str, float]:
    """Computes before-and-after correlation, RMSE, and SNR (dB)."""
    return {
        "corr_before": corr(clean_ref, noisy_sig),
        "corr_after": corr(clean_ref, cleaned_sig),
        "rmse_before": rmse(clean_ref, noisy_sig),
        "rmse_after": rmse(clean_ref, cleaned_sig),
        "snr_before": snr_db(clean_ref, noisy_sig),
        "snr_after": snr_db(clean_ref, cleaned_sig)
    }
