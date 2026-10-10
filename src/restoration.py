"""
Targeted PPG signal restoration module.
Applies specialized biomedical signal filtering tailored to the detected artifact.
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, detrend, savgol_filter, medfilt
from scipy.interpolate import interp1d
import pywt

FS_DEFAULT = 125.0


def mad(x: np.ndarray) -> float:
    """Median Absolute Deviation, scaled to match standard deviation for Gaussian noise."""
    x = np.asarray(x, dtype=float)
    return float(np.median(np.abs(x - np.median(x))) / 0.6745)


def butter_filter(sig: np.ndarray, cutoff, order: int, btype: str, fs: float = FS_DEFAULT) -> np.ndarray:
    """Zero-phase Butterworth digital filter."""
    nyq = 0.5 * fs
    cutoff = cutoff / nyq if isinstance(cutoff, (int, float)) else [c / nyq for c in cutoff]
    b, a = butter(order, cutoff, btype=btype)
    return filtfilt(b, a, sig)


def highpass_filter(sig: np.ndarray, cutoff: float = 0.3, order: int = 4, fs: float = FS_DEFAULT) -> np.ndarray:
    """Zero-phase highpass Butterworth filter."""
    return butter_filter(sig, cutoff, order, 'high', fs=fs)


def lowpass_filter(sig: np.ndarray, cutoff: float = 8.0, order: int = 4, fs: float = FS_DEFAULT) -> np.ndarray:
    """Zero-phase lowpass Butterworth filter."""
    return butter_filter(sig, cutoff, order, 'low', fs=fs)


def notch_filter(sig: np.ndarray, freq: float = 50.0, Q: float = 30.0, fs: float = FS_DEFAULT) -> np.ndarray:
    """Zero-phase IIR notch filter at specified frequency (e.g. 50 Hz powerline interference)."""
    b, a = iirnotch(freq / (fs / 2), Q)
    return filtfilt(b, a, sig)


def wavelet_denoise(sig: np.ndarray, wavelet: str = 'db6', level: int = 4) -> np.ndarray:
    """
    Wavelet soft-thresholding denoising using Donoho-Johnstone universal threshold
    estimated strictly from the finest detail band.
    """
    max_level = pywt.dwt_max_level(len(sig), pywt.Wavelet(wavelet).dec_len)
    level = min(level, max_level) if max_level > 0 else 1
    coeffs = pywt.wavedec(sig, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # finest detail band estimate
    uthresh = sigma * np.sqrt(2 * np.log(len(sig)))
    coeffs[1:] = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet)[:len(sig)]


def median_filter(sig: np.ndarray) -> np.ndarray:
    """Median filter for impulsive noise reduction."""
    return medfilt(sig, kernel_size=3)


def savgol_smooth(sig: np.ndarray, window: int = 81, polyorder: int = 3) -> np.ndarray:
    """Savitzky-Golay polynomial smoothing filter preserving peak morphology."""
    window = min(window, len(sig) if len(sig) % 2 == 1 else len(sig) - 1)
    window = max(window, polyorder + 1 + (polyorder % 2 == 0))
    if window % 2 == 0:
        window += 1
    return savgol_filter(sig, window, polyorder)


def spike_suppressor(sig: np.ndarray, window_size: int = 11, n_sigmas: float = 3.0) -> np.ndarray:
    """Replaces localized amplitude spikes with local running median."""
    sig = sig.copy()
    s = pd.Series(sig)
    rm = s.rolling(window_size, center=True).median()
    rmad = s.rolling(window_size, center=True).apply(mad, raw=True)
    thresh = n_sigmas * rmad.fillna(rmad.median())
    diff = np.abs(s - rm)
    mask = (diff > thresh).values & rm.notna().values
    sig[mask] = rm.values[mask]
    return sig


def despike_global(sig: np.ndarray, n_sigmas: float = 2.5, pad: int = 4) -> np.ndarray:
    """Detects and interpolates global amplitude spikes."""
    sig = sig.copy()
    med, m = np.median(sig), mad(sig)
    if m < 1e-9:
        m = np.std(sig) + 1e-9
    mask = (np.abs(sig - med) / m) > n_sigmas
    dil = mask.copy()
    for i in np.where(mask)[0]:
        lo, hi = max(0, i - pad), min(len(sig), i + pad + 1)
        dil[lo:hi] = True
    if dil.any() and (~dil).sum() >= 2:
        idx = np.arange(len(sig))
        f = interp1d(idx[~dil], sig[~dil], bounds_error=False, fill_value='extrapolate')
        sig[dil] = f(idx[dil])
    return sig


def clean_ppg(sig: np.ndarray, predicted_label: str, fs: float = FS_DEFAULT) -> np.ndarray:
    """
    Applies targeted restoration according to the classified artifact type:
    - Motion_Artifact: Global despiking followed by wavelet soft thresholding
    - Baseline_Wander: Highpass filtering (0.6 Hz), median filtering, and Savitzky-Golay smoothing
    - Ambient_Light: Mean centering, median filtering, Savitzky-Golay smoothing, and spike suppression
    - Powerline_Interference: IIR notch filter at 50 Hz
    - Muscle_Noise: Lowpass Butterworth filter (8 Hz), median filtering, and Savitzky-Golay smoothing
    - Clean: Detrending and light 0.3 Hz baseline correction (preventing overprocessing)
    """
    sig = detrend(np.asarray(sig, dtype=float))
    cleaned = highpass_filter(sig, cutoff=0.3, fs=fs)

    if predicted_label == 'Motion_Artifact':
        return wavelet_denoise(despike_global(cleaned))
    elif predicted_label == 'Baseline_Wander':
        cleaned = highpass_filter(cleaned, cutoff=0.6, order=8, fs=fs)
        cleaned = median_filter(cleaned)
        return savgol_smooth(cleaned)
    elif predicted_label == 'Ambient_Light':
        cleaned = cleaned - np.mean(cleaned)
        cleaned = median_filter(cleaned)
        cleaned = savgol_smooth(cleaned)
        return spike_suppressor(cleaned)
    elif predicted_label == 'Powerline_Interference':
        return notch_filter(cleaned, freq=50.0, fs=fs)
    elif predicted_label == 'Muscle_Noise':
        cleaned = lowpass_filter(cleaned, cutoff=8.0, fs=fs)
        cleaned = median_filter(cleaned)
        return savgol_smooth(cleaned)
    else:
        return cleaned


def minimal_reference(true_clean_sig: np.ndarray, fs: float = FS_DEFAULT) -> np.ndarray:
    """The true signal with ONLY the baseline correction every class receives."""
    return highpass_filter(detrend(np.asarray(true_clean_sig, dtype=float)), cutoff=0.3, fs=fs)


def matched_reference(true_clean_sig: np.ndarray, predicted_label: str, fs: float = FS_DEFAULT) -> np.ndarray:
    """Reference built by passing the clean signal through the exact same filter chain."""
    return clean_ppg(true_clean_sig, predicted_label, fs=fs)
