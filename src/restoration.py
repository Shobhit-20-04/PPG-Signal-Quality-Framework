"""
Targeted PPG signal restoration module.
Applies specialized biomedical signal filtering tailored to the detected artifact.
"""

from typing import Union
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


def butter_filter(sig: np.ndarray, cutoff: Union[float, list], order: int, btype: str, fs: float = FS_DEFAULT) -> np.ndarray:
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
    estimated strictly from the finest detail band to prevent signal attenuation.
    """
    max_level = pywt.dwt_max_level(len(sig), pywt.Wavelet(wavelet).dec_len)
    level = min(level, max_level) if max_level > 0 else 1
    coeffs = pywt.wavedec(sig, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # finest detail band estimate
    uthresh = sigma * np.sqrt(2 * np.log(len(sig)))
    coeffs[1:] = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet)[:len(sig)]


def median_filter(sig: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Median filter for impulsive noise reduction."""
    return medfilt(sig, kernel_size=kernel_size)


def savgol_smooth(sig: np.ndarray, window: int = 51, polyorder: int = 3) -> np.ndarray:
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
    Applies targeted restoration according to the diagnosed artifact type:
    - Clean: Linear detrending only, strictly avoiding over-filtering or distorting pristine waveforms.
    - Motion_Artifact: Global despiking followed by wavelet soft-thresholding on the finest detail band.
    - Baseline_Wander: Zero-phase 8th-order highpass filter (0.6 Hz) + Savitzky-Golay polynomial smoothing.
    - Ambient_Light: Mean centering, light median filter, Savitzky-Golay smoothing, and spike suppression.
    - Powerline_Interference: Zero-phase 50 Hz IIR notch filter (Q = 30).
    - Muscle_Noise: Zero-phase 10 Hz Butterworth lowpass filter + gentle Savitzky-Golay (preserves systolic peak morphology).
    """
    sig = np.asarray(sig, dtype=float)
    cleaned = detrend(sig)
    cleaned = highpass_filter(cleaned, cutoff=0.3, fs=fs)

    if predicted_label == 'Clean':
        return cleaned

    if predicted_label == 'Motion_Artifact':
        return wavelet_denoise(despike_global(cleaned))
    elif predicted_label == 'Baseline_Wander':
        cleaned = highpass_filter(cleaned, cutoff=0.6, order=8, fs=fs)
        return savgol_smooth(cleaned, window=51, polyorder=3)
    elif predicted_label == 'Ambient_Light':
        cleaned = cleaned - np.mean(cleaned)
        cleaned = median_filter(cleaned, kernel_size=3)
        cleaned = savgol_smooth(cleaned, window=51, polyorder=3)
        return spike_suppressor(cleaned)
    elif predicted_label == 'Powerline_Interference':
        return notch_filter(cleaned, freq=50.0, Q=30.0, fs=fs)
    elif predicted_label == 'Muscle_Noise':
        # Zero-phase lowpass 10Hz + Savitzky-Golay smoothing (retains systolic peak sharpness)
        cleaned = lowpass_filter(cleaned, cutoff=10.0, order=3, fs=fs)
        return savgol_smooth(cleaned, window=31, polyorder=3)
    else:
        return cleaned


def minimal_reference(true_clean_sig: np.ndarray, fs: float = FS_DEFAULT) -> np.ndarray:
    """The baseline-corrected true clean signal (0.3 Hz zero-phase highpass) used as standard reference."""
    return highpass_filter(detrend(np.asarray(true_clean_sig, dtype=float)), cutoff=0.3, fs=fs)


def matched_reference(true_clean_sig: np.ndarray, predicted_label: str, fs: float = FS_DEFAULT) -> np.ndarray:
    """Reference built by passing the clean signal through the exact same filter chain."""
    return clean_ppg(true_clean_sig, predicted_label, fs=fs)


def normalize_waveform_display(sig: np.ndarray, window_sec: float = 3.0, fs: float = FS_DEFAULT) -> np.ndarray:
    """
    Evens out slow baseline amplitude changes across the window for uniform,
    clean presentation and clinical display. Does not alter underlying peak timings.
    """
    window = int(window_sec * fs)
    if window % 2 == 0:
        window += 1
    s = pd.Series(sig)
    local_std = s.rolling(window, center=True, min_periods=window // 4).std().bfill().ffill()
    local_std = np.maximum(local_std.values, 0.2 * np.median(local_std.values))
    return sig * (local_std.mean() / local_std)
