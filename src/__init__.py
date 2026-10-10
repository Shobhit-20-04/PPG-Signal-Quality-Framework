"""
PPG Signal Quality Assessment and Artifact Removal Framework
============================================================

A comprehensive biomedical signal processing and machine learning pipeline for:
1. Detecting artifact types in PPG signals (Clean, Baseline Wander, Motion Artifact,
   Muscle Noise, Powerline Interference, Ambient Light).
2. Applying targeted, artifact-specific signal restoration.
3. Quantitatively evaluating signal recovery (Correlation, RMSE, SNR dB).
4. Cross-dataset validation and latency benchmarking.
"""

from .features import extract_sqi_features, FEATURE_NAMES
from .noise_injection import (
    inject_motion_artifact,
    inject_baseline_wander,
    inject_powerline_interference,
    inject_muscle_noise,
    inject_ambient_light,
    inject_clean_jitter,
    build_labeled_dataset,
    INJECTORS
)
from .restoration import (
    clean_ppg,
    minimal_reference,
    matched_reference,
    highpass_filter,
    lowpass_filter,
    notch_filter,
    wavelet_denoise
)
from .evaluation import corr, rmse, snr_db, evaluate_signal_pair

__all__ = [
    "extract_sqi_features",
    "FEATURE_NAMES",
    "inject_motion_artifact",
    "inject_baseline_wander",
    "inject_powerline_interference",
    "inject_muscle_noise",
    "inject_ambient_light",
    "inject_clean_jitter",
    "build_labeled_dataset",
    "INJECTORS",
    "clean_ppg",
    "minimal_reference",
    "matched_reference",
    "highpass_filter",
    "lowpass_filter",
    "notch_filter",
    "wavelet_denoise",
    "corr",
    "rmse",
    "snr_db"
]
