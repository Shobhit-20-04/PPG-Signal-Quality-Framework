"""
Standalone Demonstration Script for PPG Signal Quality & Restoration Framework
==============================================================================

This script demonstrates the end-to-end pipeline:
1. Ingests or synthesizes a clean PPG pulse waveform (16s @ 125 Hz).
2. Simulates 5 distinct physiological & environmental noise artifacts.
3. Extracts the 7 Signal Quality Index (SQI) features.
4. Cleans the corrupted signals using artifact-tailored filtering.
5. Computes and displays quantitative performance metrics (Pearson r, RMSE, SNR dB).
"""

import os
import numpy as np
import pandas as pd

import src
from src.features import extract_sqi_features, FEATURE_NAMES
from src.noise_injection import INJECTORS, inject_clean_jitter
from src.restoration import clean_ppg, minimal_reference, normalize_waveform_display
from src.evaluation import evaluate_signal_pair


def generate_synthetic_clean_ppg(duration_sec: float = 16.0, fs: float = 125.0, hr_bpm: float = 72.0) -> np.ndarray:
    """
    Synthesizes an idealized cardiovascular PPG signal with systolic and diastolic peaks.
    Used if PPG_Dataset.csv is unavailable.
    """
    n_samples = int(duration_sec * fs)
    t = np.arange(n_samples) / fs
    f0 = hr_bpm / 60.0  # fundamental heart rate frequency (~1.2 Hz)
    
    # Asymmetric pulse wave with systolic peak, dicrotic notch, and diastolic wave
    sig = (
        1.0 * np.sin(2 * np.pi * f0 * t) +
        0.5 * np.sin(4 * np.pi * f0 * t + 0.8) +
        0.25 * np.sin(6 * np.pi * f0 * t + 1.4) +
        0.12 * np.sin(8 * np.pi * f0 * t + 2.0)
    )
    sig = sig - np.mean(sig)
    return sig / np.std(sig)


def get_sample_signal() -> np.ndarray:
    """Attempts to load first signal from PPG_Dataset.csv, falling back to synthetic PPG."""
    csv_path = "PPG_Dataset.csv"
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, nrows=1)
            sig_cols = [c for c in df.columns if c != 'Label']
            sig = df[sig_cols].values[0].astype(float)
            if len(sig) == 2000:
                print(f"Loaded 2000-sample signal from '{csv_path}'.")
                return (sig - np.mean(sig)) / np.std(sig)
        except Exception as e:
            print(f"Notice: Could not load from CSV ({e}), generating synthetic pulse wave.")
    
    print("Generating physiological synthetic PPG waveform (16s @ 125 Hz).")
    return generate_synthetic_clean_ppg()


def run_demo():
    print("=" * 80)
    print("PPG Signal Quality Assessment and Artifact Removal Framework - Demo")
    print("=" * 80)

    clean_base = get_sample_signal()
    ref_signal = minimal_reference(clean_base)
    rng = np.random.default_rng(42)

    test_cases = {
        "Clean": inject_clean_jitter(clean_base, rng=rng),
        "Baseline_Wander": INJECTORS["Baseline_Wander"](clean_base, severity=1.0, rng=rng),
        "Motion_Artifact": INJECTORS["Motion_Artifact"](clean_base, severity=1.0, rng=rng),
        "Muscle_Noise": INJECTORS["Muscle_Noise"](clean_base, severity=1.0, rng=rng),
        "Powerline_Interference": INJECTORS["Powerline_Interference"](clean_base, severity=1.0, rng=rng),
        "Ambient_Light": INJECTORS["Ambient_Light"](clean_base, severity=1.0, rng=rng),
    }

    results = []

    print("\n[1] Processing Test Artifacts & Calculating SQI Features...\n")
    for label, corrupted in test_cases.items():
        # Feature extraction
        feats = extract_sqi_features(corrupted)
        feat_dict = {name: round(val, 3) for name, val in zip(FEATURE_NAMES, feats)}

        # Targeted restoration
        restored = clean_ppg(corrupted, predicted_label=label)

        # Quantitative evaluation against minimal baseline reference
        metrics = evaluate_signal_pair(ref_signal, corrupted, restored)

        delta_snr = metrics['snr_after'] - metrics['snr_before']
        results.append({
            "Artifact": label,
            "Corr Before": f"{metrics['corr_before']:.3f}",
            "Corr After": f"{metrics['corr_after']:.3f}",
            "RMSE Before": f"{metrics['rmse_before']:.3f}",
            "RMSE After": f"{metrics['rmse_after']:.3f}",
            "SNR Before (dB)": f"{metrics['snr_before']:.2f}",
            "SNR After (dB)": f"{metrics['snr_after']:.2f}",
            "Delta SNR (dB)": f"{delta_snr:+.2f}",
        })

        print(f"--> {label}:")
        print(f"    SQI Features: {feat_dict}")
        print(f"    Restoration : SNR {metrics['snr_before']:.2f} dB -> {metrics['snr_after']:.2f} dB "
              f"(Pearson r: {metrics['corr_before']:.3f} -> {metrics['corr_after']:.3f})\n")

    summary_df = pd.DataFrame(results)
    print("=" * 80)
    print("SUMMARY PERFORMANCE TABLE (All Artifact Classes)")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)
    print("Demo completed successfully! Waveforms restored with targeted filtering.")


if __name__ == "__main__":
    run_demo()
