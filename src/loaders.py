"""
Dataset ingestion and patient-aware group splitting module for PPG signals.
Supports multi-source loading:
- Original project CSV (PPG_Dataset.csv)
- BUT PPG (PhysioNet wfdb format)
- Cuff-less Blood Pressure Estimation (MIMIC-II derivative .mat files)
- PPG-DaLiA (wrist-worn sensor during sitting activities .pkl format)
"""

import os
import glob
from fractions import Fraction
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from scipy.signal import resample_poly

FS = 125.0
SEG_LEN = 2000  # 16 seconds at 125 Hz


def chunk_to_segments(sig_1d: np.ndarray, seg_len: int = SEG_LEN) -> np.ndarray:
    """Cut a long 1-D signal into back-to-back segments of exactly seg_len samples."""
    n_segs = len(sig_1d) // seg_len
    if n_segs == 0:
        return np.empty((0, seg_len))
    return np.asarray(sig_1d[:n_segs * seg_len], dtype=float).reshape(n_segs, seg_len)


def resample_to_125(sig_1d: np.ndarray, original_fs: float) -> np.ndarray:
    """Polyphase resampling to 125 Hz with edge-safe line padding."""
    sig_1d = np.asarray(sig_1d, dtype=float)
    if original_fs is None or not np.isfinite(original_fs) or original_fs <= 0:
        raise ValueError(f"Invalid sampling rate: {original_fs}")
    if abs(original_fs - FS) < 1e-9:
        return sig_1d
    ratio = Fraction(FS / original_fs).limit_denominator(1000)
    return resample_poly(sig_1d, ratio.numerator, ratio.denominator, padtype='line')


def finite_runs(x: np.ndarray, min_len: int) -> List[np.ndarray]:
    """Contiguous stretches of signal without NaN or inf."""
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    if ok.all():
        return [x] if len(x) >= min_len else []
    d = np.diff(np.concatenate([[0], ok.astype(int), [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    return [x[s:e] for s, e in zip(starts, ends) if e - s >= min_len]


def _empty() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.empty((0, SEG_LEN)), np.empty(0, dtype=object), np.empty(0, dtype=object)


def load_original_csv(path: str, seg_len: int = SEG_LEN) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads PPG_Dataset.csv if present. Assigns each record to an individual group."""
    if not os.path.exists(path):
        return _empty()
    df = pd.read_csv(path)
    sig_cols = [c for c in df.columns if c != 'Label']
    pool = df[sig_cols].values.astype(float)
    if pool.shape[1] != seg_len:
        return _empty()
    n = len(pool)
    src_labels = np.array(['original_csv'] * n, dtype=object)
    grp_labels = np.array([f"csv_{i}" for i in range(n)], dtype=object)
    return pool, src_labels, grp_labels


def group_split(
    source_labels: np.ndarray,
    group_labels: np.ndarray,
    test_size: float = 0.20,
    seed: int = 42
) -> np.ndarray:
    """
    Performs patient/group-level split inside each source dataset.
    Ensures recordings from the same individual never leak across train and test sets.
    Returns:
        is_test: boolean array where True indicates test set membership
    """
    r = np.random.default_rng(seed)
    is_test = np.zeros(len(source_labels), dtype=bool)
    for s in pd.unique(source_labels):
        idx = np.where(source_labels == s)[0]
        ug = np.unique(group_labels[idx])
        if len(ug) < 2:
            continue
        r.shuffle(ug)
        n_test = max(1, int(round(test_size * len(ug))))
        test_groups = ug[:n_test]
        is_test[idx] = np.isin(group_labels[idx], test_groups)
    return is_test
