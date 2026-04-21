"""Baseline normalisation — robust dF/F and MAD z-score."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import percentile_filter


def robust_df_over_f_1d(F, win_sec=45, perc=10, fps=30.0):
    """Rolling-percentile baseline on a 1D trace. Low-RAM."""
    F = np.asarray(F, dtype=np.float32)
    n = F.size

    finite = np.isfinite(F)
    if not finite.all():
        F = np.interp(np.arange(n), np.flatnonzero(finite), F[finite]).astype(np.float32)

    win = max(3, int(win_sec * fps) | 1)
    win = min(win, n if n % 2 == 1 else n - 1)
    if win < 3:
        F0 = np.full_like(F, np.nanpercentile(F, perc))
    else:
        F0 = percentile_filter(F, size=win, percentile=perc, mode="nearest").astype(np.float32)

    eps = np.nanpercentile(F0, 1) if np.isfinite(F0).any() else 1.0
    eps = max(eps, 1e-9)
    return (F - F0) / eps


def mad_z(x):
    """Robust z-score via MAD × 1.4826 (approx σ for normal data).

    Returns (z, median, mad) so callers can invert the transform.
    Source: Stern et al. 2024.
    """
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    return (x - med) / (1.4826 * mad), med, mad
