"""Baseline / noise estimation, Gaussian peak fitting, boundary walking.

These are the internals of density-based event detection, factored out so the
higher-level `density.detect_event_windows` can stay short and readable.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import percentile_filter
from scipy.special import erfinv

from ..core.config import EventDetectionParams


def estimate_global_baseline(density: np.ndarray, baseline_percentile: float) -> np.ndarray:
    if density.size == 0:
        return np.zeros_like(density)
    b = float(np.percentile(density, baseline_percentile))
    return np.full_like(density, b, dtype=np.float64)


def estimate_rolling_baseline(
    density: np.ndarray,
    fps_density: float,
    baseline_window_s: float,
    baseline_percentile: float,
) -> np.ndarray:
    if density.size == 0:
        return np.zeros_like(density)
    win = max(3, int(round(baseline_window_s * fps_density)) | 1)
    win = min(win, density.size if density.size % 2 == 1 else density.size - 1)
    if win < 3:
        return estimate_global_baseline(density, baseline_percentile)
    return percentile_filter(
        density.astype(np.float64),
        size=win,
        percentile=baseline_percentile,
        mode="reflect",
    )


def estimate_noise_from_quiet(
    density: np.ndarray,
    baseline_trace: np.ndarray,
    quiet_percentile: float = 40.0,
    noise_mad_factor: float = 1.4826,
) -> float:
    if density.size == 0:
        return 0.0
    resid = density - baseline_trace
    cutoff = float(np.percentile(resid, quiet_percentile))
    quiet_mask = resid <= cutoff
    if quiet_mask.sum() < 32:
        quiet_mask = np.ones_like(resid, dtype=bool)
    sample = resid[quiet_mask]
    med = np.median(sample)
    mad = np.median(np.abs(sample - med)) + 1e-12
    return float(noise_mad_factor * mad)


def walk_boundary(
    density: np.ndarray,
    peak_idx: int,
    end_threshold_trace: np.ndarray,
    direction: int,
    max_steps: int,
) -> int:
    n = density.size
    i = peak_idx
    steps = 0
    while 0 <= i + direction < n and steps < max_steps:
        nxt = i + direction
        if density[nxt] <= end_threshold_trace[nxt]:
            return i
        i = nxt
        steps += 1
    return i


def gaussian_z(q: float) -> float:
    if not (0.5 < q < 1.0):
        raise ValueError("gaussian_quantile must be in (0.5, 1.0)")
    return float(np.sqrt(2.0) * erfinv(2.0 * q - 1.0))


def fit_gaussian_to_peak(
    time_s: np.ndarray,
    density: np.ndarray,
    baseline_trace: np.ndarray,
    peak_idx: int,
    left_idx: int,
    right_idx: int,
    pad_samples: int,
    min_sigma_s: float,
) -> Tuple[float, float]:
    n = density.size
    L = max(0, left_idx - pad_samples)
    R = min(n - 1, right_idx + pad_samples)
    if R <= L:
        return float(time_s[peak_idx]), float(min_sigma_s)

    t_win = time_s[L:R + 1].astype(np.float64)
    d_win = density[L:R + 1].astype(np.float64)
    b_win = baseline_trace[L:R + 1].astype(np.float64)
    w = np.clip(d_win - b_win, 0.0, None)

    total = w.sum()
    if total <= 0:
        return float(time_s[peak_idx]), float(min_sigma_s)

    mu = float((t_win * w).sum() / total)
    var = float(((t_win - mu) ** 2 * w).sum() / total)
    sigma = max(float(np.sqrt(max(var, 0.0))), float(min_sigma_s))
    return mu, sigma


def boundaries_from_peaks(
    time_s: np.ndarray,
    smooth: np.ndarray,
    peak_indices: np.ndarray,
    params: EventDetectionParams,
) -> dict:
    """Baseline walk + Gaussian fit + merge. Returns per-event arrays."""
    time_s = np.asarray(time_s, dtype=np.float64)
    smooth = np.asarray(smooth, dtype=np.float64)
    peaks = np.asarray(peak_indices, dtype=np.int64)

    dt = float(np.median(np.diff(time_s))) if time_s.size > 1 else 1.0
    fps_density = 1.0 / max(dt, 1e-12)

    if params.baseline_mode == "global":
        baseline_trace = estimate_global_baseline(smooth, params.baseline_percentile)
    else:
        baseline_trace = estimate_rolling_baseline(
            smooth, fps_density, params.baseline_window_s, params.baseline_percentile,
        )
    noise = estimate_noise_from_quiet(
        smooth, baseline_trace,
        quiet_percentile=params.noise_quiet_percentile,
        noise_mad_factor=params.noise_mad_factor,
    )
    end_threshold_trace = baseline_trace + params.end_threshold_k * noise

    empty_f = np.array([], dtype=np.float64)
    empty_o = np.array([], dtype=object)
    if peaks.size == 0:
        return dict(
            start_s=empty_f, peak_s=empty_f, end_s=empty_f,
            peak_height=empty_f, prominence=empty_f, duration_s=empty_f,
            mu_s=empty_f, sigma_s=empty_f,
            boundary_source_left=empty_o, boundary_source_right=empty_o,
            baseline_trace=baseline_trace, end_threshold_trace=end_threshold_trace,
            baseline_noise=noise,
        )

    peaks = np.sort(peaks)

    max_steps = max(1, int(round(params.max_event_duration_s / dt)))
    start_idx = np.empty(peaks.size, dtype=np.int64)
    end_idx = np.empty(peaks.size, dtype=np.int64)
    for i, p in enumerate(peaks):
        start_idx[i] = walk_boundary(smooth, int(p), end_threshold_trace, -1, max_steps)
        end_idx[i] = walk_boundary(smooth, int(p), end_threshold_trace, +1, max_steps)

    merge_gap_samples = max(0, int(round(params.merge_gap_s / dt)))
    keep = np.ones(peaks.size, dtype=bool)
    for i in range(1, peaks.size):
        j = i - 1
        while j >= 0 and not keep[j]:
            j -= 1
        if j < 0:
            continue
        if start_idx[i] - end_idx[j] <= merge_gap_samples:
            if smooth[peaks[i]] > smooth[peaks[j]]:
                peaks[j] = peaks[i]
            end_idx[j] = max(end_idx[j], end_idx[i])
            start_idx[j] = min(start_idx[j], start_idx[i])
            keep[i] = False
    peaks = peaks[keep]
    start_idx = start_idx[keep]
    end_idx = end_idx[keep]
    n_events = peaks.size

    mu_s = np.full(n_events, np.nan, dtype=np.float64)
    sigma_s = np.full(n_events, np.nan, dtype=np.float64)
    src_left = np.empty(n_events, dtype=object)
    src_right = np.empty(n_events, dtype=object)

    base_start_s = time_s[start_idx].astype(np.float64)
    base_end_s = time_s[end_idx].astype(np.float64)

    if params.use_gaussian_boundary and n_events > 0:
        z = gaussian_z(params.gaussian_quantile)
        pad_samples = max(0, int(round(params.gaussian_fit_pad_s / dt)))
        start_s_out = np.empty(n_events, dtype=np.float64)
        end_s_out = np.empty(n_events, dtype=np.float64)
        for i in range(n_events):
            mu, sig = fit_gaussian_to_peak(
                time_s=time_s, density=smooth, baseline_trace=baseline_trace,
                peak_idx=int(peaks[i]), left_idx=int(start_idx[i]), right_idx=int(end_idx[i]),
                pad_samples=pad_samples, min_sigma_s=params.gaussian_min_sigma_s,
            )
            mu_s[i] = mu
            sigma_s[i] = sig
            g_start = mu - z * sig
            g_end = mu + z * sig
            chosen_start = max(g_start, base_start_s[i])
            chosen_end = min(g_end, base_end_s[i])
            peak_t = float(time_s[peaks[i]])
            chosen_start = min(chosen_start, peak_t)
            chosen_end = max(chosen_end, peak_t)
            start_s_out[i] = chosen_start
            end_s_out[i] = chosen_end
            src_left[i] = "gaussian" if g_start >= base_start_s[i] else "baseline"
            src_right[i] = "gaussian" if g_end <= base_end_s[i] else "baseline"
    else:
        start_s_out = base_start_s
        end_s_out = base_end_s
        src_left[:] = "baseline"
        src_right[:] = "baseline"

    peak_height = smooth[peaks]
    prominences_final = peak_height - end_threshold_trace[peaks]

    return dict(
        start_s=start_s_out,
        peak_s=time_s[peaks].astype(np.float64),
        end_s=end_s_out,
        peak_height=peak_height,
        prominence=prominences_final,
        duration_s=end_s_out - start_s_out,
        mu_s=mu_s, sigma_s=sigma_s,
        boundary_source_left=src_left,
        boundary_source_right=src_right,
        baseline_trace=baseline_trace,
        end_threshold_trace=end_threshold_trace,
        baseline_noise=noise,
    )
