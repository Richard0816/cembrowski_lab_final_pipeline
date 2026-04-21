"""
Event-detection primitives.

Two levels of detection live here:

1. **Per-ROI onsets** — hysteresis (Schmitt trigger) on a robust z-score of the
   SG-derivative trace. Used by :mod:`stages.events` to produce
   ``onsets_by_roi``.

2. **Population density events** — given ``onsets_by_roi`` for every ROI,
   build a smoothed onset-density signal, detect peaks, walk boundaries to a
   rolling-percentile baseline, and optionally refine with a Gaussian fit.
   Produces per-event ``[start_s, end_s]`` windows plus an activation matrix.

Both halves are numpy-only; matplotlib is imported lazily by the plotting
helpers so headless code never pulls it in.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter1d, percentile_filter
from scipy.signal import find_peaks
from scipy.special import erfinv


# ==========================================================================
# 1. Per-ROI primitives
# ==========================================================================

def mad_z(x: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Robust z-score of ``x`` using median and MAD (scaled by 1.4826 to
    approximate σ under normality).

    Returns ``(z, median, mad)`` so callers can invert the transform.
    (Technique from Stern et al. 2024.)
    """
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    return (x - med) / (1.4826 * mad), float(med), float(mad)


def hysteresis_onsets(
    z: np.ndarray,
    z_hi: float,
    z_lo: float,
    fps: float,
    min_sep_s: float = 0.0,
) -> np.ndarray:
    """
    Schmitt-trigger onset detection on a robust-z trace.

    * Enter state when ``z >= z_hi`` → record the frame as an onset.
    * Leave state when ``z <= z_lo``.
    * Optionally merge onsets that land within ``min_sep_s`` seconds of the
      previous kept onset.

    Returns a 1-D ``int`` array of onset frame indices.
    """
    above_hi = z >= z_hi
    onsets: list[int] = []
    active = False
    for i in range(z.size):
        if not active and above_hi[i]:
            active = True
            onsets.append(i)
        elif active and z[i] <= z_lo:
            active = False

    if not onsets:
        return np.array([], dtype=int)

    arr = np.array(onsets, dtype=int)
    if min_sep_s > 0:
        min_sep = int(min_sep_s * fps)
        merged = [arr[0]]
        for k in arr[1:]:
            if k - merged[-1] >= min_sep:
                merged.append(k)
        arr = np.asarray(merged, dtype=int)
    return arr


# ==========================================================================
# 2. Population-density event detection
# ==========================================================================

@dataclass
class EventDetectionParams:
    """
    Tunables for :func:`detect_event_windows`. Defaults are chosen for 2P
    calcium imaging at ~15 Hz.
    """
    # --- density construction ---
    bin_sec: float = 0.05
    smooth_sigma_bins: float = 2.0
    normalize_by_num_rois: bool = True

    # --- peak detection (scipy.signal.find_peaks) ---
    min_prominence: float = 0.007
    min_width_bins: float = 2.0
    min_distance_bins: float = 3.0

    # --- baseline + noise (for the boundary walk) ---
    baseline_mode: str = "rolling"       # "rolling" or "global"
    baseline_percentile: float = 5.0
    baseline_window_s: float = 30.0
    noise_quiet_percentile: float = 40.0
    noise_mad_factor: float = 1.4826

    # --- boundary walk ---
    end_threshold_k: float = 2.0         # threshold = baseline + k * noise
    max_event_duration_s: float = 10.0

    # --- overlap merging ---
    merge_gap_s: float = 0.0

    # --- Gaussian-fit refinement ---
    use_gaussian_boundary: bool = True
    gaussian_quantile: float = 0.99
    gaussian_fit_pad_s: float = 0.5
    gaussian_min_sigma_s: float = 0.05


def detect_event_windows(
    onsets_by_roi: Sequence[np.ndarray],
    T: int,
    fps: float,
    params: Optional[EventDetectionParams] = None,
    return_diagnostics: bool = False,
):
    """
    Detect population events from per-ROI onset times.

    Parameters
    ----------
    onsets_by_roi : sequence of 1-D arrays
        Per-ROI onset times **in seconds**. Length = number of ROIs.
    T : int
        Number of frames in the recording (used to derive the density axis).
    fps : float
        Sampling rate in Hz.
    params : EventDetectionParams, optional
        Detection / boundary parameters. Uses defaults if None.
    return_diagnostics : bool
        If True, also return a diagnostics dict suitable for
        :func:`plot_event_detection`.

    Returns
    -------
    event_windows : (n_events, 2) float
        ``[start_s, end_s]`` per event.
    A : (N_rois, n_events) bool
        ``A[i, e]`` == True iff ROI *i* had ≥ 1 onset inside event *e*.
    first_time : (N_rois, n_events) float
        Earliest onset time (s) of ROI *i* inside event *e*; NaN if inactive.
    diagnostics : dict (only if ``return_diagnostics``)
    """
    if params is None:
        params = EventDetectionParams()

    duration_s = float(T) / float(fps)
    centers, counts, smooth = _build_density(
        onsets_by_roi=onsets_by_roi,
        duration_s=duration_s,
        bin_sec=params.bin_sec,
        smooth_sigma_bins=params.smooth_sigma_bins,
        n_rois=len(onsets_by_roi),
        normalize_by_num_rois=params.normalize_by_num_rois,
    )

    peak_indices = _detect_density_peaks(
        smooth=smooth,
        min_prominence=params.min_prominence,
        min_width_bins=params.min_width_bins,
        min_distance_bins=params.min_distance_bins,
    )

    boundaries = _boundaries_from_peaks(
        time_s=centers,
        smooth=smooth,
        peak_indices=peak_indices,
        params=params,
    )

    event_windows = np.column_stack([boundaries["start_s"], boundaries["end_s"]])
    A, first_time = _activation_matrix_from_windows(onsets_by_roi, event_windows)

    if not return_diagnostics:
        return event_windows, A, first_time

    diagnostics = {
        "time_centers_s": centers,
        "binned_density": counts,
        "smoothed_density": smooth,
        "baseline_trace": boundaries["baseline_trace"],
        "end_threshold_trace": boundaries["end_threshold_trace"],
        "baseline_noise": boundaries["baseline_noise"],
        "peak_s": boundaries["peak_s"],
        "peak_height": boundaries["peak_height"],
        "mu_s": boundaries["mu_s"],
        "sigma_s": boundaries["sigma_s"],
        "boundary_source_left": boundaries["boundary_source_left"],
        "boundary_source_right": boundaries["boundary_source_right"],
        "prominence": boundaries["prominence"],
        "duration_s": boundaries["duration_s"],
    }
    return event_windows, A, first_time, diagnostics


# --------------------------------------------------------------------------
# Density + peaks
# --------------------------------------------------------------------------

def _build_density(
    onsets_by_roi: Sequence[np.ndarray],
    duration_s: float,
    bin_sec: float,
    smooth_sigma_bins: float,
    n_rois: int,
    normalize_by_num_rois: bool,
):
    """Flatten onsets, histogram into fixed bins, Gaussian-smooth."""
    nonempty = [np.asarray(x, dtype=np.float64) for x in onsets_by_roi if len(x) > 0]
    flat = np.concatenate(nonempty) if nonempty else np.array([], dtype=np.float64)

    edges = np.arange(0.0, duration_s + bin_sec, bin_sec, dtype=np.float64)
    if edges[-1] < duration_s:
        edges = np.append(edges, duration_s)

    counts, edges = np.histogram(flat, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])

    counts = counts.astype(np.float64)
    if normalize_by_num_rois and n_rois > 0:
        counts = counts / float(n_rois)

    smooth = gaussian_filter1d(counts, sigma=smooth_sigma_bins, mode="nearest")
    return centers, counts, smooth


def _detect_density_peaks(
    smooth: np.ndarray,
    min_prominence: float,
    min_width_bins: float,
    min_distance_bins: float,
) -> np.ndarray:
    """Run ``scipy.signal.find_peaks`` with the standard gating parameters."""
    peaks, _props = find_peaks(
        smooth,
        prominence=min_prominence,
        width=min_width_bins,
        distance=max(1, int(round(min_distance_bins))),
    )
    return peaks


# --------------------------------------------------------------------------
# Baseline + noise
# --------------------------------------------------------------------------

def _estimate_global_baseline(density: np.ndarray, baseline_percentile: float) -> np.ndarray:
    if density.size == 0:
        return np.zeros_like(density)
    b = float(np.percentile(density, baseline_percentile))
    return np.full_like(density, b, dtype=np.float64)


def _estimate_rolling_baseline(
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
        return _estimate_global_baseline(density, baseline_percentile)
    return percentile_filter(
        density.astype(np.float64),
        size=win,
        percentile=baseline_percentile,
        mode="reflect",
    )


def _estimate_noise_from_quiet(
    density: np.ndarray,
    baseline_trace: np.ndarray,
    quiet_percentile: float = 40.0,
    noise_mad_factor: float = 1.4826,
) -> float:
    """MAD of the "quiet" (below-``quiet_percentile``) residuals."""
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


# --------------------------------------------------------------------------
# Boundary walk + Gaussian refinement
# --------------------------------------------------------------------------

def _walk_boundary(
    density: np.ndarray,
    peak_idx: int,
    end_threshold_trace: np.ndarray,
    direction: int,
    max_steps: int,
) -> int:
    """Step left/right from the peak until density drops below threshold."""
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


def _gaussian_z(q: float) -> float:
    if not (0.5 < q < 1.0):
        raise ValueError("gaussian_quantile must be in (0.5, 1.0)")
    return float(np.sqrt(2.0) * erfinv(2.0 * q - 1.0))


def _fit_gaussian_to_peak(
    time_s: np.ndarray,
    density: np.ndarray,
    baseline_trace: np.ndarray,
    peak_idx: int,
    left_idx: int,
    right_idx: int,
    pad_samples: int,
    min_sigma_s: float,
) -> tuple[float, float]:
    """Weighted-moment Gaussian fit over ``[left-pad .. right+pad]``."""
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


def _boundaries_from_peaks(
    time_s: np.ndarray,
    smooth: np.ndarray,
    peak_indices: np.ndarray,
    params: EventDetectionParams,
) -> dict:
    """Baseline walk -> merge -> Gaussian refine. Returns a dict of per-event arrays."""
    time_s = np.asarray(time_s, dtype=np.float64)
    smooth = np.asarray(smooth, dtype=np.float64)
    peaks = np.asarray(peak_indices, dtype=np.int64)

    dt = float(np.median(np.diff(time_s))) if time_s.size > 1 else 1.0
    fps_density = 1.0 / max(dt, 1e-12)

    if params.baseline_mode == "global":
        baseline_trace = _estimate_global_baseline(smooth, params.baseline_percentile)
    else:
        baseline_trace = _estimate_rolling_baseline(
            smooth, fps_density, params.baseline_window_s, params.baseline_percentile,
        )
    noise = _estimate_noise_from_quiet(
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
            baseline_trace=baseline_trace,
            end_threshold_trace=end_threshold_trace,
            baseline_noise=noise,
        )

    peaks = np.sort(peaks)

    # Walk
    max_steps = max(1, int(round(params.max_event_duration_s / dt)))
    start_idx = np.empty(peaks.size, dtype=np.int64)
    end_idx = np.empty(peaks.size, dtype=np.int64)
    for i, p in enumerate(peaks):
        start_idx[i] = _walk_boundary(smooth, int(p), end_threshold_trace, -1, max_steps)
        end_idx[i] = _walk_boundary(smooth, int(p), end_threshold_trace, +1, max_steps)

    # Merge overlapping / touching events
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

    # Gaussian refine
    mu_s = np.full(n_events, np.nan, dtype=np.float64)
    sigma_s = np.full(n_events, np.nan, dtype=np.float64)
    src_left = np.empty(n_events, dtype=object)
    src_right = np.empty(n_events, dtype=object)

    base_start_s = time_s[start_idx].astype(np.float64)
    base_end_s = time_s[end_idx].astype(np.float64)

    if params.use_gaussian_boundary and n_events > 0:
        z = _gaussian_z(params.gaussian_quantile)
        pad_samples = max(0, int(round(params.gaussian_fit_pad_s / dt)))
        start_s_out = np.empty(n_events, dtype=np.float64)
        end_s_out = np.empty(n_events, dtype=np.float64)
        for i in range(n_events):
            mu, sig = _fit_gaussian_to_peak(
                time_s=time_s, density=smooth, baseline_trace=baseline_trace,
                peak_idx=int(peaks[i]), left_idx=int(start_idx[i]),
                right_idx=int(end_idx[i]),
                pad_samples=pad_samples, min_sigma_s=params.gaussian_min_sigma_s,
            )
            mu_s[i] = mu
            sigma_s[i] = sig
            g_start = mu - z * sig
            g_end = mu + z * sig
            # Whichever boundary comes first wins (tighter is better).
            chosen_start = max(g_start, base_start_s[i])
            chosen_end = min(g_end, base_end_s[i])
            # Never cross the peak.
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


# --------------------------------------------------------------------------
# Activation matrix
# --------------------------------------------------------------------------

def _activation_matrix_from_windows(
    onsets_by_roi: Sequence[np.ndarray],
    event_windows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Given per-ROI onsets and ``(n_events, 2)`` ``[start_s, end_s]`` windows,
    return:

    * ``A : (N, E) bool``      — any onset inside window?
    * ``first_time : (N, E)``  — earliest onset (s) inside window, NaN if none.
    """
    N = len(onsets_by_roi)
    E = event_windows.shape[0]
    A = np.zeros((N, E), dtype=bool)
    first_time = np.full((N, E), np.nan, dtype=np.float64)

    if E == 0:
        return A, first_time

    starts = event_windows[:, 0].astype(np.float64)
    ends = event_windows[:, 1].astype(np.float64)

    for i, ts in enumerate(onsets_by_roi):
        ts = np.asarray(ts, dtype=np.float64)
        if ts.size == 0:
            continue
        # For each window, is any onset inside?  (E x n_onsets) matrix.
        inside = (ts[None, :] >= starts[:, None]) & (ts[None, :] <= ends[:, None])
        any_inside = inside.any(axis=1)
        A[i, any_inside] = True
        for e in np.where(any_inside)[0]:
            first_time[i, e] = float(ts[inside[e]].min())

    return A, first_time


# ==========================================================================
# 3. Plotting helpers (matplotlib imported lazily)
# ==========================================================================

def plot_event_detection(
    diagnostics: dict,
    ylabel: str = "Onset density (per bin per ROI)",
    title: str = "Detected events on onset density",
    ax=None,
):
    """
    Render the diagnostics dict returned by
    ``detect_event_windows(..., return_diagnostics=True)``:

    binned counts (bars) + smoothed density (line) + baseline + end-threshold
    + peak markers. Returns the ``Figure``.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 4.5))
    else:
        fig = ax.figure

    centers = diagnostics["time_centers_s"]
    smooth = diagnostics["smoothed_density"]
    counts = diagnostics["binned_density"]

    bin_width = float(np.median(np.diff(centers))) if centers.size > 1 else 1.0

    ax.bar(centers, counts, width=bin_width, alpha=0.35, align="center",
           color="C0", edgecolor="none", label="Binned counts")
    ax.plot(centers, smooth, linewidth=1.5, color="C0", label="Smoothed density")
    ax.plot(centers, diagnostics["baseline_trace"], color="gray",
            linestyle=":", linewidth=1.0, label="Baseline")
    ax.plot(centers, diagnostics["end_threshold_trace"], color="black",
            linestyle="--", linewidth=1.0, label="End threshold")

    peak_s = diagnostics["peak_s"]
    peak_h = diagnostics["peak_height"]
    ax.plot(peak_s, peak_h, "v", color="C3", markersize=6,
            label=f"Peaks (n={peak_s.size})")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    return fig


def shade_event_windows(ax, event_windows: np.ndarray, color: str = "C1", alpha: float = 0.20):
    """Overlay shaded event windows onto an existing matplotlib axis."""
    for s, e in event_windows:
        ax.axvspan(s, e, color=color, alpha=alpha)


__all__ = [
    "mad_z",
    "hysteresis_onsets",
    "EventDetectionParams",
    "detect_event_windows",
    "plot_event_detection",
    "shade_event_windows",
]
