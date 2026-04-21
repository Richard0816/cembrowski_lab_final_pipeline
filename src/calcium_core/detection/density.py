"""Density-based population event detection.

`detect_event_windows` is the top-level entry point: flat onset times ->
histogram -> smoothed density -> peak detection -> boundary refinement ->
(windows, activation_matrix, first_time).
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from ..core.config import EventDetectionParams
from .boundaries import boundaries_from_peaks


def detect_event_windows(
    onsets_by_roi: Sequence[np.ndarray],
    T: int,
    fps: float,
    params: Optional[EventDetectionParams] = None,
    return_diagnostics: bool = False,
):
    """Detect population events from per-ROI onset times.

    Parameters
    ----------
    onsets_by_roi : sequence of 1D arrays
        Onset times in SECONDS, one array per ROI.
    T : int
        Number of frames in the recording.
    fps : float
        Sampling rate in Hz.
    params : EventDetectionParams, optional
    return_diagnostics : bool

    Returns
    -------
    event_windows : (E, 2) float — [start_s, end_s] per event
    A : (N, E) bool — activation matrix
    first_time : (N, E) float — earliest onset per ROI/event, NaN if inactive
    diagnostics : dict — only if `return_diagnostics`
    """
    if params is None:
        params = EventDetectionParams()

    duration_s = float(T) / float(fps)
    centers, counts, smooth = build_density(
        onsets_by_roi=onsets_by_roi,
        duration_s=duration_s,
        bin_sec=params.bin_sec,
        smooth_sigma_bins=params.smooth_sigma_bins,
        n_rois=len(onsets_by_roi),
        normalize_by_num_rois=params.normalize_by_num_rois,
    )

    peak_indices = detect_density_peaks(
        smooth=smooth,
        min_prominence=params.min_prominence,
        min_width_bins=params.min_width_bins,
        min_distance_bins=params.min_distance_bins,
    )

    boundaries = boundaries_from_peaks(
        time_s=centers,
        smooth=smooth,
        peak_indices=peak_indices,
        params=params,
    )

    event_windows = np.column_stack([boundaries["start_s"], boundaries["end_s"]])
    A, first_time = activation_matrix_from_windows(onsets_by_roi, event_windows)

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


def build_density(
    onsets_by_roi: Sequence[np.ndarray],
    duration_s: float,
    bin_sec: float,
    smooth_sigma_bins: float,
    n_rois: int,
    normalize_by_num_rois: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten onsets, histogram them, Gaussian-smooth. Returns (centers, counts, smooth)."""
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


def detect_density_peaks(
    smooth: np.ndarray,
    min_prominence: float,
    min_width_bins: float,
    min_distance_bins: float,
) -> np.ndarray:
    """scipy find_peaks with prominence/width/distance gates."""
    peaks, _ = find_peaks(
        smooth,
        prominence=min_prominence,
        width=min_width_bins,
        distance=max(1, int(round(min_distance_bins))),
    )
    return peaks


def activation_matrix_from_windows(
    onsets_by_roi: Sequence[np.ndarray],
    event_windows: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """For each (ROI, event) cell: did the ROI fire inside [start_s, end_s]?

    Returns (A: (N, E) bool, first_time: (N, E) float, NaN if inactive).
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
        inside = (ts[None, :] >= starts[:, None]) & (ts[None, :] <= ends[:, None])
        any_inside = inside.any(axis=1)
        A[i, any_inside] = True
        for e in np.where(any_inside)[0]:
            first_time[i, e] = float(ts[inside[e]].min())

    return A, first_time
