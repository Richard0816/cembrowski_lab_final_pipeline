"""Typed result objects returned by analysis steps.

These are the contracts between `calcium_core` compute and any caller (app,
CLI, notebooks). They are intentionally simple NumPy-backed dataclasses so
the app can render them without importing matplotlib.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class EventTable:
    """Population-event results from density-based detection.

    Attributes
    ----------
    windows : (E, 2) float
        [start_s, end_s] for each event.
    activation : (N, E) bool
        activation[i, e] = True if ROI i had an onset inside event e.
    first_time : (N, E) float
        Earliest onset time (s) of ROI i in event e, NaN if inactive.
    peak_s, peak_height, prominence, duration_s : (E,) float
    mu_s, sigma_s : (E,) float   — Gaussian fit per event (NaN if unused)
    """
    windows: np.ndarray
    activation: np.ndarray
    first_time: np.ndarray
    peak_s: Optional[np.ndarray] = None
    peak_height: Optional[np.ndarray] = None
    prominence: Optional[np.ndarray] = None
    duration_s: Optional[np.ndarray] = None
    mu_s: Optional[np.ndarray] = None
    sigma_s: Optional[np.ndarray] = None

    @property
    def n_events(self) -> int:
        return int(self.windows.shape[0])

    @property
    def n_rois(self) -> int:
        return int(self.activation.shape[0])


@dataclass
class ClusterResult:
    """Hierarchical-clustering output."""
    labels: np.ndarray          # (N,) int cluster id per ROI
    linkage: np.ndarray         # scipy.cluster.hierarchy linkage matrix
    leaf_order: np.ndarray      # (N,) dendrogram leaf order
    color_threshold: float


@dataclass
class XCorrResult:
    """Pairwise cross-correlation result for a ROI pair / cluster pair."""
    lags_s: np.ndarray          # (L,) float
    xcorr: np.ndarray           # (L,) float, normalised
    peak_lag_s: float
    peak_value: float


@dataclass
class DensityDiagnostics:
    """Internal state from density-event detection, for plots/debug."""
    time_centers_s: np.ndarray
    binned_density: np.ndarray
    smoothed_density: np.ndarray
    baseline_trace: np.ndarray
    end_threshold_trace: np.ndarray
    baseline_noise: float
    boundary_source_left: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
    boundary_source_right: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
