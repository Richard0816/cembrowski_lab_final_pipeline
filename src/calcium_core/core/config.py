"""Frozen pipeline configuration dataclasses."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EventDetectionParams:
    """Density-based event detection + boundary refinement parameters.

    Defaults tuned for 2P calcium imaging at 15 Hz.
    """
    # Density construction
    bin_sec: float = 0.05
    smooth_sigma_bins: float = 2.0
    normalize_by_num_rois: bool = True

    # Peak detection on smoothed density
    min_prominence: float = 0.007
    min_width_bins: float = 2.0
    min_distance_bins: float = 3.0

    # Baseline / noise
    baseline_mode: str = "rolling"         # "rolling" or "global"
    baseline_percentile: float = 5.0
    baseline_window_s: float = 30.0
    noise_quiet_percentile: float = 40.0
    noise_mad_factor: float = 1.4826

    # Boundary walking
    end_threshold_k: float = 2.0
    max_event_duration_s: float = 10.0

    # Overlap merging
    merge_gap_s: float = 0.0

    # Gaussian-fit refinement
    use_gaussian_boundary: bool = True
    gaussian_quantile: float = 0.99
    gaussian_fit_pad_s: float = 0.5
    gaussian_min_sigma_s: float = 0.05


@dataclass
class Suite2pRunConfig:
    """Parameters for a Suite2p run on one recording."""
    path_to_ops: str
    aav_info_csv: str
    spatial_hp_detect: float = 24.0
    threshold_scaling: float = 0.88
    tau_vals: dict = None

    def __post_init__(self):
        if self.tau_vals is None:
            self.tau_vals = {"6f": 0.7, "6m": 1.0, "6s": 1.3, "8m": 0.137}
