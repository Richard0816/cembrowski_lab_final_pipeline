"""
Default hyperparameters per stage.

One dict per stage, assembled into the nested ``DEFAULTS`` dict. The GUI
renders each sub-dict as its own tab, so keep the names short and stable —
the widget layout reflects these keys.
"""
from __future__ import annotations

# --- signal_extraction -----------------------------------------------------
# Neuropil subtraction r = 0.7; rolling-percentile baseline; Butterworth
# lowpass; Savitzky-Golay first derivative.
SIGNAL_EXTRACTION = {
    "prefix": "r0p7_",
    "neuropil_factor": 0.7,
    "baseline_window_sec": 45.0,
    "baseline_percentile": 10.0,
    "lowpass_cutoff_hz": 5.0,
    "lowpass_order": 2,
    "sg_window_ms": 333.0,
    "sg_poly": 3,
    # Filtered-by-cellfilter output (applied after predicted_cell_mask.npy)
    "filtered_prefix": "r0p7_filtered_",
}

# --- cellfilter inference --------------------------------------------------
CELLFILTER = {
    "patch_size": 32,
    "trace_crop_len": 2000,
    "threshold": 0.5,
    "predicted_prob_name": "predicted_cell_prob.npy",
    "predicted_mask_name": "predicted_cell_mask.npy",
}

# --- events: per-ROI hysteresis onsets + density events --------------------
EVENTS = {
    # per-ROI hysteresis
    "z_enter": 3.5,
    "z_exit": 1.5,
    "min_sep_s": 0.3,
    # density construction
    "bin_sec": 0.05,
    "smooth_sigma_bins": 2.0,
    "normalize_by_num_rois": True,
    # peak gating
    "min_prominence": 0.007,
    "min_width_bins": 2.0,
    "min_distance_bins": 3.0,
    # baseline + noise
    "baseline_mode": "rolling",
    "baseline_percentile": 5.0,
    "baseline_window_s": 30.0,
    "noise_quiet_percentile": 40.0,
    "noise_mad_factor": 1.4826,
    # boundary walk
    "end_threshold_k": 2.0,
    "max_event_duration_s": 10.0,
    # merging
    "merge_gap_s": 0.0,
    # Gaussian refinement
    "use_gaussian_boundary": True,
    "gaussian_quantile": 0.99,
    "gaussian_fit_pad_s": 0.5,
    "gaussian_min_sigma_s": 0.05,
}

# --- clustering ------------------------------------------------------------
CLUSTERING = {
    "metric": "correlation",
    "method": "ward",
    "z_score_rows": True,
    "target_n_groups_low": 4,
    "target_n_groups_high": 5,
    "color_threshold": None,   # None → auto-pick to hit target_n_groups
}

# --- spatial heatmap -------------------------------------------------------
SPATIAL_HEATMAP = {
    "target_bins_low": 20,
    "target_bins_high": 50,
    "coactivation_percentile_lo": 60,
    "coactivation_percentile_hi": 95,
    "edge_mask_margin_px": 5,
    "cell_score_thresh": 0.15,
}

# --- crosscorrelation ------------------------------------------------------
CROSSCORRELATION = {
    "use_gpu": True,
    "max_lag_s": 2.0,
    "n_surrogates": 1000,
    "fdr_alpha": 0.05,
    "min_coactive_rois": 4,
}

# --- pipeline orchestration -----------------------------------------------
ORCHESTRATION = {
    # Default stage sequence run by ``orchestration.runner.run_pipeline``.
    "default_pipeline": (
        "signal_extraction",
        "cellfilter",
        "events",
        "clustering",
        "spatial_heatmap",
        "crosscorrelation",
    ),
    # Fraction of failures that still counts as a successful run.
    "skip_on_error": True,
}


DEFAULTS: dict[str, dict] = {
    "signal_extraction": SIGNAL_EXTRACTION,
    "cellfilter": CELLFILTER,
    "events": EVENTS,
    "clustering": CLUSTERING,
    "spatial_heatmap": SPATIAL_HEATMAP,
    "crosscorrelation": CROSSCORRELATION,
    "orchestration": ORCHESTRATION,
}

__all__ = [
    "DEFAULTS",
    "SIGNAL_EXTRACTION",
    "CELLFILTER",
    "EVENTS",
    "CLUSTERING",
    "SPATIAL_HEATMAP",
    "CROSSCORRELATION",
    "ORCHESTRATION",
]
