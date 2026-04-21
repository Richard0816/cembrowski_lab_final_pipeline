"""
calcium_pipeline.core
---------------------
Pure-numpy primitives shared across stages:

* :mod:`.signal`  — per-trace ΔF/F, low-pass, derivative
* :mod:`.events`  — per-trace hysteresis onsets, population-density events
* :mod:`.spatial` — painting scalars onto the imaging plane, per-ROI metrics
* :mod:`.logging` — tee / per-folder logging helpers

Kept free of torch, matplotlib, GPU and GUI imports so the stage modules can
use them in headless batch scripts.
"""
from __future__ import annotations

__all__ = [
    # signal
    "robust_df_over_f_1d",
    "lowpass_causal_1d",
    "sg_first_derivative_1d",
    # events
    "mad_z",
    "hysteresis_onsets",
    "EventDetectionParams",
    "detect_event_windows",
    # spatial
    "paint_spatial",
    "roi_metric",
    "build_time_mask",
    # logging
    "Tee",
    "run_on_folders",
    "run_with_logging",
]


def __getattr__(name):
    if name in ("robust_df_over_f_1d", "lowpass_causal_1d", "sg_first_derivative_1d"):
        from . import signal as _s
        return getattr(_s, name)
    if name in ("mad_z", "hysteresis_onsets", "EventDetectionParams", "detect_event_windows"):
        from . import events as _e
        return getattr(_e, name)
    if name in ("paint_spatial", "roi_metric", "build_time_mask"):
        from . import spatial as _sp
        return getattr(_sp, name)
    if name in ("Tee", "run_on_folders", "run_with_logging"):
        from . import logging as _lg
        return getattr(_lg, name)
    raise AttributeError(name)
