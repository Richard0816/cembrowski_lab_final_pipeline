"""Event detection — population density peaks + Gaussian boundary refinement."""
from __future__ import annotations

from ..core.config import EventDetectionParams
from .boundaries import (
    boundaries_from_peaks,
    estimate_global_baseline,
    estimate_noise_from_quiet,
    estimate_rolling_baseline,
    fit_gaussian_to_peak,
    walk_boundary,
)
from .density import (
    activation_matrix_from_windows,
    build_density,
    detect_density_peaks,
    detect_event_windows,
)

__all__ = [
    "EventDetectionParams",
    "detect_event_windows",
    "build_density",
    "detect_density_peaks",
    "activation_matrix_from_windows",
    "boundaries_from_peaks",
    "estimate_global_baseline",
    "estimate_rolling_baseline",
    "estimate_noise_from_quiet",
    "walk_boundary",
    "fit_gaussian_to_peak",
]
