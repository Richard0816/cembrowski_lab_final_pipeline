"""Spatial analysis — ROI metrics, activation heatmaps, propagation vectors.

All submodules are real ports. `heatmap` and `vectors` import matplotlib at
module top, so their symbols are loaded lazily on first access to keep the
top-level package importable without matplotlib.
"""
from __future__ import annotations

from .metrics import paint_spatial, roi_metric

_LAZY = {
    "coactivation_maps":                  ("heatmap", "coactivation_maps"),
    "compute_cell_scores":                ("heatmap", "compute_cell_scores"),
    "soft_cell_mask":                     ("heatmap", "soft_cell_mask"),
    "view_roi_features":                  ("heatmap", "view_roi_features"),
    "edge_mask_from_stat":                ("heatmap", "edge_mask_from_stat"),
    "plot_leadlag_split_spatial_from_csv": ("heatmap", "plot_leadlag_split_spatial_from_csv"),
    "robust_norm":                        ("vectors", "robust_norm"),
}

__all__ = ["roi_metric", "paint_spatial", *_LAZY.keys()]


def __getattr__(name: str):
    if name in _LAZY:
        submod_name, attr = _LAZY[name]
        from importlib import import_module
        submod = import_module(f"{__name__}.{submod_name}")
        return getattr(submod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
