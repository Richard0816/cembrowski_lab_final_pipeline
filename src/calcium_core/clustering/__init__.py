"""Hierarchical clustering + pairwise cross-correlation.

Both submodules are real ports but eagerly import heavy deps (seaborn for
hierarchical, cupy for crosscorr with cpu fallback). To keep the top-level
package importable without those stacks installed, the symbols below are
loaded lazily on first attribute access.
"""
from __future__ import annotations

_LAZY = {
    "run_clustering":              ("hierarchical", "run_clustering"),
    "run_clustering_pipeline":     ("hierarchical", "run_clustering_pipeline"),
    "export_rois_by_leaf_color":   ("hierarchical", "export_rois_by_leaf_color"),
    "count_leaf_color_groups":     ("hierarchical", "count_leaf_color_groups"),
    "compute_cross_correlation":   ("crosscorr", "compute_cross_correlation"),
    "compute_zero_lag_corr_cpu":   ("crosscorr", "compute_zero_lag_corr_cpu"),
}

__all__ = list(_LAZY.keys())


def __getattr__(name: str):
    if name in _LAZY:
        submod_name, attr = _LAZY[name]
        from importlib import import_module
        submod = import_module(f"{__name__}.{submod_name}")
        return getattr(submod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
