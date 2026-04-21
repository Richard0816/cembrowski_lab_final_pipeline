"""
calcium_pipeline.viz
--------------------
Plotting helpers. Separated from ``stages`` so headless batch jobs (and the
compute path in the GUI) don't import matplotlib. Every function in this
package returns a matplotlib ``Figure`` — it is the caller's responsibility
to save / display / embed it.

Modules
-------
* :mod:`.raster`        — raster plots of onsets
* :mod:`.heatmap`       — (T, N) ΔF/F heatmaps, with optional event overlays
* :mod:`.dendrogram`    — dendrogram + cluster-coloured heatmap
* :mod:`.order_map`     — spatial order map (ROI rank painted on the plane)
* :mod:`.event_overlay` — shade event windows on any time-axis plot
* :mod:`.cmaps`         — shared colormaps (CYAN_TO_RED, grey-for-bad, etc.)
"""
from __future__ import annotations

__all__ = [
    "plot_raster",
    "plot_dff_heatmap",
    "plot_dendrogram",
    "plot_order_map",
    "shade_events",
    "CYAN_TO_RED",
]


def __getattr__(name):
    if name == "plot_raster":
        from .raster import plot_raster
        return plot_raster
    if name == "plot_dff_heatmap":
        from .heatmap import plot_dff_heatmap
        return plot_dff_heatmap
    if name == "plot_dendrogram":
        from .dendrogram import plot_dendrogram
        return plot_dendrogram
    if name == "plot_order_map":
        from .order_map import plot_order_map
        return plot_order_map
    if name == "shade_events":
        from .event_overlay import shade_events
        return shade_events
    if name == "CYAN_TO_RED":
        from .cmaps import CYAN_TO_RED
        return CYAN_TO_RED
    raise AttributeError(name)
