"""Visualization for density-based event detection."""
from __future__ import annotations

from typing import Optional

import numpy as np


def plot_event_detection(
    diagnostics: dict,
    ylabel: str = "Onset density (per bin per ROI)",
    title: str = "Detected events on onset density",
    ax=None,
):
    """Binned counts (bars) + smoothed density + baseline + peaks + thresholds."""
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
    ax.plot(centers, diagnostics["baseline_trace"], color="gray", linestyle=":",
            linewidth=1.0, label="Baseline")
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


def shade_event_windows(ax, event_windows: np.ndarray, color="C1", alpha: float = 0.20):
    """Overlay shaded event windows onto an existing axis."""
    for s, e in event_windows:
        ax.axvspan(s, e, color=color, alpha=alpha)
