"""
Raster plots of per-ROI onsets.
"""
from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_raster(
    onsets_by_roi: Sequence[np.ndarray],
    *,
    tmax: float | None = None,
    color: str = "black",
    height: float = 0.6,
    linewidth: float = 0.5,
    ax=None,
):
    """
    Draw a classic ``(ROI, time)`` raster.

    Parameters
    ----------
    onsets_by_roi : sequence of 1-D float arrays
        Per-ROI onset times in seconds (same layout as
        ``stages.events`` output).
    tmax : float, optional
        Crop the x-axis to ``[0, tmax]`` seconds.
    color, height, linewidth : matplotlib props
        Tick styling.
    ax : Axes, optional
        Draw into an existing axis instead of making a new figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, max(2.5, 0.02 * len(onsets_by_roi))))
    else:
        fig = ax.figure

    for i, ts in enumerate(onsets_by_roi):
        if len(ts) == 0:
            continue
        ax.vlines(ts, i - height / 2, i + height / 2,
                  color=color, linewidth=linewidth)

    ax.set_ylim(-0.5, len(onsets_by_roi) - 0.5)
    if tmax is not None:
        ax.set_xlim(0, tmax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ROI #")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


__all__ = ["plot_raster"]
