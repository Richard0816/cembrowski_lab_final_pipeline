"""
Dendrogram + cluster-coloured heatmap.

Used by the GUI's "Clustering" tab to show the hierarchical cut at the
threshold the stage picked (or the user overrode).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram


def plot_dendrogram(
    Z: np.ndarray,
    *,
    color_threshold: float,
    title: str = "ROI hierarchical clustering",
    ax=None,
):
    """
    Draw the dendrogram with a horizontal cut line at ``color_threshold``
    (in distance units, i.e. already multiplied by ``Z[:, 2].max()`` — the
    stage returns both the fraction and the absolute cut).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    dendrogram(Z, color_threshold=color_threshold, ax=ax)
    ax.axhline(color_threshold, linestyle="--", linewidth=2)
    ax.text(
        0.99, color_threshold, f" cut @ {color_threshold:.3g}",
        transform=ax.get_yaxis_transform(),
        ha="right", va="bottom",
    )
    ax.set_title(title)
    ax.set_xlabel("ROI")
    ax.set_ylabel("Linkage distance")
    fig.tight_layout()
    return fig


__all__ = ["plot_dendrogram"]
