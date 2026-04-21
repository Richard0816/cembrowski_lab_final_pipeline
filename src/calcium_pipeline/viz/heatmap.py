"""
(T, N) ΔF/F / lowpass / derivative heatmaps.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_dff_heatmap(
    X: np.ndarray,
    *,
    fps: float,
    cmap: str = "magma",
    vmax_quantile: float = 0.99,
    order: np.ndarray | None = None,
    title: str = "",
    ax=None,
):
    """
    Plot a ``(T, N)`` matrix as a heatmap with rows = ROIs, cols = time.

    ``order`` reorders ROI rows (handy for dendrogram-ordered views). ``vmax``
    is clipped to the specified quantile to keep the few bright events from
    washing out the rest.
    """
    if order is not None:
        X = X[:, order]
    X = X.T    # (N, T) for imshow's row-major display

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, max(3, 0.02 * X.shape[0])))
    else:
        fig = ax.figure

    vmax = float(np.nanquantile(X, vmax_quantile))
    T = X.shape[1]
    im = ax.imshow(
        X, aspect="auto", cmap=cmap,
        vmin=0, vmax=max(vmax, 1e-6),
        extent=[0, T / fps, X.shape[0], 0],
        interpolation="nearest",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ROI")
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, label="ΔF/F")
    fig.tight_layout()
    return fig


__all__ = ["plot_dff_heatmap"]
