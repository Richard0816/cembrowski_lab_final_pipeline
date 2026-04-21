"""
Spatial "order map" — ROIs painted on the imaging plane, coloured by their
activation rank within a coactivation bin.

Consumed by :mod:`stages.spatial_heatmap` once the full order-map compute is
ported.
"""
from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from .cmaps import CYAN_TO_RED


def plot_order_map(
    order_img: np.ndarray,
    *,
    title: str = "",
    cbar_label: str = "Activation order (normalised)",
    ax=None,
):
    """
    ``order_img`` is a ``(Ly, Lx)`` float image; values in ``[0, 1]`` encode
    ROI rank within a bin, NaN marks unpainted pixels. Uses the shared
    cyan-white-red colormap with grey-for-bad so NaN pixels read as
    background.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.figure

    im = ax.imshow(
        np.ma.masked_invalid(order_img),
        cmap=CYAN_TO_RED, vmin=0, vmax=1,
        interpolation="nearest",
    )
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.75, label=cbar_label)
    fig.tight_layout()
    return fig


__all__ = ["plot_order_map"]
