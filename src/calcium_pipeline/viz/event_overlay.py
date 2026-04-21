"""
Shade population-event windows over any matplotlib time-axis plot.
"""
from __future__ import annotations

import numpy as np


def shade_events(
    ax,
    event_windows: np.ndarray,
    *,
    color: str = "C1",
    alpha: float = 0.20,
):
    """
    Overlay shaded ``[start_s, end_s]`` windows onto ``ax``.
    ``event_windows`` is ``(n_events, 2)`` as written by
    :mod:`stages.events`.
    """
    for s, e in event_windows:
        ax.axvspan(s, e, color=color, alpha=alpha)


__all__ = ["shade_events"]
