"""Per-ROI summary metrics and paint-onto-image helper."""
from __future__ import annotations

import numpy as np

from ..signal.normalize import mad_z
from ..signal.spikes import hysteresis_onsets


def roi_metric(
    values,
    which: str = "event_rate",
    t_slice: slice = slice(None),
    fps: float = 30.0,
    z_enter: float = 3.5,
    z_exit: float = 1.5,
    min_sep_s: float = 0.3,
) -> np.ndarray:
    """Per-ROI scalar metric.

    `values` is a dict with 'low' (lowpass dF/F, (T, N)) and 'dt' (derivative).
    `which`: 'event_rate' (events/min), 'mean_dff', or 'peak_dz'.
    """
    lp = values["low"][t_slice]
    dd = values["dt"][t_slice]
    Tsel = lp.shape[0]
    out = np.zeros(lp.shape[1], dtype=np.float32)

    if which == "mean_dff":
        return np.nanmean(lp, axis=0).astype(np.float32)

    if which == "peak_dz":
        z = np.empty_like(dd, dtype=np.float32)
        for j in range(dd.shape[1]):
            zj, _, _ = mad_z(dd[:, j])
            z[:, j] = zj
        return np.nanmax(z, axis=0).astype(np.float32)

    if which == "event_rate":
        counts = np.zeros(dd.shape[1], dtype=np.int32)
        for j in range(dd.shape[1]):
            zj, _, _ = mad_z(dd[:, j])
            on = hysteresis_onsets(zj, z_enter, z_exit, fps, min_sep_s=min_sep_s)
            counts[j] = on.size
        duration_min = Tsel / fps / 60.0
        return (counts / max(duration_min, 1e-9)).astype(np.float32)

    raise ValueError("metric must be one of: 'event_rate', 'mean_dff', 'peak_dz'")


def paint_spatial(values_per_roi, stat_list, Ly: int, Lx: int) -> np.ndarray:
    """Paint per-ROI scalars onto the image plane using 'lam'-weighted assignment."""
    img = np.zeros((Ly, Lx), dtype=np.float32)
    w = np.zeros((Ly, Lx), dtype=np.float32)

    for j, s in enumerate(stat_list):
        v = values_per_roi[j]
        ypix = s["ypix"]
        xpix = s["xpix"]
        lam = s["lam"].astype(np.float32)
        img[ypix, xpix] += v * lam
        w[ypix, xpix] += lam

    m = w > 0
    img[m] /= w[m]
    return img
