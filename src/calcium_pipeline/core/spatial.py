"""
Spatial-plane primitives: per-ROI scalar metrics and the "paint" operator that
projects those scalars onto the imaging plane using suite2p's per-pixel
weights (``lam``).

No matplotlib here — rendering happens in :mod:`calcium_pipeline.viz.heatmap`.
"""
from __future__ import annotations

from typing import Sequence, Union

import numpy as np

from .events import mad_z, hysteresis_onsets


def roi_metric(
    values: dict,
    which: str = "event_rate",
    t_slice: slice = slice(None),
    fps: float = 30.0,
    z_enter: float = 3.5,
    z_exit: float = 1.5,
    min_sep_s: float = 0.3,
) -> np.ndarray:
    """
    Compute a single scalar per ROI from a dict of processed memmaps.

    Parameters
    ----------
    values : dict
        Must contain:

        * ``'low'`` — (T, N) lowpass ΔF/F    (used by ``mean_dff``)
        * ``'dt'``  — (T, N) SG derivative   (used by ``peak_dz``, ``event_rate``)

        Any extra keys are ignored.
    which : str
        One of ``'event_rate'``, ``'mean_dff'``, ``'peak_dz'``.
    t_slice : slice
        Time-axis slice to restrict to (e.g. pre/post stim).
    fps : float
        Sampling rate (Hz). Used for event-rate normalisation.
    z_enter, z_exit, min_sep_s : float
        Hysteresis parameters for ``'event_rate'``.

    Returns
    -------
    (N,) float32 array
    """
    lp = values["low"][t_slice]   # (Tsel, N)
    dd = values["dt"][t_slice]    # (Tsel, N)
    Tsel = lp.shape[0]
    out = np.zeros(lp.shape[1], dtype=np.float32)

    if which == "mean_dff":
        out = np.nanmean(lp, axis=0).astype(np.float32)

    elif which == "peak_dz":
        # Per-ROI MAD z, then max over time.
        z = np.empty_like(dd, dtype=np.float32)
        for j in range(dd.shape[1]):
            zj, _, _ = mad_z(dd[:, j])
            z[:, j] = zj
        out = np.nanmax(z, axis=0).astype(np.float32)

    elif which == "event_rate":
        counts = np.zeros(dd.shape[1], dtype=np.int32)
        for j in range(dd.shape[1]):
            zj, _, _ = mad_z(dd[:, j])
            on = hysteresis_onsets(zj, z_enter, z_exit, fps, min_sep_s=min_sep_s)
            counts[j] = on.size
        duration_min = Tsel / fps / 60.0
        out = (counts / max(duration_min, 1e-9)).astype(np.float32)

    else:
        raise ValueError("metric must be one of 'event_rate', 'mean_dff', 'peak_dz'")

    return out


def paint_spatial(
    values_per_roi: np.ndarray,
    stat_list: Sequence[dict],
    Ly: int,
    Lx: int,
) -> np.ndarray:
    """
    Project per-ROI scalars onto the imaging plane using suite2p ``stat``.

    Each ROI *j* contributes ``values_per_roi[j] * stat[j]['lam']`` at the
    pixels in ``stat[j]['ypix'], stat[j]['xpix']``. Output is the lam-weighted
    average at each painted pixel; unpainted pixels remain 0.

    Parameters
    ----------
    values_per_roi : (N,) float
    stat_list : suite2p ``stat.npy`` list (N entries with keys ``ypix``, ``xpix``, ``lam``)
    Ly, Lx : int
        Imaging-plane height/width (from ``ops['Ly']`` / ``ops['Lx']``).

    Returns
    -------
    img : (Ly, Lx) float32
    """
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


def build_time_mask(
    time: np.ndarray,
    t_max: Union[float, None],
) -> Union[np.ndarray, slice]:
    """
    Return a boolean mask ``time < t_max``; if ``t_max`` is None, return
    ``slice(None)`` (i.e. keep everything).
    """
    if t_max is None:
        return slice(None)
    return np.asarray(time) < float(t_max)


__all__ = [
    "roi_metric",
    "paint_spatial",
    "build_time_mask",
]
