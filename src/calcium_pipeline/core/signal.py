"""
Per-trace signal-processing primitives.

Everything here is 1-D and numpy-only. The stages apply these column-by-column
(per ROI) in batches so this module never has to know about the (T, N) layout.

Functions
---------
- :func:`robust_df_over_f_1d` — rolling-percentile ΔF/F, low-RAM.
- :func:`lowpass_causal_1d`   — causal SOS Butterworth low-pass (returns state).
- :func:`sg_first_derivative_1d` — Savitzky-Golay smoothed first derivative.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import percentile_filter
from scipy.signal import butter, sosfilt, savgol_filter


# --------------------------------------------------------------------------

def robust_df_over_f_1d(
    F: np.ndarray,
    win_sec: float = 45.0,
    perc: float = 10.0,
    fps: float = 30.0,
) -> np.ndarray:
    """
    Compute ΔF/F₀ with a rolling-percentile baseline (low-memory, 1-D).

    The baseline ``F0`` is a percentile filter of width ``win_sec * fps``
    (forced to be odd, at most the length of ``F``). The normalizer is
    ``max(percentile(F0, 1%), 1e-9)`` — this keeps the denominator stable
    when the baseline dips near zero.

    NaN / inf samples in ``F`` are linearly interpolated before filtering.
    """
    F = np.asarray(F, dtype=np.float32)
    n = F.size

    finite = np.isfinite(F)
    if not finite.all():
        F = np.interp(
            np.arange(n), np.flatnonzero(finite), F[finite]
        ).astype(np.float32)

    win = max(3, int(win_sec * fps) | 1)  # odd
    win = min(win, n if n % 2 == 1 else n - 1)

    if win < 3:
        F0 = np.full_like(F, np.nanpercentile(F, perc))
    else:
        F0 = percentile_filter(F, size=win, percentile=perc, mode="nearest").astype(np.float32)

    eps = np.nanpercentile(F0, 1) if np.isfinite(F0).any() else 1.0
    eps = max(eps, 1e-9)
    return (F - F0) / eps


# --------------------------------------------------------------------------

def lowpass_causal_1d(
    x: np.ndarray,
    fps: float,
    cutoff_hz: float = 5.0,
    order: int = 2,
    zi: np.ndarray | None = None,
    sos: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Causal second-order-sections Butterworth low-pass filter.

    Returns ``(y, zf, sos)``:

    * ``y``   — filtered signal, shape ``x.shape``, float32.
    * ``zf``  — final filter state, shape ``(n_sections, 2)``. Pass back as
                ``zi`` to resume filtering across chunks without transients.
    * ``sos`` — the SOS array; pass back in as ``sos`` to avoid rebuilding it.

    Cutoff is clipped to ``[1e-4, 0.95 * Nyquist]`` for numerical safety.
    """
    x = np.asarray(x, dtype=np.float32)
    n = x.size
    if n < 3:
        return x.copy(), zi, sos

    nyq = fps / 2.0
    cutoff = min(max(1e-4, cutoff_hz), 0.95 * nyq)

    if sos is None:
        sos = butter(order, cutoff / nyq, btype="low", output="sos")

    if zi is None:
        # Initialize state with the first sample — avoids a big transient
        # at t=0 when the caller starts filtering from silence.
        zi = np.zeros((sos.shape[0], 2), dtype=np.float32)
        zi[:, 0] = x[0]
        zi[:, 1] = x[0]

    y, zf = sosfilt(sos, x, zi=zi)
    return y.astype(np.float32), zf.astype(np.float32), sos


# --------------------------------------------------------------------------

def sg_first_derivative_1d(
    x: np.ndarray,
    fps: float,
    win_ms: float = 333.0,
    poly: int = 3,
) -> np.ndarray:
    """
    Savitzky-Golay smoothed first derivative (dF/F / sec).

    The window length is ``win_ms * fps / 1000`` samples, forced to be odd and
    clipped to the signal length. For signals shorter than 3 samples (or
    windows shorter than 3) we fall back to a forward-difference gradient.
    """
    x = np.asarray(x, dtype=np.float32)
    n = x.size

    win = max(3, int((win_ms / 1000.0) * fps) | 1)
    if win >= n:
        win = max(3, (n - (1 - n % 2)))  # largest valid odd <= n

    if win < 3 or n < 3:
        g = np.empty_like(x)
        g[0] = 0.0
        g[1:] = (x[1:] - x[:-1]) * fps
        return g

    return savgol_filter(
        x,
        window_length=win,
        polyorder=poly,
        deriv=1,
        delta=1.0 / fps,
    ).astype(np.float32)


__all__ = [
    "robust_df_over_f_1d",
    "lowpass_causal_1d",
    "sg_first_derivative_1d",
]
