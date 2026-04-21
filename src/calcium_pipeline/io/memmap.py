"""
Suite2p I/O + memmap helpers.

These are the lowest-level primitives the rest of the pipeline builds on:
opening ``F.npy`` / ``Fneu.npy`` from a suite2p ``plane0`` folder, inferring
whether those arrays are ``(N, T)`` or ``(T, N)`` (suite2p writes the former,
but downstream code always operates on ``(T, N)``), and opening the memmap
files that ``stages.signal_extraction`` emits (``dff``, ``dff_lowpass``,
``dff_dt``).

No torch / matplotlib / pandas imports on purpose — this module is imported
very early (e.g. by ``cellfilter.dataset``) and must stay cheap.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import psutil


# --------------------------------------------------------------------------
# RAM-based batch sizing (used by suite2p wrapper)
# --------------------------------------------------------------------------

def change_batch_according_to_free_ram() -> int:
    """
    Choose a suite2p ``batch_size`` based on currently-free system RAM.

    The linear model was fit from two empirically-chosen operating points:
    ``(16 GiB -> 150 frames)`` and ``(200 GiB -> 4000 frames)``; the floor of
    100 keeps suite2p alive on machines that are nearly exhausted.
    """
    available_mem = round(psutil.virtual_memory().available / (1024 ** 3), 1)
    if available_mem <= 13.5:
        return 100
    return int(20 * available_mem - 170)


# --------------------------------------------------------------------------
# Orientation helpers
# --------------------------------------------------------------------------

def s2p_infer_orientation(F: np.ndarray) -> tuple[int, int, bool]:
    """
    Infer whether a suite2p fluorescence array is shaped ``(N, T)`` (suite2p's
    native layout) or ``(T, N)`` (our preferred layout for memmaps + clustering).

    Returns
    -------
    (T, N, time_major) : tuple
        ``time_major=True`` means the input was already ``(T, N)``.
    """
    if F.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {F.shape}")
    n0, n1 = F.shape
    if n0 < n1:
        # Native suite2p: (N_rois, T_frames)
        num_frames, num_rois, time_major = n1, n0, False
    else:
        # Already transposed: (T_frames, N_rois)
        num_frames, num_rois, time_major = n0, n1, True
    return num_frames, num_rois, time_major


def s2p_load_raw(
    root: Union[str, Path],
) -> tuple[np.ndarray, np.ndarray, int, int, bool]:
    """
    Load ``F.npy`` and ``Fneu.npy`` from a suite2p ``plane0`` folder.

    Parameters
    ----------
    root : path
        Path to ``.../suite2p/plane0``.

    Returns
    -------
    (F, Fneu, T, N, time_major)
    """
    root = Path(root)
    F = np.load(root / "F.npy", allow_pickle=False)
    Fneu = np.load(root / "Fneu.npy", allow_pickle=False)
    if F.shape != Fneu.shape:
        raise ValueError(f"F and Fneu shapes differ: {F.shape} vs {Fneu.shape}")
    num_frames, num_rois, time_major = s2p_infer_orientation(F)
    return F, Fneu, num_frames, num_rois, time_major


# --------------------------------------------------------------------------
# Processed-trace memmaps
# --------------------------------------------------------------------------

def open_s2p_memmaps(
    root: Union[str, Path],
    prefix: str = "r0p7_",
) -> tuple[np.memmap, np.memmap, np.memmap, int, int]:
    """
    Open the three processed-trace memmaps written by ``signal_extraction``:

    * ``{prefix}dff.memmap.float32``         — ΔF/F
    * ``{prefix}dff_lowpass.memmap.float32`` — causal Butterworth low-pass ΔF/F
    * ``{prefix}dff_dt.memmap.float32``      — Savitzky-Golay first derivative

    All three are shaped ``(T, N)`` in read-only mode.

    The special prefix ``"r0p7_filtered_"`` (or any prefix whose second-to-last
    token is ``"filtered"``) means the memmaps were written for only the subset
    of ROIs that passed the cell filter. In that case we reduce ``num_rois``
    using ``r0p7_cell_mask_bool.npy`` so the caller gets the correct shape.
    """
    root = Path(root)

    F, _, num_frames, num_rois, _ = s2p_load_raw(root)

    parts = prefix.split("_")
    if len(parts) >= 2 and parts[-2] == "filtered":
        mask_path = root / "r0p7_cell_mask_bool.npy"
        mask = np.load(mask_path, allow_pickle=False)
        num_rois = int(mask.sum())

    dff = np.memmap(
        root / f"{prefix}dff.memmap.float32",
        dtype="float32", mode="r",
        shape=(num_frames, num_rois),
    )
    low = np.memmap(
        root / f"{prefix}dff_lowpass.memmap.float32",
        dtype="float32", mode="r",
        shape=(num_frames, num_rois),
    )
    dt = np.memmap(
        root / f"{prefix}dff_dt.memmap.float32",
        dtype="float32", mode="r",
        shape=(num_frames, num_rois),
    )
    return dff, low, dt, num_frames, num_rois


__all__ = [
    "change_batch_according_to_free_ram",
    "s2p_infer_orientation",
    "s2p_load_raw",
    "open_s2p_memmaps",
]
