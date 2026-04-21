"""Suite2p I/O — load raw F/Fneu and open derived dF/F memmaps."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import numpy as np


def infer_orientation(F: np.ndarray) -> Tuple[int, int, bool]:
    """Return (num_frames, num_rois, time_major). time_major=True means (T, N)."""
    if F.ndim != 2:
        raise ValueError(f"Expected 2D array, got {F.shape}")
    n0, n1 = F.shape
    if n0 < n1:
        return n1, n0, False  # (N, T)
    return n0, n1, True       # (T, N)


def load_raw(root: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, int, int, bool]:
    """Load F.npy and Fneu.npy. Returns (F, Fneu, num_frames, num_rois, time_major)."""
    root = Path(root)
    F = np.load(root / "F.npy", allow_pickle=False)
    Fneu = np.load(root / "Fneu.npy", allow_pickle=False)
    if F.shape != Fneu.shape:
        raise ValueError(f"F and Fneu shapes differ: {F.shape} vs {Fneu.shape}")
    num_frames, num_rois, time_major = infer_orientation(F)
    return F, Fneu, num_frames, num_rois, time_major


def open_memmaps(
    root: Union[str, Path],
    prefix: str = "r0p7_",
) -> Tuple[np.memmap, np.memmap, np.memmap, int, int]:
    """Open ΔF/F, low-pass ΔF/F, and derivative memmaps.

    Returns (dff, low, dt, T, N). All memmaps are (T, N) float32 read-only.
    """
    root = Path(root)
    F, _, num_frames, num_rois, _ = load_raw(root)

    # When the prefix signals a filtered variant, apply the cell mask.
    if prefix.split("_")[-2] == "filtered":
        mask = np.load(root / "r0p7_cell_mask_bool.npy", allow_pickle=False)
        num_rois = int(mask.sum())

    dff = np.memmap(root / f"{prefix}dff.memmap.float32", dtype="float32", mode="r", shape=(num_frames, num_rois))
    low = np.memmap(root / f"{prefix}dff_lowpass.memmap.float32", dtype="float32", mode="r", shape=(num_frames, num_rois))
    dt = np.memmap(root / f"{prefix}dff_dt.memmap.float32", dtype="float32", mode="r", shape=(num_frames, num_rois))
    return dff, low, dt, num_frames, num_rois
