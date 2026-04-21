"""Spike / event-onset detection via hysteresis thresholding."""
from __future__ import annotations

import numpy as np


def hysteresis_onsets(z, z_hi, z_lo, fps, min_sep_s=0.0):
    """Hysteresis onset detection on a robust z-scored trace.

    - Enter active state when z >= z_hi; exit when z <= z_lo.
    - Merge onsets closer than `min_sep_s` seconds.

    Returns onset frame indices as int array.
    """
    above_hi = z >= z_hi
    onsets = []
    active = False
    for i in range(z.size):
        if not active and above_hi[i]:
            active = True
            onsets.append(i)
        elif active and z[i] <= z_lo:
            active = False
    if not onsets:
        return np.array([], dtype=int)
    onsets = np.array(onsets, dtype=int)
    if min_sep_s > 0:
        min_sep = int(min_sep_s * fps)
        merged = [onsets[0]]
        for k in onsets[1:]:
            if k - merged[-1] >= min_sep:
                merged.append(k)
        onsets = np.asarray(merged, dtype=int)
    return onsets
