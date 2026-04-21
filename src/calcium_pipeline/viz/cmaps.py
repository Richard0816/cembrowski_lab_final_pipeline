"""
Shared colormaps.

Kept in one place so the GUI can swap palettes globally (e.g. for colour-blind
safe variants) without hunting through every plot function.
"""
from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap


# Cyan -> white -> red, with grey for bad/NaN cells. Used by the spatial
# order maps (and the leadlag viz) so that "before the bin peak" and "after
# the bin peak" ROIs read opposite.
CYAN_TO_RED = LinearSegmentedColormap.from_list(
    "CyanToRed",
    [
        (0.0, "#00BFC4"),
        (0.5, "#FFFFFF"),
        (1.0, "#D62728"),
    ],
)
CYAN_TO_RED.set_bad(color="#BBBBBB")


__all__ = ["CYAN_TO_RED"]
