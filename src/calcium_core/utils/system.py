"""System helpers — RAM-aware batch sizing and folder iteration."""
from __future__ import annotations

import contextlib
import os
import sys
from typing import Callable, Optional, Sequence, Union

import numpy as np
import psutil

from .logging import Tee


def estimate_batch_size() -> int:
    """RAM-aware Suite2p batch size. Linear in available GB with a floor of 100."""
    available_gb = round(psutil.virtual_memory().available / (1024 ** 3), 1)
    if available_gb <= 13.5:
        return 100
    return int(20 * available_gb - 170)


def run_on_folders(
    parent_folder: str,
    func: Callable,
    log_filename: str,
    addon_vals: Optional[Sequence] = None,
    leaf_folders_only: bool = False,
) -> None:
    """Run `func` on every immediate subfolder of `parent_folder`, logging to `log_filename`.

    Each folder's stdout/stderr is appended to the same log file via `Tee`.
    If `leaf_folders_only`, folders with their own subfolders are skipped.
    """
    for entry in os.scandir(parent_folder):
        with open(log_filename, "a", encoding="utf-8") as logfile:
            tee = Tee(sys.__stdout__, logfile)
            with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                if not entry.is_dir():
                    continue
                if leaf_folders_only and any(sub.is_dir() for sub in os.scandir(entry.path)):
                    continue
                if addon_vals:
                    func(entry.path, addon_vals)
                else:
                    func(entry.path)


def build_time_mask(time: np.ndarray, t_max: Union[float, None]) -> Union[np.ndarray, slice]:
    """Boolean mask for `time < t_max`, or `slice(None)` if `t_max` is None."""
    if t_max is None:
        return slice(None)
    return np.asarray(time) < float(t_max)
