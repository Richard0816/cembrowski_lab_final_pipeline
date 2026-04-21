"""
Stage 3 — apply cell-filter mask.

After :mod:`stages.cellfilter_infer` writes ``predicted_cell_mask.npy``, we
need physical "filtered" memmaps so downstream stages that open by prefix
pick up only the kept ROIs.

This stage slices the unfiltered ``<prefix>*.memmap.float32`` files along the
ROI axis using the boolean mask, and writes them under
``<filtered_prefix>*.memmap.float32``. For the default config that means
``r0p7_*`` → ``r0p7_filtered_*``.

Also writes a convenience ``r0p7_cell_mask_bool.npy`` next to the memmaps so
that :func:`io.memmap.open_s2p_memmaps` (which expects that filename when the
prefix ends in ``_filtered_``) can count ROIs.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np

from ..config.schema import PipelineConfig
from ..io.memmap import open_s2p_memmaps
from ..io.recording import Recording
from ..orchestration.progress import (
    ProgressEvent,
    check_cancelled,
    emit,
    stage_scope,
)
from ._common import resolve_recording


STAGE_NAME = "apply_cellfilter"


def run(
    recording: Union[Recording, str, Path],
    config: PipelineConfig,
    progress_callback: Optional[Callable] = None,
) -> dict:
    rec = resolve_recording(recording, config)
    params = config.signal_extraction

    src_prefix = params["prefix"]                  # e.g. "r0p7_"
    dst_prefix = params["filtered_prefix"]          # e.g. "r0p7_filtered_"
    plane0 = rec.plane0

    mask_path = rec.predicted_mask_path
    if not mask_path.exists():
        raise FileNotFoundError(
            f"{mask_path} not found — run stage 'cellfilter_infer' first."
        )

    with stage_scope(progress_callback, STAGE_NAME, total=3,
                     message=f"applying cellfilter mask for {rec.name}"):
        t0 = time.monotonic()

        mask_bool = np.asarray(np.load(mask_path), dtype=bool).ravel()
        kept = int(mask_bool.sum())

        # Save the bool mask under the name downstream code expects.
        np.save(plane0 / "r0p7_cell_mask_bool.npy", mask_bool)

        emit(progress_callback, ProgressEvent(
            stage=STAGE_NAME, kind="log",
            message=f"mask keeps {kept}/{mask_bool.size} ROIs",
        ))

        # Open unfiltered memmaps read-only.
        dff, low, dt, T, N = open_s2p_memmaps(plane0, prefix=src_prefix)
        if N != mask_bool.size:
            raise ValueError(
                f"Mask length {mask_bool.size} does not match N={N} in {plane0}"
            )

        out_paths = {}
        for tag, src in (("dff", dff), ("lowpass", low), ("dt", dt)):
            check_cancelled(progress_callback)

            suffix_map = {
                "dff": "dff.memmap.float32",
                "lowpass": "dff_lowpass.memmap.float32",
                "dt": "dff_dt.memmap.float32",
            }
            dst = plane0 / f"{dst_prefix}{suffix_map[tag]}"

            out_mm = np.memmap(
                dst, mode="w+", dtype="float32", shape=(T, kept),
            )
            # Slice-copy along the ROI axis. np.memmap supports fancy indexing
            # with an ndarray, and copying in one shot streams from disk.
            out_mm[:] = src[:, mask_bool]
            out_mm.flush()
            del out_mm

            out_paths[tag] = dst
            emit(progress_callback, ProgressEvent(
                stage=STAGE_NAME, kind="tick",
                message=f"wrote {dst.name}",
                current=len(out_paths), total=3,
            ))

        elapsed = time.monotonic() - t0

    return {
        "kept": kept,
        "total": int(mask_bool.size),
        "paths": out_paths,
        "mask_path": plane0 / "r0p7_cell_mask_bool.npy",
        "elapsed_s": float(elapsed),
    }


__all__ = ["run", "STAGE_NAME"]
