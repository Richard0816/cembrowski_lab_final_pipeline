"""
Stage 6 — spatial heatmap / coactivation-bin order maps.

Given the event-detection output and suite2p ``stat``, this stage produces:

* per-time-bin **order maps** (painted on the imaging plane, one ROI coloured
  by its activation rank within that coactivation bin)
* **cell scores** for each ROI (logistic combination of event rate, peak dz,
  and ROI area) with an edge-mask penalty for ROIs near the FOV border
* **lead/lag spatial splits** (propagation vectors, displayed in µm)

The heavy lifting lives in :mod:`viz.order_map` and
:mod:`core.spatial.paint_spatial`; this stage ties them together and writes
the artifacts to ``<plane0>/<prefix>spatial/``.

.. note::
   This is a scaffold — the exact numerical recipe (coactivation-bin selection
   with iterative relax/tighten, propagation-vector angle computation) lives
   in the legacy ``spatial_heatmap.py`` and will be ported in a follow-up
   commit. The ``run`` signature and artifact layout are fixed here so the
   GUI and orchestrator can integrate against them today.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np

from ..config.schema import PipelineConfig
from ..core.spatial import paint_spatial, roi_metric
from ..io.memmap import open_s2p_memmaps
from ..io.recording import Recording
from ..orchestration.progress import (
    ProgressEvent,
    check_cancelled,
    emit,
    stage_scope,
)
from ._common import resolve_fps, resolve_recording


STAGE_NAME = "spatial_heatmap"


def run(
    recording: Union[Recording, str, Path],
    config: PipelineConfig,
    progress_callback: Optional[Callable] = None,
) -> dict:
    rec = resolve_recording(recording, config)
    fps = resolve_fps(rec, config)
    p = config.spatial_heatmap
    prefix = config.signal_extraction["filtered_prefix"]

    plane0 = rec.plane0
    out_dir = plane0 / f"{prefix}spatial"
    out_dir.mkdir(exist_ok=True)

    with stage_scope(progress_callback, STAGE_NAME, total=3,
                     message=f"spatial maps for {rec.name}"):
        t0 = time.monotonic()

        # --- load inputs ----------------------------------------------------
        dff, low, dt, T, N = open_s2p_memmaps(plane0, prefix=prefix)
        stat = np.load(plane0 / "stat.npy", allow_pickle=True)
        ops = np.load(plane0 / "ops.npy", allow_pickle=True).item()
        Ly = int(ops["Ly"])
        Lx = int(ops["Lx"])

        # Honour the cellfilter mask on ``stat`` so that stat entries line up
        # 1:1 with ROI columns in the filtered memmaps.
        mask_path = plane0 / "r0p7_cell_mask_bool.npy"
        if mask_path.exists():
            mask = np.load(mask_path).astype(bool)
            stat = stat[mask]

        emit(progress_callback, ProgressEvent(
            stage=STAGE_NAME, kind="tick", current=1, total=3,
            message=f"inputs loaded (N={N}, FOV={Ly}x{Lx})",
        ))

        # --- per-ROI event-rate map -----------------------------------------
        check_cancelled(progress_callback)
        er = roi_metric(
            {"low": low, "dt": dt},
            which="event_rate",
            fps=fps,
        )
        er_img = paint_spatial(er, stat, Ly, Lx)
        np.save(out_dir / "event_rate_per_roi.npy", er)
        np.save(out_dir / "event_rate_spatial.npy", er_img)
        emit(progress_callback, ProgressEvent(
            stage=STAGE_NAME, kind="tick", current=2, total=3,
            message="event-rate spatial map written",
        ))

        # --- TODO: coactivation-bin order maps, cell scores, propagation ----
        # Ported from spatial_heatmap.coactivation_order_heatmaps /
        # compute_cell_scores / _compute_propagation_vector_for_bin.
        emit(progress_callback, ProgressEvent(
            stage=STAGE_NAME, kind="tick", current=3, total=3,
            message="order-map port pending — see TODO in spatial_heatmap.py",
        ))

        elapsed = time.monotonic() - t0

    return {
        "out_dir": out_dir,
        "event_rate_image": out_dir / "event_rate_spatial.npy",
        "elapsed_s": float(elapsed),
    }


__all__ = ["run", "STAGE_NAME"]
