"""
Stage 5 — hierarchical (Ward) clustering of ROI traces.

Reads the low-pass ΔF/F memmap (filtered prefix) and clusters ROIs with
``scipy.cluster.hierarchy.linkage``. Writes the linkage matrix, dendrogram
ordering, and cluster-membership arrays; the actual figures and per-leaf ROI
exports are done by :mod:`viz.dendrogram` so this stage stays GUI-friendly
(no matplotlib import in the headless path).

Config (``config.clustering``):
    * ``metric``                    — pdist metric, typically ``"correlation"``
    * ``method``                    — linkage method, typically ``"ward"``
    * ``z_score_rows``              — z-score per-ROI before pdist
    * ``target_n_groups_{low,high}`` — auto-pick a cut threshold in this range
    * ``color_threshold``           — if set, override the auto-pick
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

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


STAGE_NAME = "clustering"


def _auto_color_threshold(
    Z: np.ndarray,
    target_lo: int,
    target_hi: int,
) -> float:
    """
    Pick a ``color_threshold`` fraction (of max linkage height) that yields
    between ``target_lo`` and ``target_hi`` clusters. Returns the fraction
    (0..1); multiply by ``Z[:, 2].max()`` for the raw distance.
    """
    max_h = float(Z[:, 2].max())
    if max_h <= 0:
        return 0.7

    # Sweep candidate fractions; pick one in band (middle of candidates).
    candidates = np.linspace(0.1, 0.95, 50)
    for frac in candidates:
        labels = fcluster(Z, t=frac * max_h, criterion="distance")
        n = int(labels.max())
        if target_lo <= n <= target_hi:
            return float(frac)

    # Didn't find one — return whatever gives closest to midpoint.
    mid = (target_lo + target_hi) / 2
    best = min(candidates, key=lambda f: abs(
        int(fcluster(Z, t=f * max_h, criterion="distance").max()) - mid
    ))
    return float(best)


def run(
    recording: Union[Recording, str, Path],
    config: PipelineConfig,
    progress_callback: Optional[Callable] = None,
) -> dict:
    rec = resolve_recording(recording, config)
    p = config.clustering
    prefix = config.signal_extraction["filtered_prefix"]

    with stage_scope(progress_callback, STAGE_NAME, total=4,
                     message=f"clustering ROIs of {rec.name}"):
        t0 = time.monotonic()

        # Load lowpass ΔF/F (clustering on the smoothed signal is far cleaner
        # than raw ΔF/F).
        _, low, _, T, N = open_s2p_memmaps(rec.plane0, prefix=prefix)
        X = np.asarray(low, dtype=np.float32)       # (T, N)
        emit(progress_callback, ProgressEvent(
            stage=STAGE_NAME, kind="tick", current=1, total=4,
            message=f"loaded lowpass ΔF/F ({T} frames x {N} ROIs)",
        ))

        check_cancelled(progress_callback)
        if p["z_score_rows"]:
            X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

        # pdist on ROI x ROI distances -> linkage.
        D = pdist(X.T, metric=p["metric"])
        emit(progress_callback, ProgressEvent(
            stage=STAGE_NAME, kind="tick", current=2, total=4,
            message=f"pdist done (metric={p['metric']})",
        ))

        check_cancelled(progress_callback)
        Z = linkage(D, method=p["method"])
        emit(progress_callback, ProgressEvent(
            stage=STAGE_NAME, kind="tick", current=3, total=4,
            message=f"linkage done (method={p['method']})",
        ))

        # Cluster cut.
        if p["color_threshold"] is None:
            frac = _auto_color_threshold(
                Z, p["target_n_groups_low"], p["target_n_groups_high"],
            )
        else:
            frac = float(p["color_threshold"])

        cut = frac * float(Z[:, 2].max())
        labels = fcluster(Z, t=cut, criterion="distance").astype(np.int32)
        n_clusters = int(labels.max())

        # Persist.
        out_dir = rec.cluster_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "linkage.npy", Z)
        np.save(out_dir / "cluster_labels.npy", labels)
        np.save(out_dir / "color_threshold.npy",
                np.array([frac, cut], dtype=np.float64))

        # Per-cluster ROI-index arrays (consumed by stages.spatial_heatmap and viz).
        for c in range(1, n_clusters + 1):
            np.save(out_dir / f"cluster_{c:02d}_rois.npy",
                    np.flatnonzero(labels == c).astype(np.int32))

        emit(progress_callback, ProgressEvent(
            stage=STAGE_NAME, kind="tick", current=4, total=4,
            message=f"{n_clusters} clusters at cut={cut:.3g} (frac={frac:.3f})",
        ))

        elapsed = time.monotonic() - t0

    return {
        "out_dir": out_dir,
        "n_clusters": n_clusters,
        "color_threshold_frac": float(frac),
        "color_threshold_distance": float(cut),
        "linkage": out_dir / "linkage.npy",
        "labels": out_dir / "cluster_labels.npy",
        "elapsed_s": float(elapsed),
    }


__all__ = ["run", "STAGE_NAME"]
