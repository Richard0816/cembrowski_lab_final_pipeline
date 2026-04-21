"""Pipeline step definitions.

Each step is a function `fn(recording_path: str, progress: ProgressReporter) -> None`
that calls into the real ported modules under `calcium_core`. Steps form a
registry so the CLI / GUI can select subsets (e.g. rerun just clustering).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

from .progress import NullReporter, ProgressReporter


@dataclass(frozen=True)
class Step:
    name: str
    fn: Callable[[str, ProgressReporter], None]
    description: str = ""


def _suite2p(recording_path: str, progress: ProgressReporter) -> None:
    from calcium_core.core.suite2p_runner import run_suite2p_on_folder
    import numpy as np
    from ..core.config import Suite2pRunConfig

    cfg = Suite2pRunConfig(
        path_to_ops=r"E:\suite2p_2p_ops_240621.npy",
        aav_info_csv="human_SLE_2p_meta.csv",
    )
    ops = np.load(cfg.path_to_ops, allow_pickle=True).item()
    progress.report("suite2p", 0.0, "running Suite2p on recording")
    run_suite2p_on_folder(recording_path, [ops, cfg.tau_vals])
    progress.report("suite2p", 1.0)


def _analyze(recording_path: str, progress: ProgressReporter) -> None:
    from calcium_core.pipeline.analyze import run_analysis_on_folder
    progress.report("analyze", 0.0, "suite2p dF/F + lowpass + derivative")
    run_analysis_on_folder(recording_path)
    progress.report("analyze", 1.0)


def _heatmap(recording_path: str, progress: ProgressReporter) -> None:
    from calcium_core.spatial.heatmap import coactivation_maps
    progress.report("heatmap", 0.0, "co-activation + propagation maps")
    coactivation_maps(recording_path)
    progress.report("heatmap", 1.0)


def _image_all(recording_path: str, progress: ProgressReporter) -> None:
    from calcium_core.viz.summary import run_full_imaging_on_folder
    progress.report("image_all", 0.0, "per-recording summary")
    run_full_imaging_on_folder(recording_path)
    progress.report("image_all", 1.0)


def _cluster(recording_path: str, progress: ProgressReporter) -> None:
    from pathlib import Path
    from calcium_core.clustering.hierarchical import main as cluster_main
    progress.report("cluster", 0.0, "Ward linkage + leaf colouring")
    cluster_main(Path(recording_path))
    progress.report("cluster", 1.0)


def _correlate(recording_path: str, progress: ProgressReporter) -> None:
    from pathlib import Path
    from calcium_core.clustering.crosscorr import run_cluster_cross_correlations_gpu
    progress.report("correlate", 0.0, "pairwise cross-correlation")
    run_cluster_cross_correlations_gpu(Path(recording_path))
    progress.report("correlate", 1.0)


def _fft(recording_path: str, progress: ProgressReporter) -> None:
    from calcium_core.signal.spectral import run_on_folder
    progress.report("fft", 0.0)
    run_on_folder(recording_path)
    progress.report("fft", 1.0)


REGISTRY: Dict[str, Step] = {
    "suite2p":   Step("suite2p",   _suite2p,   "run Suite2p on raw TIFFs"),
    "analyze":   Step("analyze",   _analyze,   "dF/F, lowpass, derivative"),
    "heatmap":   Step("heatmap",   _heatmap,   "co-activation + propagation maps"),
    "image_all": Step("image_all", _image_all, "per-recording summary figures"),
    "cluster":   Step("cluster",   _cluster,   "hierarchical clustering"),
    "correlate": Step("correlate", _correlate, "pairwise cross-correlation"),
    "fft":       Step("fft",       _fft,       "per-ROI FFT"),
}

DEFAULT_STEPS = ("analyze", "heatmap", "image_all", "cluster", "correlate")
FULL_STEPS = ("suite2p", "analyze", "heatmap", "image_all", "cluster", "correlate")


def list_steps() -> list[str]:
    return list(REGISTRY.keys())


def get_step(name: str) -> Step:
    if name not in REGISTRY:
        raise KeyError(f"Unknown step: {name!r}. Available: {list_steps()}")
    return REGISTRY[name]


def run_step(
    name: str,
    recording_path: str,
    progress: Optional[ProgressReporter] = None,
) -> None:
    step = get_step(name)
    step.fn(recording_path, progress or NullReporter())
