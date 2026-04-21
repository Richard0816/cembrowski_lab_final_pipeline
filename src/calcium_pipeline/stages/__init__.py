"""
calcium_pipeline.stages
-----------------------
One module per pipeline stage. Every stage exposes the same contract::

    def run(
        recording: Recording | str | Path,
        config: PipelineConfig,
        progress_callback=None,
    ) -> dict

``progress_callback`` is optional. When set it receives
:class:`orchestration.progress.ProgressEvent` instances — so the GUI can bind
progress bars, logs, and cancellation to any stage without the stage knowing
Qt exists.

The return dict is a stage-specific "artifacts" manifest (paths, counts,
timing) that the orchestrator can pretty-print and the GUI can summarise.

Stages available:

* :mod:`.signal_extraction` — raw F/Fneu -> dF/F, lowpass, SG derivative memmaps
* :mod:`.cellfilter_infer`  — trained CNN scores every ROI + writes mask
* :mod:`.apply_cellfilter`  — slice memmaps by predicted mask -> filtered memmaps
* :mod:`.events`            — per-ROI onsets + population density events
* :mod:`.clustering`        — ward hierarchical clustering of ROI traces
* :mod:`.spatial_heatmap`   — coactivation-bin order maps and cell scores
* :mod:`.crosscorrelation`  — pairwise cross-correlations, surrogates, FDR
"""
from __future__ import annotations

STAGE_ORDER = (
    "signal_extraction",
    "cellfilter_infer",
    "apply_cellfilter",
    "events",
    "clustering",
    "spatial_heatmap",
    "crosscorrelation",
)

__all__ = ["STAGE_ORDER", "get_stage"]


def get_stage(name: str):
    """
    Return the stage module's ``run`` function. Raises ``KeyError`` on unknown
    stage name. Used by :func:`orchestration.runner.run_pipeline`.
    """
    if name == "signal_extraction":
        from .signal_extraction import run
        return run
    if name == "cellfilter_infer":
        from .cellfilter_infer import run
        return run
    if name == "apply_cellfilter":
        from .apply_cellfilter import run
        return run
    if name == "events":
        from .events import run
        return run
    if name == "clustering":
        from .clustering import run
        return run
    if name == "spatial_heatmap":
        from .spatial_heatmap import run
        return run
    if name == "crosscorrelation":
        from .crosscorrelation import run
        return run
    raise KeyError(f"Unknown stage {name!r}. Known: {STAGE_ORDER}")
