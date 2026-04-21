"""
``PipelineConfig`` — the single object passed into every stage's ``run(...)``.

The GUI binds its widgets to the ``PipelineConfig`` instance's fields so that
every stage re-runs with whatever the user has tweaked. Stage functions should
NOT accept scattered keyword arguments; they take this whole object and dip
into the relevant section themselves. That keeps the GUI wiring trivial: one
config-editor panel, N stage panels, no per-stage parameter dialogs.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

from .aav_table import AAV_TABLE
from .defaults import DEFAULTS
from .paths import Paths


@dataclass
class PipelineConfig:
    """
    Frozen-by-convention container for every tunable the pipeline exposes.

    Fields
    ------
    paths : Paths
        Filesystem roots. See :mod:`config.paths`.
    signal_extraction, cellfilter, events, clustering, spatial_heatmap,
    crosscorrelation, orchestration : dict
        Per-stage hyperparameter dicts. Defaults come from :mod:`config.defaults`.
    aav_table : dict[str, float]
        AAV variant -> dF/F scale. Defaults to :data:`config.aav_table.AAV_TABLE`.
    fps_override, zoom_override : float | None
        If set, bypass the notes-xlsx lookup for every recording. Handy in the
        GUI for ad-hoc analysis on data that isn't in the lab notebook.
    """

    paths: Paths = field(default_factory=Paths)

    signal_extraction: dict[str, Any] = field(
        default_factory=lambda: copy.deepcopy(DEFAULTS["signal_extraction"]))
    cellfilter: dict[str, Any] = field(
        default_factory=lambda: copy.deepcopy(DEFAULTS["cellfilter"]))
    events: dict[str, Any] = field(
        default_factory=lambda: copy.deepcopy(DEFAULTS["events"]))
    clustering: dict[str, Any] = field(
        default_factory=lambda: copy.deepcopy(DEFAULTS["clustering"]))
    spatial_heatmap: dict[str, Any] = field(
        default_factory=lambda: copy.deepcopy(DEFAULTS["spatial_heatmap"]))
    crosscorrelation: dict[str, Any] = field(
        default_factory=lambda: copy.deepcopy(DEFAULTS["crosscorrelation"]))
    orchestration: dict[str, Any] = field(
        default_factory=lambda: copy.deepcopy(DEFAULTS["orchestration"]))

    aav_table: dict[str, float] = field(
        default_factory=lambda: dict(AAV_TABLE))

    fps_override: Union[float, None] = None
    zoom_override: Union[float, None] = None

    # --- helpers ---------------------------------------------------------

    def for_stage(self, stage: str) -> dict[str, Any]:
        """Return the parameter dict for a given stage name."""
        try:
            return getattr(self, stage)
        except AttributeError:
            raise KeyError(f"Unknown stage {stage!r}")

    def update_stage(self, stage: str, **kwargs) -> None:
        """In-place update of a single stage's parameters. GUI uses this."""
        target = self.for_stage(stage)
        target.update(kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Serialisable snapshot — for ``json.dump`` or run manifests."""
        return {
            "paths": {k: str(v) for k, v in self.paths.__dict__.items()
                      if isinstance(v, (str, Path, tuple))},
            "signal_extraction": dict(self.signal_extraction),
            "cellfilter": dict(self.cellfilter),
            "events": dict(self.events),
            "clustering": dict(self.clustering),
            "spatial_heatmap": dict(self.spatial_heatmap),
            "crosscorrelation": dict(self.crosscorrelation),
            "orchestration": dict(self.orchestration),
            "aav_table": dict(self.aav_table),
            "fps_override": self.fps_override,
            "zoom_override": self.zoom_override,
        }


__all__ = ["PipelineConfig"]
