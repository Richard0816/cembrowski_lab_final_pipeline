"""
calcium_pipeline.config
-----------------------
Central, GUI-editable settings. Split so the GUI only has to know four
surface objects:

* :mod:`.paths`      — filesystem locations (data root, notes workbook, etc.)
* :mod:`.defaults`   — per-stage default hyperparameters
* :mod:`.aav_table`  — AAV-variant -> dF/F-scale table
* :mod:`.schema`     — the ``PipelineConfig`` dataclass the GUI binds to
"""
from __future__ import annotations

__all__ = [
    "Paths",
    "DEFAULTS",
    "AAV_TABLE",
    "PipelineConfig",
]


def __getattr__(name):
    if name == "Paths":
        from .paths import Paths
        return Paths
    if name == "DEFAULTS":
        from .defaults import DEFAULTS
        return DEFAULTS
    if name == "AAV_TABLE":
        from .aav_table import AAV_TABLE
        return AAV_TABLE
    if name == "PipelineConfig":
        from .schema import PipelineConfig
        return PipelineConfig
    raise AttributeError(name)
