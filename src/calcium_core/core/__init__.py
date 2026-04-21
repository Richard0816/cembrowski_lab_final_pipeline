"""Domain model — Recording, paths, config, typed result objects."""
from __future__ import annotations

from .config import EventDetectionParams, Suite2pRunConfig
from .models import ClusterResult, DensityDiagnostics, EventTable, XCorrResult
from .paths import RECORDING_DIR_RE, find_recording_root, plane0
from .recording import Recording

__all__ = [
    "Recording",
    "RECORDING_DIR_RE",
    "find_recording_root",
    "plane0",
    "EventDetectionParams",
    "Suite2pRunConfig",
    "EventTable",
    "ClusterResult",
    "XCorrResult",
    "DensityDiagnostics",
]
