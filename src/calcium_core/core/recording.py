"""Recording — the central domain object the app binds UI to."""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Optional

import numpy as np

from .paths import RECORDING_DIR_RE, plane0


@dataclass
class Recording:
    """Wrap a Suite2p recording directory `YYYY-MM-DD_#####`.

    Lazily reads metadata (fps, zoom, ops, stat) on first access. No work at
    construction so the app can cheaply instantiate Recordings for a browser.
    """

    root: Path

    def __post_init__(self):
        self.root = Path(self.root)
        if not self.root.exists():
            raise FileNotFoundError(self.root)

    @property
    def name(self) -> str:
        return self.root.name

    @property
    def is_valid_name(self) -> bool:
        return RECORDING_DIR_RE.fullmatch(self.name) is not None

    @property
    def plane0(self) -> Path:
        return plane0(self.root)

    @cached_property
    def fps(self) -> float:
        from ..io.metadata import get_fps_from_notes
        return get_fps_from_notes(str(self.root))

    @cached_property
    def zoom(self) -> float:
        from ..io.metadata import get_zoom_from_notes
        return get_zoom_from_notes(str(self.root))

    @cached_property
    def ops(self) -> dict:
        return np.load(self.plane0 / "ops.npy", allow_pickle=True).item()

    @cached_property
    def stat(self) -> np.ndarray:
        return np.load(self.plane0 / "stat.npy", allow_pickle=True)

    def has_processed_traces(self, prefix: str = "r0p7_") -> bool:
        return (self.plane0 / f"{prefix}dff.memmap.float32").exists()

    def has_clustering(self, prefix: str = "r0p7_") -> bool:
        return any(self.plane0.glob(f"{prefix}combined_cluster*"))

    def cluster_dir(self, prefix: str = "r0p7_") -> Optional[Path]:
        hits = sorted(self.plane0.glob(f"{prefix}combined_cluster*"))
        return hits[0] if hits else None

    def n_clusters(self, prefix: str = "r0p7_") -> int:
        d = self.cluster_dir(prefix)
        if d is None:
            return 0
        return sum(1 for _ in d.glob("cluster_*"))
