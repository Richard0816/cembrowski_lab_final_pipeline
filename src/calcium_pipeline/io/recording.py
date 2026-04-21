"""
``Recording`` — one object that bundles every path + metadatum for a single
2-photon session, so stages never have to do path arithmetic themselves.

Constructed either from a raw ``pathlib.Path`` or from a
``(rec_id, paths: config.Paths)`` pair via :meth:`Recording.from_id`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from . import notes_lookup


class Recording:
    """
    Wraps a single recording directory and exposes all derived paths plus
    metadata (fps, zoom) resolved from the lab's notes spreadsheets.

    Parameters
    ----------
    path : str | Path
        Full path to the recording root, e.g.
        ``F:\\data\\2p_shifted\\Cx\\2024-07-01_00018``.
    prefix : str
        Memmap / output prefix used throughout the pipeline. Defaults to
        ``"r0p7_filtered_"`` (neuropil r=0.7, cellfilter-mask applied).
    notes_root : str | Path
        Root directory of the Excel metadata sheets. Defaults to
        ``F:\\notes_recordings``.
    fps_override, zoom_override : float | None
        If set, bypass notes-xlsx lookup and return this value directly.
    """

    DEFAULT_PREFIX = "r0p7_filtered_"
    DEFAULT_NOTES_ROOT = r"F:\notes_recordings"

    def __init__(
        self,
        path: Union[str, Path],
        prefix: str = DEFAULT_PREFIX,
        notes_root: Union[str, Path] = DEFAULT_NOTES_ROOT,
        fps_override: Optional[float] = None,
        zoom_override: Optional[float] = None,
    ) -> None:
        self.path = Path(path).resolve()
        self.prefix = prefix
        self._notes_root = notes_root
        self._fps_override = fps_override
        self._zoom_override = zoom_override

        if not self.path.exists():
            raise FileNotFoundError(f"Recording path does not exist: {self.path}")

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_id(cls, rec_id: str, paths, **kwargs) -> "Recording":
        """
        Build a ``Recording`` from a bare ID like ``"2024-07-01_00018"`` and a
        :class:`config.paths.Paths` object — resolving the region folder
        (``Cx`` / ``Hip``) via ``paths.recording_path(rec_id)``.
        """
        return cls(
            path=paths.recording_path(rec_id),
            notes_root=paths.notes_root,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Folder name, e.g. ``2024-07-01_00018``."""
        return self.path.name

    def __str__(self) -> str:
        return str(self.path)

    def __repr__(self) -> str:
        return f"Recording({self.path!r})"

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    @property
    def plane0(self) -> Path:
        return self.path / "suite2p" / "plane0"

    @property
    def cluster_dir(self) -> Path:
        """``<plane0>/<prefix>cluster_results``"""
        return self.plane0 / f"{self.prefix}cluster_results"

    @property
    def cell_mask_path(self) -> Path:
        return self.plane0 / "r0p7_cell_mask_bool.npy"

    @property
    def predicted_prob_path(self) -> Path:
        return self.plane0 / "predicted_cell_prob.npy"

    @property
    def predicted_mask_path(self) -> Path:
        return self.plane0 / "predicted_cell_mask.npy"

    def plane0_file(self, name: str) -> Path:
        """Convenience: ``<plane0>/<name>``."""
        return self.plane0 / name

    # ------------------------------------------------------------------
    # Metadata (lazy, cached)
    # ------------------------------------------------------------------

    @property
    def fps(self) -> float:
        if self._fps_override is not None:
            return float(self._fps_override)
        return notes_lookup.get_fps_from_notes(
            self.path, notes_root=self._notes_root,
        )

    @property
    def zoom(self) -> float:
        if self._zoom_override is not None:
            return float(self._zoom_override)
        return notes_lookup.get_zoom_from_notes(
            self.path, notes_root=self._notes_root,
        )

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def n_clusters(self) -> int:
        """Number of cluster ROI files (``*_rois.npy``) in cluster_dir."""
        if not self.cluster_dir.exists():
            return 0
        return len(list(self.cluster_dir.glob("*_rois.npy")))

    def has_processed_traces(self) -> bool:
        """True iff all three processed memmap files for this prefix exist."""
        suffixes = [
            f"{self.prefix}dff.memmap.float32",
            f"{self.prefix}dff_lowpass.memmap.float32",
            f"{self.prefix}dff_dt.memmap.float32",
        ]
        return all((self.plane0 / s).exists() for s in suffixes)

    def has_clustering(self) -> bool:
        return self.cluster_dir.exists() and self.n_clusters() > 0

    def has_cellfilter_output(self) -> bool:
        return self.predicted_prob_path.exists() and self.predicted_mask_path.exists()


__all__ = ["Recording"]
