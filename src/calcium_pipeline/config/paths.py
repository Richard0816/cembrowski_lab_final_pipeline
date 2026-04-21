"""
Filesystem locations used by the pipeline.

All paths default to the lab workstation's layout under ``F:\\``. The GUI
reads these at startup and lets the user override any of them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_DATA_ROOT = Path(r"F:\data\2p_shifted")
DEFAULT_NOTES_ROOT = Path(r"F:\notes_recordings")
DEFAULT_LABELS_CSV = Path(r"F:\roi_curation.csv")
DEFAULT_CELLFILTER_CKPT_DIR = Path(r"F:\cellfilter_checkpoints")
DEFAULT_AAV_INFO_CSV = Path(r"F:\aav_info.csv")


@dataclass
class Paths:
    """
    Container for every path the pipeline touches. Instantiate with keyword
    overrides to customise; the GUI persists a populated instance to
    ``~/.calcium_pipeline/paths.json``.
    """
    data_root: Path = DEFAULT_DATA_ROOT
    notes_root: Path = DEFAULT_NOTES_ROOT
    labels_csv: Path = DEFAULT_LABELS_CSV
    cellfilter_ckpt_dir: Path = DEFAULT_CELLFILTER_CKPT_DIR
    aav_info_csv: Path = DEFAULT_AAV_INFO_CSV

    # Sub-regions to iterate when listing all recordings.
    regions: tuple[str, ...] = field(default_factory=lambda: ("Cx", "Hip"))

    def __post_init__(self):
        # Normalize to Path (allows construction from str).
        for attr in (
            "data_root", "notes_root", "labels_csv",
            "cellfilter_ckpt_dir", "aav_info_csv",
        ):
            val = getattr(self, attr)
            if not isinstance(val, Path):
                setattr(self, attr, Path(val))

    # --- convenience -----------------------------------------------------

    def recording_path(self, rec_id: str) -> Path:
        """Return ``data_root/<region>/<rec_id>`` for the first region that contains it."""
        for region in self.regions:
            candidate = self.data_root / region / rec_id
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Recording {rec_id!r} not found under {self.data_root}")

    def plane0(self, rec_id: str) -> Path:
        return self.recording_path(rec_id) / "suite2p" / "plane0"


__all__ = [
    "Paths",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_NOTES_ROOT",
    "DEFAULT_LABELS_CSV",
    "DEFAULT_CELLFILTER_CKPT_DIR",
    "DEFAULT_AAV_INFO_CSV",
]
