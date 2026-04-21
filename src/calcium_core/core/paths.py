"""Recording-directory conventions.

A recording lives under a folder named `YYYY-MM-DD_#####`. Inside are
`suite2p/plane0/` (raw + derived arrays) and optional cluster subfolders.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

RECORDING_DIR_RE = re.compile(r"\d{4}-\d{2}-\d{2}_\d+")


def find_recording_root(path: Path | str) -> Optional[Path]:
    """Walk upward from `path` until we find a folder named `YYYY-MM-DD_#####`."""
    path = Path(path)
    for p in [path, *path.parents]:
        if RECORDING_DIR_RE.fullmatch(p.name):
            return p
    return None


def plane0(recording_root: Path | str) -> Path:
    return Path(recording_root) / "suite2p" / "plane0"
