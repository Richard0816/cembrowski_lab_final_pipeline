"""
Recording-metadata lookup: frame rate, zoom, and AAV-variant tag.

All three live in the lab's ``F:\\notes_recordings\\YYYY-MM-DD.xlsx`` notebook
(sheet ``"2P settings"``), keyed by recording filename ``YYYY-MM-DD-#####``.
The functions here are deliberately forgiving — they never raise on missing
files or missing columns; they return the caller's ``default_*`` value. This
keeps the GUI usable even when the network drive is unavailable.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Union

import pandas as pd


# Matches suite2p-style recording folder names like ``2024-07-01_00018``.
RECORDING_DIR_RE = re.compile(r"\d{4}-\d{2}-\d{2}_\d+")


# --------------------------------------------------------------------------
# Path helpers
# --------------------------------------------------------------------------

def find_recording_root(path: Union[str, Path]) -> Path | None:
    """
    Walk upward from ``path`` until we find a folder named ``YYYY-MM-DD_#####``.
    Returns ``None`` if no such ancestor exists.
    """
    path = Path(path)
    for p in [path] + list(path.parents):
        if RECORDING_DIR_RE.fullmatch(p.name):
            return p
    return None


def _resolve_notes_xlsx(notes_root: Path, date_str: str) -> Path | None:
    """Locate the daily notes workbook for a given ``YYYY-MM-DD``."""
    direct = notes_root / f"{date_str}.xlsx"
    if direct.exists():
        return direct
    candidates = sorted(notes_root.glob(f"*{date_str}*.xlsx"))
    return candidates[0] if candidates else None


def _lookup_2p_setting(
    path: Union[str, Path],
    column_lc: str,
    *,
    notes_root: Union[str, Path],
    default,
):
    """
    Shared machinery for :func:`get_fps_from_notes` and :func:`get_zoom_from_notes`.

    Returns ``default`` on *any* failure (missing path, missing file, missing
    column, NaN cell, unexpected exception).
    """
    try:
        rec_root = find_recording_root(path)
        if rec_root is None:
            return default

        date_str, rec_str = rec_root.name.split("_", 1)
        target = f"{date_str}-{rec_str}"

        notes_path = _resolve_notes_xlsx(Path(notes_root), date_str)
        if notes_path is None:
            return default

        df = pd.read_excel(notes_path, sheet_name="2P settings")
        df.columns = [str(c).strip() for c in df.columns]
        cols = {c.lower(): c for c in df.columns}

        if "filename" not in cols or column_lc not in cols:
            return default

        fn_col = cols["filename"]
        val_col = cols[column_lc]

        fn = df[fn_col].astype(str).str.strip()

        hits = df.loc[fn == target]
        if hits.empty:
            # Fuzzier fallback — some rows are saved as just ``#####``
            # or ``something-#####``.
            hits = df.loc[
                fn.str.endswith(f"-{rec_str}", na=False) | (fn == rec_str)
            ]
        if hits.empty:
            return default

        val = hits.iloc[0][val_col]
        if pd.isna(val):
            return default
        return float(val)

    except Exception:
        return default


def get_fps_from_notes(
    path: Union[str, Path],
    notes_root: Union[str, Path] = r"F:\notes_recordings",
    default_fps: float = 15.0,
) -> float:
    """
    Return the 2-photon frame rate for the recording that ``path`` lives in.
    Falls back to ``default_fps`` if the notes xlsx can't be read.
    """
    return _lookup_2p_setting(
        path, "rate (hz)", notes_root=notes_root, default=default_fps
    )


def get_zoom_from_notes(
    path: Union[str, Path],
    notes_root: Union[str, Path] = r"F:\notes_recordings",
    default_zoom: float = 1.0,
) -> float:
    """
    Return the objective zoom factor for the recording that ``path`` lives in.
    Falls back to ``default_zoom`` if the notes xlsx can't be read.
    """
    return _lookup_2p_setting(
        path, "zoom", notes_root=notes_root, default=default_zoom
    )


# --------------------------------------------------------------------------
# AAV variant -> scaling factor lookup
# --------------------------------------------------------------------------

def aav_cleanup_and_dictionary_lookup(aav: str, dic: dict) -> float:
    """
    Map a raw AAV-variant string (e.g. ``"AAV1-CaMKII-GCaMP6f"``) to a numeric
    tag via ``dic`` (e.g. ``{"6f": 0.7, "6m": 1.0, "6s": 1.3, "8m": 0.137}``).

    Strategy:
      1. Drop the ``"rg"`` substring (red-shifted tag we don't care about).
      2. Split on ``-``, ``_``, ``+``.
      3. Intersect (case-insensitive) with the dict's keys.
      4. Return the first matching value.
    """
    aav = aav.replace("rg", "")
    components = re.split(r"[-_+]", aav)

    dict_lower = {k.lower(): v for k, v in dic.items()}
    list_lower = {item.lower() for item in components}

    common = dict_lower.keys() & list_lower
    if not common:
        raise KeyError(f"No AAV variant in {aav!r} matched dictionary keys {list(dic)!r}")
    return dict_lower[next(iter(common))]


def _get_row_number_csv_module(
    csv_filename: Union[str, Path],
    header_name: str,
    target_element: str,
) -> int | None:
    """
    Find the 1-based row index (header excluded) where ``target_element``
    appears in the column ``header_name``. Comparison ignores ``-`` / ``_``
    delimiters so ``"2024-07-01-00018"`` matches ``"2024_07_01_00018"``.
    """
    try:
        col = pd.read_csv(csv_filename, usecols=[header_name])
    except ValueError:
        return None

    def to_int_list(s: str):
        return [int(x) for x in re.split(r"[-_]", str(s)) if x.isdigit()]

    target_list = to_int_list(target_element)

    for idx, val in col[header_name].items():
        if to_int_list(val) == target_list:
            return idx + 1
    return None


def file_name_to_aav_to_dictionary_lookup(
    file_name: str,
    aav_info_csv: Union[str, Path],
    dic: dict,
) -> float:
    """
    Given a recording filename and a CSV that maps ``video`` -> ``AAV``, return
    the numeric tag from ``dic`` for that recording's AAV variant.
    """
    row_num = _get_row_number_csv_module(aav_info_csv, "video", file_name)
    if row_num is None:
        raise LookupError(f"{file_name!r} not found in {aav_info_csv}")

    col = pd.read_csv(aav_info_csv, usecols=["AAV"])
    element = str(col["AAV"].iloc[row_num - 1])
    return aav_cleanup_and_dictionary_lookup(element, dic)


__all__ = [
    "RECORDING_DIR_RE",
    "find_recording_root",
    "get_fps_from_notes",
    "get_zoom_from_notes",
    "aav_cleanup_and_dictionary_lookup",
    "file_name_to_aav_to_dictionary_lookup",
]
