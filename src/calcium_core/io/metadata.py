"""Recording metadata — AAV → tau / cutoff lookup, FPS, zoom from Excel notes."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from ..core.paths import find_recording_root


def _aav_cleanup_and_dictionary_lookup(aav: str, dic: dict) -> float:
    """Return `dic[key]` where `key` is the GCaMP variant token in `aav`."""
    aav = aav.replace("rg", "")
    components = re.split(r"[-_+]", aav)
    dict_lower = {k.lower(): v for k, v in dic.items()}
    list_lower = {item.lower() for item in components}
    common = dict_lower.keys() & list_lower
    return dict_lower[next(iter(common))]


def _get_row_number(csv_filename: str, header_name: str, target_element: str):
    try:
        col = pd.read_csv(csv_filename, usecols=[header_name])
    except ValueError:
        print(f"Error: Header '{header_name}' not found in the CSV file.")
        return None

    def to_int_list(s: str):
        return [int(x) for x in re.split(r"[-_]", str(s)) if x.isdigit()]

    target_list = to_int_list(target_element)
    for idx, val in col[header_name].items():
        if to_int_list(val) == target_list:
            return idx + 1
    return None


def lookup_aav_value(file_name: str, aav_info_csv: str, dic: dict) -> float:
    """For `file_name` in `aav_info_csv`, translate the AAV column via `dic`."""
    row_num = _get_row_number(aav_info_csv, "video", file_name)
    element = str(pd.read_csv(aav_info_csv, usecols=["AAV"])["AAV"].iloc[row_num - 1])
    return _aav_cleanup_and_dictionary_lookup(element, dic)


def _read_2p_settings(date_str: str, notes_root: Path) -> pd.DataFrame | None:
    notes_path = notes_root / f"{date_str}.xlsx"
    if not notes_path.exists():
        candidates = sorted(notes_root.glob(f"*{date_str}*.xlsx"))
        if not candidates:
            return None
        notes_path = candidates[0]
    df = pd.read_excel(notes_path, sheet_name="2P settings")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _lookup_column(
    path: str,
    column_key: str,
    notes_root: str | Path,
    default: float,
) -> float:
    """Generic lookup for a per-recording column (rate, zoom, ...)."""
    try:
        p = Path(path)
        rec_root = find_recording_root(p)
        if rec_root is None:
            return default

        date_str, rec_str = rec_root.name.split("_", 1)
        target = f"{date_str}-{rec_str}"

        df = _read_2p_settings(date_str, Path(notes_root))
        if df is None:
            return default

        cols = {c.lower(): c for c in df.columns}
        if "filename" not in cols or column_key not in cols:
            return default

        fn = df[cols["filename"]].astype(str).str.strip()
        hits = df.loc[fn == target]
        if hits.empty:
            hits = df.loc[fn.str.endswith(f"-{rec_str}", na=False) | (fn == rec_str)]
        if hits.empty:
            return default

        val = hits.iloc[0][cols[column_key]]
        return default if pd.isna(val) else float(val)
    except Exception:
        return default


def get_fps_from_notes(
    path: str,
    notes_root: str | Path = r"F:\notes_recordings",
    default_fps: float = 15.0,
) -> float:
    return _lookup_column(path, "rate (hz)", notes_root, default_fps)


def get_zoom_from_notes(
    path: str,
    notes_root: str | Path = r"F:\notes_recordings",
    default_zoom: float = 1.0,
) -> float:
    return _lookup_column(path, "zoom", notes_root, default_zoom)
