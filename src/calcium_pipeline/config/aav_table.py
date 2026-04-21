"""
AAV-variant -> dF/F scale factor.

These numbers are empirical: they normalise per-ROI dF/F responses across
different GCaMP variants so downstream thresholds are invariant.

Derived from internal characterisation of our AAV stocks; edit in place if
the stocks change. The GUI does NOT edit this — it's treated as a constant
of the imaging rig.
"""
from __future__ import annotations

# Keys are case-insensitive when looked up via
# ``io.notes_lookup.aav_cleanup_and_dictionary_lookup``.
AAV_TABLE: dict[str, float] = {
    "6f": 0.7,
    "6m": 1.0,
    "6s": 1.3,
    "8m": 0.137,
}

__all__ = ["AAV_TABLE"]
