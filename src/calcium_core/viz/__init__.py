"""Matplotlib rendering — optional, separated from compute.

Matplotlib is imported lazily inside each submodule. The app does not need
to import this subpackage unless it wants to render pre-built figures.
"""
from __future__ import annotations

__all__: list[str] = []
