"""calcium_core — library for 2-photon calcium imaging analysis.

Typical app usage:

    from calcium_core.core import Recording
    from calcium_core.pipeline import run_pipeline, PrintReporter

    rec = Recording("F:/data/2p_shifted/Cx/2024-11-20_00003")
    run_pipeline(rec.root, steps=["analyze", "cluster"])

Subpackages are imported on demand — importing this module does NOT pull in
matplotlib, torch, cupy, or suite2p.
"""

__version__ = "0.1.0"
