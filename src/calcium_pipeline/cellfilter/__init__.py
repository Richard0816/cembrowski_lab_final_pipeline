"""
calcium_pipeline.cellfilter
---------------------------
Dual-branch CNN that classifies each suite2p ROI as cell / not-cell.

Submodules are imported on demand so `import calcium_pipeline.cellfilter`
does not force-load torch, pandas, etc. when you only need one piece.

Public surface used by the pipeline / GUI:
    - CellFilter          : the nn.Module
    - predict_recording   : run the trained model on one recording
    - config              : tunables (paths, hyperparameters, thresholds)
    - load_labels, ROIDataset, split_by_recording   (training helpers)
"""
from . import config  # re-export as a submodule

__all__ = [
    "config",
    "CellFilter",
    "predict_recording",
    "ROIDataset",
    "load_labels",
    "split_by_recording",
]


def __getattr__(name):  # lazy re-exports so torch/pandas aren't loaded eagerly
    if name == "CellFilter":
        from .model import CellFilter
        return CellFilter
    if name == "predict_recording":
        from .predict import predict_recording
        return predict_recording
    if name in ("ROIDataset", "load_labels", "split_by_recording"):
        from . import dataset as _d
        return getattr(_d, name)
    raise AttributeError(name)
