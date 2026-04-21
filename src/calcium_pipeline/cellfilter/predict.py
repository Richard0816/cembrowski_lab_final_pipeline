"""
Run the trained cell filter over every ROI in a set of recordings and write
    suite2p/plane0/predicted_cell_prob.npy   float32 (N,)   sigmoid scores
    suite2p/plane0/predicted_cell_mask.npy   bool    (N,)   prob >= THRESHOLD

Usage
-----
    python -m cell_filter.predict
        --- predicts for every recording folder found under DATA_ROOT\\{Cx,Hip}\\

    python -m cell_filter.predict --rec 2024-07-01_00018
        --- predicts for a specific recording

Accepts full traces at inference (no random crop).
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
from pathlib import Path

import numpy as np
import torch

from . import config as C
from .dataset import _RecordingCache, find_recording_root
from .model import CellFilter


@torch.no_grad()
def predict_recording(
    rec_id: str,
    model: CellFilter,
    device: torch.device,
    progress_cb=None,
) -> Path:
    """
    Score every ROI in one recording and write
        <plane0>/predicted_cell_prob.npy   (float32, N)
        <plane0>/predicted_cell_mask.npy   (bool,    N)

    Parameters
    ----------
    rec_id : str
        Folder name of the recording (e.g. ``2024-07-01_00018``).
    model : CellFilter
        Trained, already moved to the right device and in eval mode.
    device : torch.device
    progress_cb : callable(i, n, roi_prob) or None
        Optional GUI hook; called once per ROI.
    """
    rec = _RecordingCache(rec_id)
    N = rec.N

    probs = np.zeros(N, dtype=np.float32)
    for roi in range(N):
        patch = rec.get_patch(roi, C.PATCH_SIZE)          # (3, H, W)
        trace = rec.get_trace(roi)                        # (T,)
        spatial = torch.from_numpy(patch)[None].to(device)
        trace_t = torch.from_numpy(trace)[None, None].to(device)  # (1, 1, T)
        logit = model(spatial, trace_t)
        p = float(torch.sigmoid(logit).item())
        probs[roi] = p
        if progress_cb is not None:
            progress_cb(roi + 1, N, p)

    out_prob = rec.plane0 / C.PREDICTED_PROB_NAME
    out_mask = rec.plane0 / C.PREDICTED_MASK_NAME
    np.save(out_prob, probs)
    np.save(out_mask, probs >= C.THRESHOLD)
    print(f"{rec_id}: {(probs >= C.THRESHOLD).sum()}/{N} kept   "
          f"-> {out_prob.name}, {out_mask.name}")
    return out_prob


def load_model(ckpt_path: Path | None = None, device: torch.device | None = None) -> tuple[CellFilter, torch.device]:
    """Load a trained CellFilter checkpoint. Used by the GUI + orchestrator."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(ckpt_path) if ckpt_path else (C.CHECKPOINT_DIR / "best.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}. Train first.")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = CellFilter().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, device


def list_all_recordings(data_root: Path = C.DATA_ROOT) -> list[str]:
    rec_ids = []
    for region in ("Cx", "Hip"):
        region_dir = data_root / region
        if not region_dir.exists():
            continue
        for p in region_dir.iterdir():
            if p.is_dir() and (p / "suite2p" / "plane0" / "stat.npy").exists():
                rec_ids.append(p.name)
    return sorted(rec_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", type=str, default=None,
                    help="Single recording ID (e.g. 2024-07-01_00018). "
                         "If omitted, predicts every recording under DATA_ROOT.")
    ap.add_argument("--ckpt", type=str, default=None,
                    help="Path to checkpoint. Defaults to CHECKPOINT_DIR/best.pt")
    args = ap.parse_args()

    model, device = load_model(args.ckpt)
    print(f"device: {device}")

    if args.rec:
        rec_ids = [args.rec]
    else:
        rec_ids = list_all_recordings()
        print(f"Found {len(rec_ids)} recordings.")

    for rid in rec_ids:
        try:
            predict_recording(rid, model, device)
        except Exception as ex:
            print(f"[skip] {rid}: {ex}")


if __name__ == "__main__":
    main()