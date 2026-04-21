# calcium_core — architecture

A library-first redesign of the Suite2p calcium imaging pipeline. The package is
consumed by three kinds of caller:

1. **An application** (desktop / web GUI) — the primary driver of this redesign.
2. **Thin CLI scripts** under `scripts/`.
3. **Notebooks / figure scripts** under `figures/`.

The library exposes pure compute functions that return typed data structures.
Rendering is isolated in `viz/` so the app can replace it with its own UI
(plotly, Qt, web canvas, etc.) without touching analysis code.

## Top-level layout

```
calcium_core/
├── ARCHITECTURE.md      # this file
├── src/calcium_core/    # the installable package
├── app/                 # placeholder — future GUI code
├── scripts/             # thin CLI entry points
├── figures/             # publication figure scripts (Fig1–7, stats)
└── tests/
```

The package follows `src/` layout so the app imports `calcium_core` rather
than reaching into relative paths.

## Package modules

| Module        | Role                                                      |
| ------------- | --------------------------------------------------------- |
| `core/`       | Domain model: `Recording`, `Paths`, config dataclasses    |
| `io/`         | All file I/O — suite2p outputs, metadata, cache           |
| `signal/`     | Pure signal processing (filters, dF/F, z-score, spikes)   |
| `detection/`  | Event detection (density, boundaries, per-ROI onsets)    |
| `clustering/` | Hierarchical clustering + pairwise cross-correlation      |
| `spatial/`    | Spatial activation maps, propagation vectors              |
| `reduction/`  | PCA / UMAP                                                |
| `ml/`         | CellFilter deep model (cell vs non-cell classifier)       |
| `pipeline/`   | Orchestration — step runner + progress protocol for UIs   |
| `viz/`        | Matplotlib rendering (kept separate from compute)         |
| `utils/`      | Logging, system helpers (RAM, batch sizing)               |

## Design rules

- **Compute returns data, not plots.** Functions return numpy arrays or
  dataclasses. Plotting lives only under `viz/`.
- **No module-level heavy imports.** `cupy`, `torch`, `seaborn`, `matplotlib`
  are imported inside the functions that use them so the app can pull in the
  library without the full scientific stack.
- **Stable public API at each subpackage `__init__.py`.** Callers (the app
  especially) import from the package, never from internal files.
- **Progress is a protocol.** Long-running steps accept a `ProgressReporter`
  (see `pipeline/progress.py`) so GUI progress bars and CLI logs share one
  contract.
- **Paths are data.** `core/paths.py` owns the Suite2p folder layout; no file
  paths are hard-coded elsewhere.

## Migration map (old → new)

Files in the existing repo map to the new layout as follows. The redesign
intentionally splits plotting out of the large compute files.

| Existing file                              | New home                                                   |
| ------------------------------------------ | ---------------------------------------------------------- |
| `main.py`                                  | `core/suite2p_runner.py` (compute) + `scripts/run_suite2p.py` |
| `Full_work_flow.py`                        | `pipeline/runner.py` + `pipeline/steps.py`                 |
| `analyze_output.py`                        | `pipeline/steps.py::analyze` (uses `signal/`, `detection/`) |
| `event_detection.py`                       | `detection/density.py`, `detection/per_roi.py`             |
| `event_boundaries.py` (compute)            | `detection/boundaries.py`                                  |
| `event_boundaries.py` (plot helpers)       | `viz/events.py`                                            |
| `hierarchical_clustering.py` (compute)     | `clustering/hierarchical.py`                               |
| `hierarchical_clustering.py` (dendrogram)  | `viz/clustering.py`                                        |
| `crosscorrelation.py` (compute, GPU)       | `clustering/crosscorr.py`                                  |
| `crosscorrelation.py` (plots)              | `viz/crosscorr.py`                                         |
| `spatial_heatmap.py` / `_updated.py`       | `spatial/heatmap.py` (one consolidated module)             |
| `image_all.py`                             | `viz/summary.py`                                           |
| `fft_all_rois.py`                          | `signal/spectral.py` + `viz/spectral.py`                   |
| `vectors.py`                               | `spatial/vectors.py` (+ `viz/vectors.py`)                  |
| `pca.py`                                   | `reduction/pca.py` (+ `viz/reduction.py`)                  |
| `umap_embedding.py`                        | `reduction/umap.py`                                        |
| `cellfilter/` (subpackage)                 | `ml/` (drop-in move)                                       |
| `roi_curation_app.py`                      | `app/curation/` (app owns the GUI)                         |
| `generate_cell_detection_parameters.py`    | `ml/logistic_scorer.py`                                    |
| `shifting.py`                              | `scripts/preprocess_shift.py` (one-off)                    |
| `testing.py`                               | `tests/`                                                   |
| `Fig1.py` … `Fig7_1.py`, `stats_figure.py` | `figures/` (as-is; they import from `calcium_core`)        |
| `utils.py` — `Tee`, logging                | `utils/logging.py`                                         |
| `utils.py` — RAM, batch sizing             | `utils/system.py`                                          |
| `utils.py` — suite2p load / memmaps        | `io/suite2p.py`                                            |
| `utils.py` — AAV/FPS/zoom metadata         | `io/metadata.py`                                           |
| `utils.py` — signal helpers                | `signal/filters.py`, `signal/normalize.py`, `signal/spikes.py` |
| `utils.py` — `roi_metric`, `paint_spatial` | `spatial/metrics.py`                                       |
| `utils.py` — `detect_event_windows`        | `detection/density.py`                                     |
| `utils.py` — event-plot helpers            | `viz/events.py`                                            |

## App-library contract

The app should depend only on the package's public API, for example:

```python
from calcium_core.core import Recording
from calcium_core.pipeline import run_pipeline, Steps, ProgressReporter
from calcium_core.detection import detect_events
from calcium_core.clustering import hierarchical_cluster, cross_correlate
```

The app is free to render results with whatever UI framework it chooses;
`calcium_core.viz` is optional and exists for notebooks and figure scripts.

## Current state of the migration

Migration **complete** — every legacy module has a real port under
`src/calcium_core/`. The `_legacy.py` sys.path shim has been removed; the
package no longer reaches back into the legacy repo root.

**Fully ported subpackages:**
- `core/` — Recording, paths, models, EventDetectionParams, Suite2pRunConfig
- `core/suite2p_runner.py` — Suite2p detection on raw TIFFs (from `main.py`)
- `io/` — Suite2p loaders, metadata (AAV/FPS/zoom)
- `signal/` — filters, normalize, spikes, spectral (FFT)
- `detection/` — density, boundaries, per-ROI onsets, event tables
- `spatial/` — metrics, heatmap, heatmap_legacy, vectors
- `clustering/` — hierarchical (Ward), crosscorr (GPU/CPU)
- `reduction/` — pca, umap
- `ml/` — CellFilter (config, model, dataset, train, predict) + logistic_scorer
- `viz/` — events, summary (per-recording figures)
- `pipeline/` — step registry, runner, progress protocol, workflow
  orchestrator, email alerts, analyze, full-pipeline entry point
- `utils/` — Tee, run_with_logging, batch sizing, folder walker

**Lazy loading policy.** `calcium_core` itself pulls only `numpy`, `scipy`,
`pandas`, `psutil`, `openpyxl`. Heavy deps (matplotlib, seaborn, torch, cupy,
suite2p) are imported inside the submodules that need them, and the
`clustering/` and `spatial/` subpackages route symbol lookups through
`__getattr__` so top-level access does not trigger those imports.

**End-to-end entry points:**
- `scripts/run_pipeline.py` — run a recording that already has Suite2p outputs.
- `scripts/run_new_recording.py` — raw TIFF → Suite2p → full analysis chain.

Step registry (see `pipeline/steps.py`):
```
suite2p  -> analyze  -> heatmap  -> image_all  -> cluster  -> correlate  (+ fft)
```
