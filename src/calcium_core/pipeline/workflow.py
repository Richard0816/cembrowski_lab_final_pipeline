"""Master pipeline orchestrator.

Runs the full calcium-imaging analysis chain on a single recording folder:

    analyze_output -> spatial_heatmap -> image_all -> hierarchical_clustering
                                                  -> cross-correlation

Each stage is guarded by a ``need_to_run_*`` gate so reruns can skip work that
has already produced its expected outputs.

Email / SMTP handling is *not* imported here.  If you want completion /
failure alerts, import :mod:`calcium_core.pipeline.email_alerts` separately
from the CLI entry point in ``scripts/``.
"""
from __future__ import annotations

import glob
import os
import re
from pathlib import Path
from typing import Tuple, Union

from calcium_core.utils import system as _utils_system  # run_on_folders, get_fps_from_notes, ...
from calcium_core.utils.logging import run_with_logging

from calcium_core.clustering import hierarchical as hierarchical_clustering
from calcium_core.clustering import crosscorr as crosscorrelation
from calcium_core.pipeline import analyze as analyze_output
from calcium_core.spatial import heatmap as spatial_heatmap
from calcium_core.viz import summary as image_all


# --- gate helpers -----------------------------------------------------------
def need_to_run_analysis_py(
    folder_name: str, override: bool = False
) -> Union[Tuple[bool, str], Tuple[bool, None]]:
    """Decide whether ``analyze_output.run_analysis_on_folder`` must run.

    :param folder_name: Current recording folder.
    :param override: Force-run regardless of on-disk state.
    :return: ``(need_to_run, detected_prefix_or_None)``.
    """
    if override:
        return True, None

    working_directory = Path(folder_name + r"\suite2p\plane0")

    required_files_suffix = [  # r0p7 is an arbitrary choice and subject to change
        "_filtered_dff.memmap.float32",
        "_filtered_dff_dt.memmap.float32",
        "_filtered_dff_lowpass.memmap.float32",
    ]

    files = {p.name for p in working_directory.iterdir() if p.is_file()}

    for file_name in files:
        for suffix in required_files_suffix:
            if file_name.endswith(suffix):
                prefix = file_name.removesuffix(suffix)
                if all(f"{prefix}{suf}" in files for suf in required_files_suffix):
                    return False, prefix

    return True, None


def need_to_run_spatial_heatmap(folder_name: str) -> bool:
    return True


def need_to_run_image_all_py(folder_name: str) -> bool:
    return True


def need_to_run_hierarchial_cluster(folder_name: str, override: bool = False) -> bool:
    if override:
        return True

    exists = any(
        os.path.isdir(p)
        for p in glob.glob(os.path.join(folder_name, "*_filtered_cluster_results"))
    )
    return not exists


def need_to_run_crosscorrelation(folder_name: str) -> bool:
    return True


def folder_in_all_logs(folder_name: str, log_files: list[str]) -> bool:
    """Return ``True`` iff ``folder_name`` appears in every given log file.

    A missing log file is treated as "not completed" so the pipeline will rerun.
    """
    for log in log_files:
        if not os.path.exists(log):
            return False
        try:
            with open(log, "r", encoding="utf-8", errors="ignore") as f:
                if folder_name not in f.read():
                    return False
        except Exception:
            return False
    return True


def count_cluster_roi_files(base_dir: Path) -> int:
    """Count how many ``*_rois.npy`` cluster files exist in ``base_dir``."""
    if not base_dir.exists():
        return 0
    return len(list(base_dir.glob("*_rois.npy")))


# --- main orchestration -----------------------------------------------------
def main(folder_name: str):
    """Run the full pipeline on a single recording folder.

    Stages are skipped either because all pipeline log files already mention
    ``folder_name`` or because the stage's individual ``need_to_run_*`` gate
    reports the outputs already exist.
    """
    ALL_LOGS = [
        "fluorescence_analysis.log",
        "raster_and_heatmaps_plots.log",
        "image_all.log",
        "hierarchical_clustering.log",
        "crosscorrelation.log",
    ]

    if folder_in_all_logs(folder_name, ALL_LOGS):
        print(f"[SKIP] {folder_name} already present in all logs.")
        return None

    # --- fluorescence analysis ---
    need_to_run_analysis_py_truth, prefix = need_to_run_analysis_py(folder_name, override=True)
    if need_to_run_analysis_py_truth:
        run_with_logging(
            "fluorescence_analysis.log",
            analyze_output.run_analysis_on_folder,
            folder_name,
        )

    # --- spatial heatmap / raster ---
    if need_to_run_spatial_heatmap(folder_name):
        run_with_logging(
            "raster_and_heatmaps_plots.log",
            spatial_heatmap.run_spatial_heatmap,
            folder_name,
            score_threshold=0.15,  # classify as cell if P >= 0.5
        )
        run_with_logging(
            "raster_and_heatmaps_plots.log",
            spatial_heatmap.coactivation_order_heatmaps,
            folder_name,
            score_threshold=0.15,
        )

    # --- whole-FOV imaging summary ---
    if need_to_run_image_all_py(folder_name):
        run_with_logging(
            "image_all.log",
            image_all.run_full_imaging_on_folder,
            folder_name,
        )

    # --- hierarchical clustering + cross-correlation ---
    if need_to_run_hierarchial_cluster(folder_name, override=True):
        params = dict(
            root=Path(folder_name + r"\suite2p\plane0"),
            fps=_utils_system.get_fps_from_notes(folder_name),
            prefix="r0p7_filtered_",
            method="ward",
            metric="euclidean",
        )
        run_with_logging(
            "hierarchical_clustering.log",
            hierarchical_clustering.main,
            **params,
        )

        params = dict(
            root=Path(folder_name + r"\suite2p\plane0"),
            fps=_utils_system.get_fps_from_notes(folder_name),
            prefix="r0p7_filtered_",
            cluster_folder="",
            max_lag_seconds=5.0,
            cpu_fallback=True,
            zero_lag=True,
            zero_lag_only=False,
        )

        cluster_dir = Path(folder_name) / "suite2p" / "plane0" / "r0p7_filtered_cluster_results"
        n_clusters = count_cluster_roi_files(cluster_dir)

        if n_clusters < 2:
            run_with_logging(
                "crosscorrelation.log",
                print,
                f"[SKIP] cross-correlation for {folder_name}: \n "
                f"found {n_clusters} '*_rois.npy' files in {cluster_dir} (need >= 2).",
            )
        else:
            params = dict(
                root=Path(folder_name + r"\suite2p\plane0"),
                prefix="r0p7_filtered_",
                fps=_utils_system.get_fps_from_notes(folder_name),
                cluster_folder="",
                max_lag_seconds=5.0,
                cpu_fallback=True,
                zero_lag=True,
                zero_lag_only=False,
            )
            run_with_logging(
                "crosscorrelation.log",
                crosscorrelation.run_cluster_cross_correlations_gpu,
                **params,
            )

            params = dict(
                root=Path(folder_name + r"\suite2p\plane0"),
                prefix="r0p7_filtered_",
                fps=_utils_system.get_fps_from_notes(folder_name),
                n_surrogates=5000,
                min_shift_s=1,
                max_shift_s=500,
                shift_cluster="B",
                two_sided=False,
                seed=0,
                use_gpu=True,
                fdr_alpha=0.05,
                save_pairwise_csv=True,
            )
            run_with_logging(
                "crosscorrelation.log",
                crosscorrelation.run_clusterpair_zero_lag_shift_surrogate_stats,
                **params,
            )

    return None


# --- CLI helpers ------------------------------------------------------------
def entries_to_run_to_list(entries: list) -> list:
    """Convert a list of ``"YYYY-MM-DD-NNNNN"`` strings into lists of ints."""
    final = []
    for e in entries:
        final.append(string_to_int_list(e))
    return final


def string_to_int_list(string: str) -> list:
    """Split on ``-`` / ``_`` and coerce each piece to ``int``."""
    temp = []
    delimiters_pattern = r"-|_"
    lst = re.split(delimiters_pattern, string)
    for i in lst:
        temp.append(int(i))
    return temp
