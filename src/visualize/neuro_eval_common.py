#!/usr/bin/env python3
"""
Shared 90_810 tensor loading and Nethobench split scoring for visualize scripts.

Loads aligned ``[n_seq, n_time, n_reg]`` arrays from a data directory, runs
``compute_neuro_scores`` per sequence split, and exposes the same tables/cache
behavior as the former notebook shim (without depending on nethobench/notebooks).
"""

from __future__ import annotations

import json
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cns_plotting import setup_cnsplots_style

SUB_DATA_DIR = "90_810"
N_SPLITS = 4
EPS = 1e-8

MODEL_FILES = OrderedDict(
    [
        ("VAR", "long_predictions_90_810_VAR_BASELINE.npy"),
        ("SSM", "long_predictions_90_810_cDMM_SSM.npy"),
        ("RNN", "long_predictions_90_810_GRU_AR.npy"),
        ("1_step", "long_predictions_90_810_1_step.npy"),
        ("AR", "long_predictions_90_810_AR_KV.npy"),
        ("TF", "long_predictions_90_810_TF.npy"),
        ("TF_QL_0.08_KL_0.02", "long_predictions_90_810_TF_QTL_0.08_KL_0.02.npy"),
        ("sequifier", "long_predictions_sequifier_last100.npy"),
    ]
)

GT_FILES = {
    "bench": "long_ground_truth_90_810.npy",
    "sequifier": "long_ground_truth_sequifier_last100.npy",
}

MODEL_TO_GT = {
    "VAR": "bench",
    "SSM": "bench",
    "RNN": "bench",
    "1_step": "bench",
    "AR": "bench",
    "TF": "bench",
    "TF_QL_0.08_KL_0.02": "bench",
    "sequifier": "sequifier",
}

MODEL_ORDER = list(MODEL_FILES)
MODEL_LABELS = {
    "VAR": "VAR",
    "SSM": "SSM",
    "RNN": "RNN",
    "1_step": "1_step",
    "AR": "AR",
    "TF": "TF",
    "TF_QL_0.08_KL_0.02": "TF_QL_KL",
    "sequifier": "Sequifier",
}
MODEL_COLORS = {
    "VAR": "#4DBBD5",
    "SSM": "#3C5488",
    "RNN": "#E64B35",
    "1_step": "#7A7A7A",
    "AR": "#00A087",
    "TF": "#EFC000",
    "TF_QL_0.08_KL_0.02": "#B24775",
    "sequifier": "#7E57C2",
}

# Matches ``neuro_metric_specific_visualizations_90_810`` (includes K12 occupancy).
SELECTED_METRICS = OrderedDict(
    [
        ("distribution", ["KL_or_JSD_score01", "QNT_score01", "MOM_score01", "Mean_score01"]),
        ("temporal_spectral", ["TRJDIST_score01"]),
        (
            "relational",
            [
                "GRAPH_score01",
                "CrossRegionMI_score01",
                "LaggedCovariance_score01",
                "ImpulseResponse_score01",
            ],
        ),
        ("geometry", ["MANI_score01", "SubspaceAngle_score01"]),
        (
            "state_dynamics",
            [
                "LatentStateOccupancyK11_score01",
                "LatentStateOccupancyK12_score01",
                "LatentStateTransitionLag1K11_score01",
                "LatentStateTransitionLag2K11_score01",
                "LatentStateTransitionLag3K11_score01",
            ],
        ),
    ]
)


@dataclass(frozen=True)
class ModelArrays:
    gt: np.ndarray
    pred: np.ndarray
    gt_key: str


def setup_plot_style() -> None:
    setup_cnsplots_style(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "svg.fonttype": "none",
            "axes.linewidth": 0.8,
            "font.size": 9,
        }
    )


def load_required_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    arr = np.load(path, allow_pickle=False)
    if arr.ndim != 3:
        raise ValueError(f"Expected [n_seq, n_time, n_reg] array, got {arr.shape} from {path}")
    return arr.astype(np.float64, copy=False)


def align_gt_pred(gt: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_seq = min(gt.shape[0], pred.shape[0])
    n_time = min(gt.shape[1], pred.shape[1])
    n_reg = min(gt.shape[2], pred.shape[2])
    return gt[:n_seq, :n_time, :n_reg], pred[:n_seq, :n_time, :n_reg]


def load_model_arrays(data_dir: Path) -> dict[str, ModelArrays]:
    gt_arrays = {key: load_required_array(data_dir / name) for key, name in GT_FILES.items()}
    model_arrays: dict[str, ModelArrays] = {}
    for model_name, file_name in MODEL_FILES.items():
        gt_key = MODEL_TO_GT[model_name]
        pred = load_required_array(data_dir / file_name)
        gt, pred = align_gt_pred(gt_arrays[gt_key], pred)
        model_arrays[model_name] = ModelArrays(gt=gt, pred=pred, gt_key=gt_key)
    return model_arrays


def write_neurobench_csv_from_arrays(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    region_names: list[str],
    out_gt_csv: Path,
    out_pred_csv: Path,
) -> None:
    if gt_arr.shape != pred_arr.shape:
        raise ValueError(f"GT/pred shape mismatch: {gt_arr.shape} vs {pred_arr.shape}")

    n_seq, n_time, n_reg = gt_arr.shape
    seq_ids = np.repeat(np.arange(n_seq), n_time)
    item_pos = np.tile(np.arange(n_time), n_seq)

    gt_df = pd.DataFrame(gt_arr.reshape(-1, n_reg), columns=region_names)
    gt_df.insert(0, "itemPosition", item_pos)
    gt_df.insert(0, "sequenceId", seq_ids)

    pred_df = pd.DataFrame(pred_arr.reshape(-1, n_reg), columns=region_names)
    pred_df.index = seq_ids

    out_gt_csv.parent.mkdir(parents=True, exist_ok=True)
    out_pred_csv.parent.mkdir(parents=True, exist_ok=True)
    gt_df.to_csv(out_gt_csv, index=False)
    pred_df.to_csv(out_pred_csv)


_COMPUTE_NEURO_SCORES_FN = None


def load_compute_neuro_scores():
    """Use the installed ``compute_neuro_scores`` from the nethobench package."""
    global _COMPUTE_NEURO_SCORES_FN
    if _COMPUTE_NEURO_SCORES_FN is None:
        from nethobench.neuro.pipeline import compute_neuro_scores as fn

        _COMPUTE_NEURO_SCORES_FN = fn
    return _COMPUTE_NEURO_SCORES_FN


def scores_for_arrays(gt_arr: np.ndarray, pred_arr: np.ndarray, region_names: list[str]) -> dict[str, float]:
    compute_neuro_scores = load_compute_neuro_scores()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        gt_csv = tmp_path / "gt.csv"
        pred_csv = tmp_path / "pred.csv"
        write_neurobench_csv_from_arrays(gt_arr, pred_arr, region_names, gt_csv, pred_csv)
        scores = compute_neuro_scores(pred_csv, gt_csv)
    return {key: float(value) if value is not None else float("nan") for key, value in scores.items()}


def sequence_split_indices(n_seq: int, n_splits: int) -> list[np.ndarray]:
    if n_splits < 1:
        raise ValueError("n_splits must be positive.")
    if n_seq < n_splits:
        raise ValueError(f"Need at least {n_splits} sequences; got {n_seq}.")
    splits = list(np.array_split(np.arange(n_seq), n_splits))
    if any(split.size == 0 for split in splits):
        raise ValueError(f"Empty split produced for n_seq={n_seq}, n_splits={n_splits}.")
    return splits


def nansem(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan")
    return float(values.std(ddof=1) / np.sqrt(values.size))


def aggregate_split_scores(split_scores: list[dict[str, float]]) -> tuple[dict[str, float], dict[str, float]]:
    keys = sorted({key for scores in split_scores for key in scores})
    means: dict[str, float] = {}
    sems: dict[str, float] = {}
    for key in keys:
        values = np.asarray([scores.get(key, np.nan) for scores in split_scores], dtype=float)
        means[key] = float(np.nanmean(values))
        sems[key] = nansem(values)
    return means, sems


def compute_or_load_scores(
    model_arrays: dict[str, ModelArrays],
    output_dir: Path,
    region_names: list[str],
    n_splits: int,
    force_scores: bool,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], dict[str, list[dict[str, float]]]]:
    cache_path = output_dir / f"scores_cache_{SUB_DATA_DIR}_{n_splits}split_submetric_viz.json"
    signature = {
        "kind": "submetric_visualization_scores",
        "sub_data_dir": SUB_DATA_DIR,
        "n_splits": int(n_splits),
        "models": MODEL_ORDER,
        "selected_metrics": {family: list(metrics) for family, metrics in SELECTED_METRICS.items()},
    }
    if cache_path.exists() and not force_scores:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("_cache_signature") == signature:
            means = {
                model: {key: float(value) for key, value in values.items()}
                for model, values in payload["means"].items()
            }
            sems = {
                model: {key: float(value) for key, value in values.items()}
                for model, values in payload["sems"].items()
            }
            return means, sems, payload["per_split"]
        print("Score cache signature mismatch; recomputing.")

    means: dict[str, dict[str, float]] = {}
    sems: dict[str, dict[str, float]] = {}
    per_split: dict[str, list[dict[str, float]]] = {}
    for model_name in MODEL_ORDER:
        arrays = model_arrays[model_name]
        print(f"Computing Nethobench scores for {model_name} over {n_splits} sequence splits")
        split_scores: list[dict[str, float]] = []
        for split_idx, indices in enumerate(sequence_split_indices(arrays.gt.shape[0], n_splits), start=1):
            print(f"  split {split_idx}/{n_splits}: sequence {int(indices[0])}..{int(indices[-1])}")
            split_scores.append(scores_for_arrays(arrays.gt[indices], arrays.pred[indices], region_names))
        means[model_name], sems[model_name] = aggregate_split_scores(split_scores)
        per_split[model_name] = split_scores

    cache_path.write_text(
        json.dumps(
            {
                "_cache_signature": signature,
                "means": means,
                "sems": sems,
                "per_split": per_split,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved score cache to {cache_path}")
    return means, sems, per_split


def selected_metric_keys() -> list[str]:
    return [metric for metrics in SELECTED_METRICS.values() for metric in metrics]


def score_tables(
    means: dict[str, dict[str, float]],
    sems: dict[str, dict[str, float]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = selected_metric_keys()
    metric_df = pd.DataFrame(
        {model: {metric: means[model].get(metric, np.nan) for metric in metrics} for model in MODEL_ORDER}
    )
    metric_sem_df = pd.DataFrame(
        {model: {metric: sems[model].get(metric, np.nan) for metric in metrics} for model in MODEL_ORDER}
    )
    family_cols = [f"family_{family}" for family in SELECTED_METRICS]
    family_cols += ["FINAL_COMPOSITE_SCORE"]
    family_df = pd.DataFrame(
        {model: {metric: means[model].get(metric, np.nan) for metric in family_cols} for model in MODEL_ORDER}
    )
    family_sem_df = pd.DataFrame(
        {model: {metric: sems[model].get(metric, np.nan) for metric in family_cols} for model in MODEL_ORDER}
    )
    return metric_df, metric_sem_df, family_df, family_sem_df
