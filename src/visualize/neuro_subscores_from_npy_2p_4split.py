#!/usr/bin/env python3
"""
Neuro subscores from NumPy — **2p notebook variant with 4-way sequence split**.

Mirrors ``neuro_subscores_from_npy_2p.ipynb`` with the same change as
``neuro_subscores_from_npy_with_sequifier_4split.py``: aligned ground truth and
predictions are split into four contiguous sequence chunks; ``compute_neuro_scores``
runs on each chunk; summaries and plots use the **mean** across chunks with **SEM**
(ddof=1, n=4) as error bars.

Outputs go to ``outputs/neuro_subscores_from_npy_2p_merged_4split/`` (the original
notebook output directory is untouched).

The upstream notebook has no polar family-radar cell; this script includes the
bar chart, metric dot plot, horizon sweep, and a compact example overlay aligned
with ``sub_data_dir`` and ``model_files`` (the notebook's example cell mixes
another subfolder/model; here it matches the scoring tensors).
"""

from __future__ import annotations

import json
import tempfile
from collections import OrderedDict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd

import nethobench
from nethobench import compute_neuro_scores
from nethobench.analysis.score_definitions import NEURO_FAMILY_METRICS

# ---------------------------------------------------------------------------
# Paths and toggles (aligned with neuro_subscores_from_npy_2p.ipynb cell 1)
# ---------------------------------------------------------------------------
package_dir = Path(nethobench.__file__).resolve().parent
data_dir = package_dir / "data"
sub_data_dir = "90_240"
outputs_dir = package_dir / "outputs" / "neuro_subscores_from_npy_2p_merged_4split"
csv_dir = outputs_dir / "organized_csv"
CONTEXT_STEPS_TO_DROP = 0

N_SPLITS = 4

cache_path = outputs_dir / f"scores_cache_{sub_data_dir}_4split.json"
outputs_dir.mkdir(parents=True, exist_ok=True)
csv_dir.mkdir(parents=True, exist_ok=True)

USE_CACHE = True
RUN_SCORES = True
SAVE_ORGANIZED_CSV = True
RUN_EXAMPLE_OVERLAY_PLOT = True
RUN_FAMILY_BAR = True
RUN_METRIC_DOTPLOT = True
RUN_HORIZON_SCORES = True

# ---------------------------------------------------------------------------
# Ground truth and models (notebook cell 2)
# ---------------------------------------------------------------------------
gt_path_bench = data_dir / sub_data_dir / "long_ground_truth_POCO_LITE.npy"

model_files = {
    "POCO_1_step": data_dir / sub_data_dir / "long_predictions_POCO_LITE_1_step.npy",
}

MODEL_TO_GT = {
    "POCO_1_step": "bench",
}


def _load_gt(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D GT [n_seq, n_time, n_reg], got {arr.shape} from {path}")
    return arr


missing = [str(p) for p in [*model_files.values(), gt_path_bench] if not p.exists()]
if missing:
    raise FileNotFoundError("Missing required .npy files:\n" + "\n".join(missing))

gt_full_bench = _load_gt(gt_path_bench)

if CONTEXT_STEPS_TO_DROP < 0:
    raise ValueError("CONTEXT_STEPS_TO_DROP must be >= 0")

for label, arr in [("bench", gt_full_bench)]:
    if CONTEXT_STEPS_TO_DROP >= arr.shape[1]:
        raise ValueError(
            f"{label}: CONTEXT_STEPS_TO_DROP={CONTEXT_STEPS_TO_DROP} >= n_time={arr.shape[1]}"
        )

gt_bench = gt_full_bench[:, CONTEXT_STEPS_TO_DROP:, :]
gt = gt_bench


def _gt_for_model(model_name: str) -> np.ndarray:
    return gt_bench


n_seq_b, n_time_b, n_reg_b = gt_bench.shape
region_names = [f"R{i}" for i in range(n_reg_b)]

# ---------------------------------------------------------------------------
# Dimensionless MAE (notebook cell 2)
# ---------------------------------------------------------------------------
MAE_PRED_START = 90
_eps_mae_scale = 1e-8
MAE_SCALE_MODE = "std"


def _aligned_gt_pred_for_model(model_name: str, pred_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gt_arr = _gt_for_model(model_name)

    if pred_arr.ndim != 3:
        raise ValueError(f"{model_name}: expected 3D prediction array, got {pred_arr.shape}")

    n_seq = min(gt_arr.shape[0], pred_arr.shape[0])
    n_t = min(gt_arr.shape[1], pred_arr.shape[1])
    n_reg = min(gt_arr.shape[2], pred_arr.shape[2])

    gt_al = gt_arr[:n_seq, :n_t, :n_reg].astype(np.float64, copy=False)
    pred_al = pred_arr[:n_seq, :n_t, :n_reg].astype(np.float64, copy=False)
    return gt_al, pred_al


def _scale_from_gt_forecast(gt_f: np.ndarray, mode: str = MAE_SCALE_MODE, eps: float = _eps_mae_scale) -> float:
    vals = gt_f[np.isfinite(gt_f)]
    if vals.size == 0:
        return 1.0

    if mode == "std":
        s = float(np.std(vals))
    elif mode == "iqr":
        q75, q25 = np.percentile(vals, [75, 25])
        s = float(q75 - q25)
    elif mode == "mad":
        med = float(np.median(vals))
        s = float(np.median(np.abs(vals - med)))
    else:
        raise ValueError(f"Unknown MAE_SCALE_MODE: {mode}")

    return max(s, eps)


def _dimensionless_mae_from_aligned(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    pred_start: int = MAE_PRED_START,
    scale_mode: str = MAE_SCALE_MODE,
    eps: float = _eps_mae_scale,
) -> float:
    T = gt_arr.shape[1]
    pred_start = int(max(0, min(pred_start, T - 1)))

    gt_f = gt_arr[:, pred_start:, :]
    pred_f = pred_arr[:, pred_start:, :]

    err = np.abs(pred_f - gt_f)
    valid = np.isfinite(err)

    mae = np.sum(err[valid]) / max(np.sum(valid), 1)
    scale = _scale_from_gt_forecast(gt_f, mode=scale_mode, eps=eps)

    return float(mae / scale)


def _nansem_across_values(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    a = a[np.isfinite(a)]
    if a.size < 2:
        return float("nan")
    return float(a.std(ddof=1) / np.sqrt(a.size))


def _mae_mean_sem_over_sequence_splits(model_name: str, pred_path: Path) -> tuple[float, float]:
    pred_arr = np.load(pred_path, allow_pickle=False)
    gt_arr, pred_arr = _aligned_gt_pred_for_model(model_name, pred_arr)
    n_seq = gt_arr.shape[0]
    if n_seq < N_SPLITS:
        return float("nan"), float("nan")
    splits = np.array_split(np.arange(n_seq), N_SPLITS)
    vals = [
        _dimensionless_mae_from_aligned(gt_arr[idx], pred_arr[idx], pred_start=MAE_PRED_START)
        for idx in splits
        if idx.size > 0
    ]
    a = np.asarray(vals, dtype=float)
    return float(np.nanmean(a)), _nansem_across_values(a)


print("Ground truth (benchmark):")
print("  path:", gt_path_bench)
print("  scored shape:", gt_bench.shape)

pred_shapes: dict[str, tuple[int, ...]] = {}
for model_name, pred_path in model_files.items():
    pred = np.load(pred_path, allow_pickle=False)
    if pred.ndim != 3:
        raise ValueError(f"{model_name}: expected 3D [n_seq, n_time, n_reg], got {pred.shape}")
    pred_shapes[model_name] = pred.shape
    gt_ref = _gt_for_model(model_name)
    if pred.shape[0] != gt_ref.shape[0] or pred.shape[2] != gt_ref.shape[2]:
        raise ValueError(
            f"{model_name}: sequence/region mismatch vs its GT tensor. "
            f"GT={gt_ref.shape} pred={pred.shape}"
        )

print("Prediction shapes:")
for k, v in pred_shapes.items():
    print(f"  {k}: {v}")


# ---------------------------------------------------------------------------
# CSV bridge
# ---------------------------------------------------------------------------
def write_neurobench_csv_from_arrays(
    *, gt_arr, pred_arr, region_names, out_gt_csv, out_pred_csv
):
    gt_arr = np.asarray(gt_arr, dtype=np.float64)
    pred_arr = np.asarray(pred_arr, dtype=np.float64)

    if gt_arr.ndim != 3 or pred_arr.ndim != 3:
        raise ValueError(f"Expected [n_seq,n_time,n_reg] arrays, got {gt_arr.shape} and {pred_arr.shape}")

    if gt_arr.shape[0] != pred_arr.shape[0] or gt_arr.shape[2] != pred_arr.shape[2]:
        raise ValueError(f"GT/pred sequence-region mismatch: {gt_arr.shape} vs {pred_arr.shape}")

    max_len = min(gt_arr.shape[1], pred_arr.shape[1])
    gt_arr = gt_arr[:, :max_len, :]
    pred_arr = pred_arr[:, :max_len, :]

    n_seq_local, n_time_local, n_reg_local = gt_arr.shape
    if n_reg_local != len(region_names):
        raise ValueError(f"region_names length {len(region_names)} != n_reg {n_reg_local}")

    seq_ids = np.repeat(np.arange(n_seq_local), n_time_local)
    item_pos = np.tile(np.arange(n_time_local), n_seq_local)

    gt_df = pd.DataFrame(gt_arr.reshape(-1, n_reg_local), columns=region_names)
    gt_df.insert(0, "itemPosition", item_pos)
    gt_df.insert(0, "sequenceId", seq_ids)

    pred_df = pd.DataFrame(pred_arr.reshape(-1, n_reg_local), columns=region_names)
    pred_df.index = seq_ids

    out_gt_csv.parent.mkdir(parents=True, exist_ok=True)
    out_pred_csv.parent.mkdir(parents=True, exist_ok=True)

    gt_df.to_csv(out_gt_csv, index=False)
    pred_df.to_csv(out_pred_csv)
    return out_gt_csv, out_pred_csv


def _scores_for_arrays(gt_slice: np.ndarray, pred_slice: np.ndarray) -> dict[str, float]:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        out_gt_csv = tmp_path / "gt.csv"
        out_pred_csv = tmp_path / "pred.csv"

        write_neurobench_csv_from_arrays(
            gt_arr=gt_slice,
            pred_arr=pred_slice,
            region_names=region_names,
            out_gt_csv=out_gt_csv,
            out_pred_csv=out_pred_csv,
        )
        scores = compute_neuro_scores(out_pred_csv, out_gt_csv)

    return {k: (float(v) if v is not None else float("nan")) for k, v in scores.items()}


def _sequence_split_indices(n_seq: int, n_splits: int) -> list[np.ndarray]:
    if n_seq < n_splits:
        raise ValueError(
            f"Need at least {n_splits} sequences for an {n_splits}-way split along "
            f"the sequence axis, got n_seq={n_seq}."
        )
    parts = np.array_split(np.arange(n_seq), n_splits)
    if any(p.size == 0 for p in parts):
        raise ValueError(f"array_split produced an empty chunk for n_seq={n_seq}, n_splits={n_splits}")
    return list(parts)


def _aggregate_split_score_dicts(split_dicts: list[dict[str, float]]) -> tuple[dict[str, float], dict[str, float]]:
    keys: set[str] = set()
    for d in split_dicts:
        keys.update(d.keys())
    mean_d: dict[str, float] = {}
    sem_d: dict[str, float] = {}
    for k in keys:
        arr = np.array([float(d.get(k, float("nan"))) for d in split_dicts], dtype=float)
        mean_d[k] = float(np.nanmean(arr))
        sem_d[k] = _nansem_across_values(arr)
    return mean_d, sem_d


# ---------------------------------------------------------------------------
# Main scoring
# ---------------------------------------------------------------------------
family_order = [
    "distribution",
    "fidelity",
    "temporal_spectral",
    "relational",
    "geometry",
]

CACHE_SIGNATURE = {
    "kind": "neuro_scores_4split_sequence_axis_2p",
    "n_splits": N_SPLITS,
    "models": list(model_files.keys()),
    "sub_data_dir": sub_data_dir,
}

all_scores: dict[str, dict[str, float]] = {}
all_scores_sem: dict[str, dict[str, float]] = {}
per_model_split_scores: dict[str, list[dict[str, float]]] = {}

if USE_CACHE and cache_path.exists():
    print("Loading cached 4-split results from", cache_path)
    loaded = json.loads(cache_path.read_text(encoding="utf-8"))
    if isinstance(loaded, dict) and loaded.get("_cache_signature") == CACHE_SIGNATURE:
        all_scores = {k: {kk: float(vv) for kk, vv in v.items()} for k, v in loaded["means"].items()}
        all_scores_sem = {k: {kk: float(vv) for kk, vv in v.items()} for k, v in loaded["sem"].items()}
        per_model_split_scores = loaded["per_split"]
    else:
        print("Cache signature mismatch; recomputing 4-split scores.")

if RUN_SCORES and (not all_scores or not USE_CACHE):
    for model_name, pred_path in model_files.items():
        print(f"\nComputing {N_SPLITS}-split neuro scores for: {model_name}")
        pred = np.load(pred_path, allow_pickle=False)
        gt_model = _gt_for_model(model_name)
        gt_al, pred_al = _aligned_gt_pred_for_model(model_name, pred)

        organized_model_dir = csv_dir / model_name
        out_gt_csv = organized_model_dir / "gt.csv"
        out_pred_csv = organized_model_dir / "pred.csv"

        if SAVE_ORGANIZED_CSV:
            write_neurobench_csv_from_arrays(
                gt_arr=gt_model,
                pred_arr=pred,
                region_names=region_names,
                out_gt_csv=out_gt_csv,
                out_pred_csv=out_pred_csv,
            )
            print("  wrote CSVs under", organized_model_dir)

        split_indices = _sequence_split_indices(gt_al.shape[0], N_SPLITS)
        split_dicts: list[dict[str, float]] = []
        for si, idx in enumerate(split_indices):
            print(f"  split {si + 1}/{N_SPLITS}: sequences {int(idx[0])}..{int(idx[-1])} (n={idx.size})")
            split_dicts.append(_scores_for_arrays(gt_al[idx], pred_al[idx]))

        mean_d, sem_d = _aggregate_split_score_dicts(split_dicts)
        all_scores[model_name] = mean_d
        all_scores_sem[model_name] = sem_d
        per_model_split_scores[model_name] = split_dicts

    payload = {
        "_cache_signature": CACHE_SIGNATURE,
        "means": all_scores,
        "sem": all_scores_sem,
        "per_split": per_model_split_scores,
    }
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Saved cache to", cache_path)

# ---------------------------------------------------------------------------
# Family / composite table
# ---------------------------------------------------------------------------
rows = []
rows_sem = []
for model_name, scores in all_scores.items():
    sem = all_scores_sem.get(model_name, {})
    row = {"model": model_name}
    row_sem = {"model": model_name}
    for fam in family_order:
        row[f"family_{fam}"] = scores.get(f"family_{fam}", np.nan)
        row_sem[f"family_{fam}"] = sem.get(f"family_{fam}", np.nan)
    row["FINAL_COMPOSITE_SCORE"] = scores.get("FINAL_COMPOSITE_SCORE", np.nan)
    row_sem["FINAL_COMPOSITE_SCORE"] = sem.get("FINAL_COMPOSITE_SCORE", np.nan)
    rows.append(row)
    rows_sem.append(row_sem)

df = pd.DataFrame(rows).set_index("model")
df_sem = pd.DataFrame(rows_sem).set_index("model")
cols = [f"family_{f}" for f in family_order] + ["FINAL_COMPOSITE_SCORE"]
df = df[cols]
df_sem = df_sem.reindex(df.index)[cols]

print("\nNeuro family subscores — mean over sequence splits (higher is better):")
print(df.to_string())
print("\nSEM across splits (ddof=1, n=4):")
print(df_sem.to_string())


def top_metrics(scores: dict[str, float], k: int = 5):
    ignore_prefixes = ("family_", "FINAL_", "composite_score")
    items = []
    for key, value in scores.items():
        if value is None:
            continue
        if not np.isfinite(value):
            continue
        if key.startswith(ignore_prefixes):
            continue
        items.append((key, float(value)))
    items.sort(key=lambda x: x[1], reverse=True)
    return items[:k]


print("\nTop-scoring individual metrics per model:")
for model_name, scores in all_scores.items():
    items = top_metrics(scores, k=5)
    pretty = ", ".join([f"{n}={v:.3f}" for n, v in items]) if items else "(none)"
    print(f"  {model_name}: {pretty}")

# ---------------------------------------------------------------------------
# Submetrics export (notebook cell 7)
# ---------------------------------------------------------------------------
model_order = [m for m in ["POCO_1_step"] if m in all_scores]
if not model_order:
    model_order = list(all_scores.keys())
if not model_order:
    raise ValueError("No models in all_scores.")

all_metric_keys = sorted({k for m in model_order for k in all_scores[m].keys()})
submetric_keys = [
    k
    for k in all_metric_keys
    if k.endswith("_score01") and not k.startswith("family_") and "COMPOSITE" not in k
]

if not submetric_keys:
    raise ValueError("No submetrics found in all_scores.")

submetrics_wide_df = pd.DataFrame(
    {m: {k: float(all_scores[m].get(k, float("nan"))) for k in submetric_keys} for m in model_order}
)
submetrics_wide_df.index.name = "submetric"

submetrics_sem_wide_df = pd.DataFrame(
    {m: {k: float(all_scores_sem[m].get(k, float("nan"))) for k in submetric_keys} for m in model_order}
)
submetrics_sem_wide_df.index.name = "submetric"

tsv_path = outputs_dir / f"submetrics_comparison_{sub_data_dir}_4split.tsv"
csv_path = outputs_dir / f"submetrics_comparison_{sub_data_dir}_4split.csv"
sem_tsv_path = outputs_dir / f"submetrics_comparison_{sub_data_dir}_4split_sem.tsv"
md_path = outputs_dir / f"submetrics_comparison_{sub_data_dir}_4split.md"

submetrics_wide_df.to_csv(tsv_path, sep="\t", float_format="%.6f")
submetrics_wide_df.to_csv(csv_path, float_format="%.6f")
submetrics_sem_wide_df.to_csv(sem_tsv_path, sep="\t", float_format="%.6f")
md_path.write_text(submetrics_wide_df.to_markdown(floatfmt=".6f") + "\n", encoding="utf-8")
print(f"\nSaved submetrics (mean): {tsv_path}")
print(f"Saved submetrics (SEM): {sem_tsv_path}")
print(f"Saved Markdown: {md_path}")

# ---------------------------------------------------------------------------
# Example overlay — same layout as notebook, paths aligned with scoring
# ---------------------------------------------------------------------------
if RUN_EXAMPLE_OVERLAY_PLOT:
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["axes.linewidth"] = 0.8

    EXAMPLE_SUBDIR = sub_data_dir
    PRED_START = 0
    SAVE_EXAMPLE_SVG = True

    MODELS_TO_PLOT = [m for m in ["POCO_1_step"] if m in model_files]
    _n_ex = int(pred_shapes.get("POCO_1_step", (1,))[0])
    MODEL_TO_EXAMPLE_SEQ = {"POCO_1_step": min(9, max(0, _n_ex - 1))}

    example_gt_paths = {"bench": gt_path_bench}
    MODEL_TO_EXAMPLE_GT = {m: "bench" for m in MODELS_TO_PLOT}
    example_model_paths = {m: model_files[m] for m in MODELS_TO_PLOT}

    run_colors_ex = {"POCO_1_step": "#A23B72"}

    gt_example_cache = {"bench": np.load(example_gt_paths["bench"], allow_pickle=False)}

    def _gt_for_example_model(model_name: str) -> np.ndarray:
        return gt_example_cache[MODEL_TO_EXAMPLE_GT[model_name]]

    def _example_seq_for_model(model_name: str) -> int:
        return int(MODEL_TO_EXAMPLE_SEQ[model_name])

    def _compute_panel_ylim(gt_overlay: np.ndarray, pred_overlay: np.ndarray) -> tuple[float, float]:
        vals = np.concatenate([gt_overlay.ravel(), pred_overlay.ravel()])
        finite_vals = vals[np.isfinite(vals)]
        if finite_vals.size == 0:
            return -1.0, 1.0
        y_min = float(np.min(finite_vals))
        y_max = float(np.max(finite_vals))
        if np.isclose(y_min, y_max):
            pad = 0.1 if y_max == 0 else 0.1 * abs(y_max)
        else:
            pad = 0.06 * (y_max - y_min)
        return y_min - pad, y_max + pad

    pred_ex: dict[str, np.ndarray] = {}
    missing_models: list[str] = []
    for model_name in MODELS_TO_PLOT:
        pth = example_model_paths[model_name]
        if not pth.exists():
            print(f"Warning: missing prediction file for {model_name}: {pth}")
            missing_models.append(model_name)
            continue
        pred_ex[model_name] = np.load(pth, allow_pickle=False)

    models_order = [m for m in MODELS_TO_PLOT if m in pred_ex]
    if models_order:
        fig_h = max(0.95 * len(models_order), 3.6)
        fig_ex, axes = plt.subplots(
            len(models_order),
            1,
            figsize=(3, fig_h),
            sharex=True,
            sharey=False,
            gridspec_kw={"hspace": 0.06},
        )
        if len(models_order) == 1:
            axes = [axes]

        for i, (ax, model_name) in enumerate(zip(axes, models_order)):
            pred_arr = pred_ex[model_name]
            gt_ex = _gt_for_example_model(model_name)
            seq_idx = _example_seq_for_model(model_name)
            T = min(gt_ex.shape[1], pred_arr.shape[1])
            n_reg = min(gt_ex.shape[2], pred_arr.shape[2])
            this_pred_start = T // 2 if PRED_START is None else int(PRED_START)
            this_pred_start = max(0, min(this_pred_start, T - 1))
            gt_seq = gt_ex[seq_idx, this_pred_start:T, :n_reg]
            pred_seq = pred_arr[seq_idx, this_pred_start:T, :n_reg]
            time_steps = np.arange(T - this_pred_start)
            y_min, y_max = _compute_panel_ylim(gt_seq, pred_seq)
            ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
            ax.set_axisbelow(True)
            for r in range(n_reg):
                ax.plot(time_steps, gt_seq[:, r], color="#9A9A9A", linewidth=0.85, alpha=0.24, zorder=1)
            for r in range(n_reg):
                ax.plot(
                    time_steps,
                    pred_seq[:, r],
                    color=run_colors_ex.get(model_name, "#1f77b4"),
                    linewidth=1.0,
                    alpha=0.42,
                    zorder=3,
                )
            ax.set_ylim(y_min, y_max)
            ax.margins(x=0.0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_yticks([])
            ax.set_ylabel("")
            ax.tick_params(axis="y", left=False, labelleft=False)
            if i < len(models_order) - 1:
                ax.tick_params(axis="x", bottom=False, labelbottom=False)
            else:
                ax.tick_params(axis="x", labelsize=8.5, length=3)
            ax.text(
                0.0,
                0.92,
                model_name,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9.2,
                fontweight="medium",
                color=run_colors_ex.get(model_name, "#1f77b4"),
            )

        plt.subplots_adjust(left=0.06, right=0.995, top=0.995, bottom=0.09, hspace=0.05)
        if SAVE_EXAMPLE_SVG:
            models_tag = "_".join([m.lower().replace(" ", "_") for m in models_order])
            out_ex = outputs_dir / f"prediction_example_overlay_compact_{EXAMPLE_SUBDIR}_4split_{models_tag}.svg"
            fig_ex.savefig(out_ex, format="svg", bbox_inches="tight")
            print(f"Saved example plot to: {out_ex}")
        plt.close(fig_ex)

# ---------------------------------------------------------------------------
# Grouped bar chart (notebook cell 9) — heights = split means, yerr = split SEM
# ---------------------------------------------------------------------------
if RUN_FAMILY_BAR:
    mpl.rcParams["svg.fonttype"] = "none"
    normalize = False

    models_to_plot_bar = list(model_files.keys())
    plot_cols_requested = [
        "family_distribution",
        "family_temporal_spectral",
        "family_relational",
        "family_geometry",
        "FINAL_COMPOSITE_SCORE",
    ]

    df_plot = df.copy()
    mae_values: dict[str, float] = {}
    mae_sem_values: dict[str, float] = {}
    for model_name in df_plot.index:
        try:
            m, s = _mae_mean_sem_over_sequence_splits(model_name, model_files[model_name])
            mae_values[model_name] = m
            mae_sem_values[model_name] = s
        except Exception as e:
            print(f"Warning: MAE split aggregation for {model_name}: {e}")
            mae_values[model_name] = float("nan")
            mae_sem_values[model_name] = float("nan")

    df_plot["MAE"] = pd.Series(mae_values)
    df_sem_plot = df_sem.copy()
    df_sem_plot["MAE"] = pd.Series(mae_sem_values)

    plot_cols = [c for c in plot_cols_requested if c in df_plot.columns]
    if not plot_cols:
        raise ValueError("None of the requested columns were found in df_plot.")

    plot_cols = plot_cols

    bar_df = df_plot[plot_cols].copy()
    bar_err_df = df_sem_plot.reindex(bar_df.index)[plot_cols].astype(float).fillna(0.0)

    x_labels = []
    for c in plot_cols:
        if c == "FINAL_COMPOSITE_SCORE":
            x_labels.append("Composite Score")
        elif c == "MAE":
            x_labels.append("Domain-normalized MAE")
        else:
            x_labels.append(c.replace("family_", "").replace("_", " ").title())

    selected_models = [m for m in models_to_plot_bar if m in bar_df.index]
    if not selected_models:
        raise ValueError("No bar-chart models found.")

    family_medians_bar = bar_df.median(axis=0, skipna=True)
    bar_plot_df = bar_df.fillna(family_medians_bar)
    bar_err_plot_df = bar_err_df.reindex(bar_plot_df.index)[plot_cols].astype(float).fillna(0.0)

    if normalize:
        mins = bar_plot_df.min(axis=0)
        maxs = bar_plot_df.max(axis=0)
        spans = (maxs - mins).replace(0, 1.0)
        bar_plot_df = (bar_plot_df - mins) / spans
        bar_err_plot_df = bar_err_plot_df / spans
        if "MAE" in bar_plot_df.columns:
            bar_plot_df["MAE"] = 1.0 - bar_plot_df["MAE"]

    run_colors_bar = {m: "#A23B72" for m in model_files.keys()}

    n_families = len(plot_cols)
    n_models = len(selected_models)
    x = np.arange(n_families)
    group_width = 0.8
    bar_width = (group_width / n_models) * 0.7

    fig_b, ax_b = plt.subplots(figsize=(6.5, 4.2))
    for i, model_name in enumerate(selected_models):
        offsets = x + (i - (n_models - 1) / 2) * bar_width
        values = bar_plot_df.loc[model_name, plot_cols].to_numpy(dtype=float)
        yerr = bar_err_plot_df.loc[model_name, plot_cols].to_numpy(dtype=float)
        ax_b.bar(
            offsets,
            values,
            width=bar_width,
            label=str(model_name),
            color=run_colors_bar.get(model_name, None),
            edgecolor="black",
            linewidth=0.6,
            alpha=0.9,
            yerr=yerr,
            capsize=2.0,
            error_kw=dict(elinewidth=0.8, capthick=0.8),
        )

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(x_labels, fontsize=11, rotation=10)
    ax_b.tick_params(axis="y", labelsize=10)
    ax_b.grid(axis="y", alpha=0.3)
    ax_b.set_axisbelow(True)
    ax_b.set_ylabel("Score", fontsize=10)
    if "FINAL_COMPOSITE_SCORE" in plot_cols:
        mae_idx = plot_cols.index("FINAL_COMPOSITE_SCORE")
        ax_b.axvline(mae_idx - 0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)

    vals = bar_plot_df.to_numpy(dtype=float)
    err = bar_err_plot_df.to_numpy(dtype=float)
    rng = np.concatenate([vals.ravel(), (vals + err).ravel(), (vals - err).ravel()])
    vmin_b = float(np.nanmin(rng))
    vmax_b = float(np.nanmax(rng))
    if not np.isfinite(vmin_b) or not np.isfinite(vmax_b):
        ymin_b, ymax_b = 0.0, 1.0
    elif normalize:
        ymin_b, ymax_b = 0.0, 1.05
    else:
        margin = 0.08 * (vmax_b - vmin_b) if not np.isclose(vmin_b, vmax_b) else 0.1
        ymin_b = max(0.0, vmin_b - margin)
        ymax_b = vmax_b + margin
        if ymax_b <= ymin_b:
            ymin_b, ymax_b = 0.0, 1.0
    ax_b.set_ylim(ymin_b, ymax_b)
    ax_b.legend(frameon=False, title="Model", fontsize=9)
    plt.tight_layout()
    bar_plot_path = outputs_dir / f"bar_family_scores_{sub_data_dir}_4split_{'norm' if normalize else 'raw'}.svg"
    fig_b.savefig(bar_plot_path, format="svg", bbox_inches="tight")
    print(f"Bar chart saved to: {bar_plot_path}")
    plt.close(fig_b)

# ---------------------------------------------------------------------------
# Metric dot plot (notebook cell 11) + horizontal SEM across splits
# ---------------------------------------------------------------------------
if RUN_METRIC_DOTPLOT:
    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["axes.linewidth"] = 0.8

    family_metric_map = OrderedDict(
        (fam, list(metrics.keys())) for fam, metrics in NEURO_FAMILY_METRICS.items()
    )
    family_order_dot = list(family_metric_map.keys())
    all_metrics = [m for fam in family_order_dot for m in family_metric_map[fam]]

    metric_display_names = {m: m.replace("_score01", "").replace("_", " ") for m in all_metrics}
    family_display_names = {fam: fam.replace("_", " ").title() for fam in family_order_dot}

    preferred_models_dot = list(model_files.keys())
    selected_models = [m for m in preferred_models_dot if m in all_scores]
    if not selected_models:
        raise ValueError("No models found in all_scores for dot plot.")

    plot_df = pd.DataFrame(
        [[float(all_scores[model].get(m, np.nan)) for m in all_metrics] for model in selected_models],
        index=selected_models,
        columns=all_metrics,
    ).T

    plot_sem_df = pd.DataFrame(
        [[float(all_scores_sem[model].get(m, np.nan)) for m in all_metrics] for model in selected_models],
        index=selected_models,
        columns=all_metrics,
    ).T

    metric_labels = [metric_display_names[m] for m in all_metrics]
    y = np.arange(len(all_metrics))
    offsets = np.linspace(-0.18, 0.18, len(selected_models))

    run_colors_dot = {m: "#A23B72" for m in model_files.keys()}

    fig_d, ax_d = plt.subplots(figsize=(10.8, 7.1))
    fig_d.subplots_adjust(left=0.50, right=0.98, top=0.88, bottom=0.10)

    family_lengths = [len(family_metric_map[f]) for f in family_order_dot]
    family_blocks = []
    start = 0
    for i, (fam, fam_len) in enumerate(zip(family_order_dot, family_lengths)):
        end = start + fam_len - 1
        family_blocks.append((fam, start, end))
        if i % 2 == 0:
            ax_d.axhspan(start - 0.5, end + 0.5, color="#F6F6F6", zorder=0)
        if end < len(all_metrics) - 1:
            ax_d.axhline(end + 0.5, color="#B0B0B0", lw=1.0, ls=(0, (3, 3)), zorder=1)
        start = end + 1

    vals_lo = plot_df.to_numpy(dtype=float) - np.nan_to_num(plot_sem_df.to_numpy(dtype=float), nan=0.0)
    vals_hi = plot_df.to_numpy(dtype=float) + np.nan_to_num(plot_sem_df.to_numpy(dtype=float), nan=0.0)
    mins = np.nanmin(vals_lo, axis=1)
    maxs = np.nanmax(vals_hi, axis=1)

    for yi, lo, hi in zip(y, mins, maxs):
        ax_d.hlines(yi, lo, hi, color="#D3D3D3", linewidth=4.5, zorder=1, capstyle="round")

    handles = []
    labels = []
    for off, model_name in zip(offsets, selected_models):
        xs = plot_df[model_name].to_numpy(dtype=float)
        xerrs = np.nan_to_num(plot_sem_df[model_name].to_numpy(dtype=float), nan=0.0)
        color = run_colors_dot.get(model_name, "#A23B72")
        eb = ax_d.errorbar(
            xs,
            y + off,
            xerr=xerrs,
            fmt="o",
            markersize=6.0,
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=2.0,
            label=model_name,
            zorder=3,
        )
        handles.append(eb[0])
        labels.append(model_name)

    ax_d.set_xlim(0, 1.0)
    ax_d.set_xticks(np.arange(0, 1.01, 0.2))
    ax_d.set_xlabel("Nethobench subscore (mean ± SEM over sequence splits)", fontsize=9)
    ax_d.set_yticks(y)
    ax_d.set_yticklabels(metric_labels, fontsize=9)
    ax_d.invert_yaxis()
    ax_d.xaxis.grid(True, color="#D0D0D0", linewidth=0.8)
    ax_d.yaxis.grid(False)
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)
    ax_d.tick_params(axis="y", length=0, pad=6)

    trans = mtransforms.blended_transform_factory(ax_d.transAxes, ax_d.transData)
    for fam, start, end in family_blocks:
        ymid = 0.5 * (start + end)
        ax_d.annotate(
            family_display_names[fam],
            xy=(0, ymid),
            xycoords=trans,
            xytext=(-160, 0),
            textcoords="offset points",
            rotation=60,
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#555555",
        )

    fig_d.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.6, 0.95),
        ncol=len(selected_models),
        frameon=False,
        fontsize=9,
        handletextpad=0.5,
        columnspacing=1,
    )

    dot_out = outputs_dir / f"metric_dotplot_{sub_data_dir}_4split.svg"
    fig_d.savefig(dot_out, format="svg", bbox_inches="tight")
    print(f"Saved metric dot plot to: {dot_out}")
    plt.close(fig_d)

# ---------------------------------------------------------------------------
# Horizon (notebook cells 13–14) — composite mean ± SEM over 4 sequence splits
# ---------------------------------------------------------------------------
if RUN_HORIZON_SCORES:
    USE_HORIZON_CACHE = True
    FORCE_RECOMPUTE_HORIZON = False
    SCORED_HORIZONS = [30, 120, 240]
    HORIZON_START_BY_GT = {"bench": 0, "sequifier": 0}

    preferred_model_order = list(model_files.keys())

    _horiz_tag = "_".join(str(h) for h in SCORED_HORIZONS)
    horizon_cache_path = outputs_dir / f"scores_cache_{sub_data_dir}_h{_horiz_tag}_4split_composite_only.json"

    def _align_gt_pred(gt_a: np.ndarray, pred_a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_seq = min(gt_a.shape[0], pred_a.shape[0])
        n_t = min(gt_a.shape[1], pred_a.shape[1])
        n_reg = min(gt_a.shape[2], pred_a.shape[2])
        return gt_a[:n_seq, :n_t, :n_reg], pred_a[:n_seq, :n_t, :n_reg]

    def _horizon_window_for_model(model_name: str, H: int) -> tuple[int, int]:
        gt_key = MODEL_TO_GT.get(model_name, "bench")
        start = int(HORIZON_START_BY_GT.get(gt_key, 0))
        end = start + int(H)
        return start, end

    horizon_defs = [(f"{H}", int(H)) for H in SCORED_HORIZONS]

    HORIZON_CACHE_SIGNATURE = {
        "kind": "horizon_composite_4split_2p",
        "horizons": [int(h) for h in SCORED_HORIZONS],
        "models": list(preferred_model_order),
        "horizon_start_by_gt": dict(HORIZON_START_BY_GT),
        "n_splits": N_SPLITS,
    }

    horizon_scores_nested: dict[str, dict[str, dict]] = {}

    if USE_HORIZON_CACHE and horizon_cache_path.exists() and (not FORCE_RECOMPUTE_HORIZON):
        print(f"\nLoading horizon cache from: {horizon_cache_path}")
        loaded_h = json.loads(horizon_cache_path.read_text(encoding="utf-8"))
        if loaded_h.get("_cache_signature") == HORIZON_CACHE_SIGNATURE:
            horizon_scores_nested = loaded_h.get("scores", {})
        else:
            print("Horizon cache signature mismatch; recomputing.")

    for model_name in preferred_model_order:
        pred_path = model_files[model_name]
        pred = np.load(pred_path, allow_pickle=False)
        if model_name not in horizon_scores_nested:
            horizon_scores_nested[model_name] = {}

        start_max, end_max = _horizon_window_for_model(model_name, max(SCORED_HORIZONS))
        gt_ref = _gt_for_model(model_name)
        g, p = _align_gt_pred(gt_ref, pred)
        if g.shape[1] < end_max or p.shape[1] < end_max:
            raise ValueError(
                f"{model_name}: need at least {end_max} timesteps, got gt={g.shape[1]}, pred={p.shape[1]}"
            )

        for label, H in horizon_defs:
            entry = horizon_scores_nested[model_name].get(str(H), {})
            cache_ok = bool(entry) and entry.get("FINAL_COMPOSITE_SCORE_sem") is not None
            if cache_ok and (not FORCE_RECOMPUTE_HORIZON):
                print(f"Skipping {model_name} @ H={H} (cached)")
                continue

            start_t, end_t = _horizon_window_for_model(model_name, H)
            gt_h = g[:, start_t:end_t, :]
            pred_h = p[:, start_t:end_t, :]

            split_indices = _sequence_split_indices(gt_h.shape[0], N_SPLITS)
            comp_splits = [
                float(_scores_for_arrays(gt_h[idx], pred_h[idx]).get("FINAL_COMPOSITE_SCORE", float("nan")))
                for idx in split_indices
            ]
            arr_c = np.asarray(comp_splits, dtype=float)
            comp_mean = float(np.nanmean(arr_c))
            comp_sem = _nansem_across_values(arr_c)

            horizon_scores_nested[model_name][str(H)] = {
                "horizon_label": label,
                "n_steps": H,
                "start_t": int(start_t),
                "end_t": int(end_t),
                "FINAL_COMPOSITE_SCORE": comp_mean,
                "FINAL_COMPOSITE_SCORE_sem": comp_sem,
            }
            print(
                f"Computed {model_name} @ H={H} (t={start_t}:{end_t}): "
                f"Composite={comp_mean:.6f} ± {comp_sem:.6f} (over {N_SPLITS} sequence splits)"
            )

    horizon_cache_path.write_text(
        json.dumps(
            {"_cache_signature": HORIZON_CACHE_SIGNATURE, "scores": horizon_scores_nested},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved horizon cache to: {horizon_cache_path}")

    rows_h = []
    for model_name in preferred_model_order:
        for _, H in horizon_defs:
            entry = horizon_scores_nested.get(model_name, {}).get(str(H))
            if not entry:
                continue
            rows_h.append(
                {
                    "model": model_name,
                    "horizon_label": entry["horizon_label"],
                    "n_steps": int(entry["n_steps"]),
                    "FINAL_COMPOSITE_SCORE": float(entry["FINAL_COMPOSITE_SCORE"]),
                    "FINAL_COMPOSITE_SCORE_sem": float(entry.get("FINAL_COMPOSITE_SCORE_sem", float("nan"))),
                }
            )

    horizon_long_df = pd.DataFrame(rows_h).sort_values(["n_steps", "model"]).reset_index(drop=True)
    print("\nHorizon sweep (mean ± SEM over sequence splits):")
    print(horizon_long_df.to_string(index=False))

    run_colors_h = {m: "#A23B72" for m in model_files.keys()}

    fig_nb, ax_nb = plt.subplots(1, 1, figsize=(5.4, 4.1))
    ax_nb.grid(True, which="major", color="#D9D9D9", linewidth=0.8)
    ax_nb.set_axisbelow(True)
    ax_nb.set_xticks(SCORED_HORIZONS)
    ax_nb.set_xticklabels([str(h) for h in SCORED_HORIZONS], fontsize=10)
    ax_nb.set_xlabel("Prediction horizon (timesteps)", fontsize=11)
    ax_nb.set_ylabel("Nethobench composite score", fontsize=11)

    for model_name in preferred_model_order:
        sub = horizon_long_df[horizon_long_df["model"] == model_name].sort_values("n_steps")
        if sub.empty:
            continue
        color = run_colors_h.get(model_name, "#4C78A8")
        ax_nb.errorbar(
            sub["n_steps"],
            sub["FINAL_COMPOSITE_SCORE"],
            yerr=sub["FINAL_COMPOSITE_SCORE_sem"],
            linestyle="-",
            capsize=3,
            color=color,
            marker="o",
            linewidth=1.9,
            markersize=6.5,
            label=model_name,
        )

    ax_nb.legend(frameon=False, fontsize=9, loc="best")
    fig_nb.tight_layout()
    nb_plot_path = outputs_dir / f"horizon_nethobench_{sub_data_dir}_{_horiz_tag}_4split_sem.svg"
    fig_nb.savefig(nb_plot_path, format="svg", bbox_inches="tight")
    print(f"Saved Nethobench horizon SVG to: {nb_plot_path}")
    plt.close(fig_nb)

print("\nDone.")
