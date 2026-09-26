#!/usr/bin/env python3
"""
Metric-faithful Nethobench diagnostic visualizations for the 90_810 arrays.

Tensor paths default to the three training-seed directories below
``evaluation_results/90_810``. Figures and score CSVs go under
``output/neuro_metric_specific_visualizations_90_810/`` at the repo root.
Official score tables are read from the three-seed score cache by default; the
figures visualize raw quantities aggregated across those same predictions.

When official helpers expose the intermediate, this file imports them.  For the
notebook-only metrics (KL/JSD, QNT, Mean), the extractor functions below mirror
the exact notebook cell formulas and parameters.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_NETHOBENCH_INSTALL = _REPO_ROOT / "nethobench"
if not (_NETHOBENCH_INSTALL / "nethobench" / "__init__.py").is_file():
    raise RuntimeError(
        f"Required Nethobench checkout not found: {_NETHOBENCH_INSTALL}"
    )
_nb_path = str(_NETHOBENCH_INSTALL.resolve())
if _nb_path not in sys.path:
    sys.path.insert(0, _nb_path)

NETHOBENCH_PKG = _NETHOBENCH_INSTALL / "nethobench"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from nethobench.neuro.metrics import additional as addm
from nethobench.neuro.metrics import sensitive as smc
from scipy.linalg import subspace_angles
from scipy.stats import entropy, kurtosis, skew

from neuro_scoring_windows import align_forecast_only

_SENSITIVE_MODULE_PATH = Path(smc.__file__).resolve()
if _NETHOBENCH_INSTALL.resolve() not in _SENSITIVE_MODULE_PATH.parents:
    raise RuntimeError(
        "Resolved Nethobench metrics outside nethobench: "
        f"{_SENSITIVE_MODULE_PATH}"
    )
_ADDITIONAL_MODULE_PATH = Path(addm.__file__).resolve()
if _NETHOBENCH_INSTALL.resolve() not in _ADDITIONAL_MODULE_PATH.parents:
    raise RuntimeError(
        "Resolved Nethobench metrics outside nethobench: "
        f"{_ADDITIONAL_MODULE_PATH}"
    )

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


base = load_module_from_path(
    "_neuro_eval_common",
    Path(__file__).resolve().parent / "neuro_eval_common.py",
)


EPS = 1e-12
FULL_SEQUENCE_LENGTH = 810
CONTEXT_STEPS = 90

SELECTED_METRICS = base.SELECTED_METRICS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate metric-specific Nethobench diagnostic figures for 90_810."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_REPO_ROOT / "evaluation_results" / "90_810",
        help=(
            "Root containing val_seed_102_train_seed_* directories, or one "
            "explicit directory containing 90_810 tensors."
        ),
    )
    parser.add_argument(
        "--score-cache",
        type=Path,
        default=(
            _REPO_ROOT
            / "output"
            / "neuro_subscores_from_npy_merged_4split_3seeds_new"
            / "scores_cache_90_810_4split_3seeds.json"
        ),
        help="Existing three-training-seed official Nethobench score cache.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "output" / "neuro_metric_specific_visualizations_90_810",
        help="Directory for metric-specific SVGs and manifest.",
    )
    parser.add_argument(
        "--skip-score-cache",
        action="store_true",
        help="Skip exporting the official split-score CSVs from the existing score pipeline.",
    )
    parser.add_argument(
        "--force-scores",
        action="store_true",
        help=(
            "Ignore --score-cache and recompute official split scores with "
            "the nethobench checkout."
        ),
    )
    parser.add_argument("--n-splits", type=int, default=base.N_SPLITS)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def load_three_seed_model_arrays(
    data_dir: Path,
    score_cache: Path,
) -> dict[str, base.ModelArrays]:
    """Load and concatenate predictions from the cache-declared seed folders."""
    if (data_dir / next(iter(base.MODEL_FILES.values()))).is_file():
        seed_dirs = [data_dir]
    else:
        if not score_cache.is_file():
            raise FileNotFoundError(
                f"Score cache is required to identify training-seed folders: "
                f"{score_cache}"
            )
        payload = json.loads(score_cache.read_text(encoding="utf-8"))
        folder_names = payload.get("_cache_signature", {}).get("seed_folders")
        if not isinstance(folder_names, list) or not folder_names:
            raise ValueError(
                f"{score_cache} does not declare non-empty seed_folders"
            )
        seed_dirs = [data_dir / str(folder_name) for folder_name in folder_names]

    loaded_by_seed = [base.load_model_arrays(seed_dir) for seed_dir in seed_dirs]
    combined: dict[str, base.ModelArrays] = {}
    for model in base.MODEL_ORDER:
        seed_arrays = [arrays[model] for arrays in loaded_by_seed]
        gt_keys = {arrays.gt_key for arrays in seed_arrays}
        if len(gt_keys) != 1:
            raise ValueError(f"Inconsistent ground-truth keys for {model}: {gt_keys}")
        # Match the four-split scoring scripts: tensors with 810 prediction
        # samples contain 90 context samples followed by the 720-sample
        # forecast.  Align each training seed before concatenating it for the
        # diagnostic plots.  The former generic min-shape alignment silently
        # included the context window in every visualization.
        aligned = [
            align_forecast_only(
                arrays.gt,
                arrays.pred,
                full_sequence_length=FULL_SEQUENCE_LENGTH,
                context_steps=CONTEXT_STEPS,
            )
            for arrays in seed_arrays
        ]
        combined[model] = base.ModelArrays(
            gt=np.concatenate([pair[0] for pair in aligned], axis=0),
            pred=np.concatenate([pair[1] for pair in aligned], axis=0),
            gt_key=seed_arrays[0].gt_key,
        )

    print(
        "Loaded prediction tensors from:",
        ", ".join(seed_dir.name for seed_dir in seed_dirs),
    )
    return combined


def selected_metric_keys() -> list[str]:
    return [metric for metrics in SELECTED_METRICS.values() for metric in metrics]


def family_for_key(score_key: str) -> str:
    for family, keys in SELECTED_METRICS.items():
        if score_key in keys:
            return family
    return "unknown"


def align_for_metric(gt: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match the notebook/additional helper alignment behavior."""
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if pred.shape[1] != gt.shape[1] and pred.shape[1] % gt.shape[1] == 0:
        factor = pred.shape[1] // gt.shape[1]
        pred = pred.reshape(pred.shape[0], gt.shape[1], factor, pred.shape[2]).mean(axis=2)
    elif pred.shape[1] != gt.shape[1]:
        keep = min(gt.shape[1], pred.shape[1])
        gt = gt[:, :keep, :]
        pred = pred[:, :keep, :]
    if gt.shape != pred.shape:
        raise ValueError(f"Aligned mismatch: {gt.shape} vs {pred.shape}")
    return gt, pred


def panel_grid(n_panels: int) -> tuple[plt.Figure, np.ndarray]:
    n_cols = 3
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11.4, 3.35 * n_rows), constrained_layout=True)
    return fig, np.asarray(axes).ravel()


def finalize_unused_axes(axes: np.ndarray, used: int) -> None:
    for ax in axes[used:]:
        ax.axis("off")


def save(fig: plt.Figure, output_dir: Path, filename: str) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, format="svg", bbox_inches="tight")
    plt.close(fig)
    return filename


def matrix_corr(a: np.ndarray, b: np.ndarray, *, upper_only: bool = True) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if upper_only and a.ndim == 2 and a.shape[0] == a.shape[1]:
        idx = np.triu_indices_from(a, k=1)
        av = a[idx]
        bv = b[idx]
    else:
        av = a.ravel()
        bv = b.ravel()
    mask = np.isfinite(av) & np.isfinite(bv)
    if mask.sum() < 3:
        return np.nan
    av = av[mask]
    bv = bv[mask]
    if np.nanstd(av) <= EPS or np.nanstd(bv) <= EPS:
        return np.nan
    return float(np.corrcoef(av, bv)[0, 1])


def total_variation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(0.5 * np.nansum(np.abs(a - b)))


def entropy_of_prob(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = p[np.isfinite(p) & (p > 0)]
    return float(-np.sum(p * np.log(p))) if p.size else np.nan


def finite_percentile(values: list[np.ndarray], q: float, default: float) -> float:
    flat = np.concatenate([np.asarray(v, dtype=np.float64).ravel() for v in values if v is not None])
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return default
    return float(np.nanpercentile(flat, q))


def plot_matrix_triptych(
    matrices: list[tuple[str, np.ndarray, np.ndarray, str]],
    output_dir: Path,
    filename: str,
    suptitle: str,
    *,
    gt_pred_cmap: str = "viridis",
    gt_pred_diverging: bool = False,
    delta_label: str = "Prediction - GT",
    sequential_floor: float = 0.0,
) -> str:
    n_models = len(matrices)
    fig, axes = plt.subplots(n_models, 3, figsize=(9.8, max(2.2 * n_models, 4.0)), constrained_layout=True)
    axes = np.asarray(axes)
    gt_pred_values = [m for _, gt, pred, _ in matrices for m in (gt, pred)]
    delta_values = [pred - gt for _, gt, pred, _ in matrices]

    if gt_pred_diverging:
        lim = max(0.01, finite_percentile([np.abs(v) for v in gt_pred_values], 98, 1.0))
        gt_pred_norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
        gt_pred_vmin = gt_pred_vmax = None
    else:
        gt_pred_norm = None
        gt_pred_vmin = sequential_floor
        gt_pred_vmax = max(sequential_floor + EPS, finite_percentile(gt_pred_values, 98, 1.0))

    delta_lim = max(0.005, finite_percentile([np.abs(v) for v in delta_values], 98, 0.1))
    delta_norm = TwoSlopeNorm(vmin=-delta_lim, vcenter=0.0, vmax=delta_lim)

    seq_im = delta_im = None
    for row, (model_label, gt, pred, summary) in enumerate(matrices):
        delta = pred - gt
        for col, (mat, title) in enumerate(((gt, "GT"), (pred, "Prediction"))):
            ax = axes[row, col]
            seq_im = ax.imshow(
                mat,
                cmap=gt_pred_cmap,
                norm=gt_pred_norm,
                vmin=gt_pred_vmin if gt_pred_norm is None else None,
                vmax=gt_pred_vmax if gt_pred_norm is None else None,
            )
            ax.set_title(f"{model_label} {title}" if col == 0 else title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

        ax = axes[row, 2]
        delta_im = ax.imshow(delta, cmap="coolwarm", norm=delta_norm)
        ax.set_title(f"Delta\n{summary}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(seq_im, ax=axes[:, :2].ravel().tolist(), fraction=0.018, pad=0.01, label="GT / prediction")
    fig.colorbar(delta_im, ax=axes[:, 2].ravel().tolist(), fraction=0.035, pad=0.02, label=delta_label)
    fig.suptitle(suptitle, y=1.01)
    return save(fig, output_dir, filename)


def robust_iqr(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 10:
        s = float(np.nanstd(x))
        return s if np.isfinite(s) and s > 0 else 1.0
    q25, q75 = np.nanquantile(x, [0.25, 0.75])
    s = float(q75 - q25)
    if not np.isfinite(s) or s <= 0:
        s = float(np.nanstd(x))
    return s if np.isfinite(s) and s > 0 else 1.0


def binary_topology_from_corr(corr: np.ndarray, frac: float = 0.15) -> np.ndarray:
    """Exact copy of the legacy GRAPH top-edge helper used by the official path."""
    corr = np.abs(np.asarray(corr, dtype=np.float64))
    np.fill_diagonal(corr, 0.0)
    iu = np.triu_indices_from(corr, k=1)
    n_edges = len(iu[0])
    topk = max(3, int(np.ceil(frac * n_edges)))
    topk = min(topk, n_edges)
    idx = np.argsort(corr[iu])[-topk:]
    adj = np.zeros_like(corr, dtype=np.int64)
    adj[iu[0][idx], iu[1][idx]] = 1
    return adj + adj.T


def binary_clustering(adj: np.ndarray) -> np.ndarray:
    """Exact copy of the legacy GRAPH binary clustering helper."""
    adj = np.asarray(adj, dtype=np.int64)
    n_nodes = adj.shape[0]
    out = np.zeros(n_nodes, dtype=np.float64)
    for idx in range(n_nodes):
        neighbors = np.flatnonzero(adj[idx])
        degree = neighbors.size
        if degree < 2:
            continue
        sub = adj[np.ix_(neighbors, neighbors)]
        edges = float(np.sum(sub) / 2.0)
        out[idx] = (2.0 * edges) / (degree * (degree - 1))
    return out


# ---------------------------------------------------------------------------
# Notebook-only selected distribution metrics.
# ---------------------------------------------------------------------------


def tail_binned_hist(
    gt_vals: np.ndarray,
    pred_vals: np.ndarray,
    bins: int = 60,
    support_q: tuple[float, float] = (0.001, 0.999),
    eps: float = 1e-12,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Exact mirror of the KL/JSD notebook histogram construction."""
    gt_vals = np.asarray(gt_vals, dtype=np.float64)
    pred_vals = np.asarray(pred_vals, dtype=np.float64)
    gt_vals = gt_vals[np.isfinite(gt_vals)]
    pred_vals = pred_vals[np.isfinite(pred_vals)]
    if gt_vals.size < 10 or pred_vals.size < 10:
        return None, None, None
    pool = np.concatenate([gt_vals, pred_vals])
    lo = np.quantile(pool, support_q[0])
    hi = np.quantile(pool, support_q[1])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo = float(np.min(pool))
        hi = float(np.max(pool))
        if lo >= hi:
            hi = lo + 1e-6
    interior_edges = np.linspace(lo, hi, bins + 1)
    gt_counts = [np.sum(gt_vals < lo)]
    pred_counts = [np.sum(pred_vals < lo)]
    h_gt, _ = np.histogram(gt_vals, bins=interior_edges)
    h_pred, _ = np.histogram(pred_vals, bins=interior_edges)
    gt_counts += list(h_gt)
    pred_counts += list(h_pred)
    gt_counts += [np.sum(gt_vals > hi)]
    pred_counts += [np.sum(pred_vals > hi)]
    gt_counts = np.asarray(gt_counts, dtype=np.float64)
    pred_counts = np.asarray(pred_counts, dtype=np.float64)
    p = (gt_counts + eps) / (gt_counts + eps).sum()
    q = (pred_counts + eps) / (pred_counts + eps).sum()
    edges = np.concatenate(([lo - (hi - lo) / bins], interior_edges, [hi + (hi - lo) / bins]))
    return p, q, edges


def extract_kl(gt: np.ndarray, pred: np.ndarray) -> dict[str, np.ndarray | float]:
    """KL_or_JSD_score01: per-sequence/region symmetric KL, then geo mean and q10."""
    gt, pred = align_for_metric(gt, pred)
    n_seq, _, n_reg = gt.shape
    kl_sym = np.full((n_seq, n_reg), np.nan, dtype=np.float64)
    for s in range(n_seq):
        for r in range(n_reg):
            p, q, _ = tail_binned_hist(gt[s, :, r], pred[s, :, r])
            if p is None or q is None:
                continue
            kl_sym[s, r] = 0.5 * (entropy(p, q) + entropy(q, p))
    kl_geo_seq = np.full(n_seq, np.nan, dtype=np.float64)
    for s in range(n_seq):
        row = kl_sym[s]
        row = row[np.isfinite(row)]
        if row.size:
            sim = np.clip(1.0 / (1.0 + row), 1e-12, 1.0)
            kl_geo_seq[s] = float(np.exp(np.mean(np.log(sim))))
    valid = kl_geo_seq[np.isfinite(kl_geo_seq)]
    kl_mean = float(np.mean(valid)) if valid.size else np.nan
    kl_q10 = float(np.quantile(valid, 0.10)) if valid.size else np.nan
    score = 0.5 * (kl_mean + kl_q10) if np.isfinite(kl_mean) and np.isfinite(kl_q10) else np.nan
    return {"kl_sym": kl_sym, "kl_geo_seq": kl_geo_seq, "KL_mean": kl_mean, "KL_q10": kl_q10, "score": score}


def extract_quantile(gt: np.ndarray, pred: np.ndarray) -> dict[str, np.ndarray | float]:
    """QNT_score01: tail quantile distances, top-25% worst regions per sequence."""
    gt, pred = align_for_metric(gt, pred)
    n_seq, _, n_reg = gt.shape
    rng = np.random.default_rng(0)
    quantiles = np.linspace(0.01, 0.99, 99)
    tail_mask = (quantiles <= 0.10) | (quantiles >= 0.90)
    iqr_gt = np.array([robust_iqr(gt[:, :, r].reshape(-1)) for r in range(n_reg)], dtype=np.float64)
    d_tail = np.full((n_seq, n_reg), np.nan, dtype=np.float64)
    d_full = np.full((n_seq, n_reg), np.nan, dtype=np.float64)
    per_quantile_err = np.full((n_reg, quantiles.size), np.nan, dtype=np.float64)
    for r in range(n_reg):
        err_rows = []
        for s in range(n_seq):
            x = gt[s, :, r]
            y = pred[s, :, r]
            x = x[np.isfinite(x)]
            y = y[np.isfinite(y)]
            if x.size < 80 or y.size < 80:
                continue
            if x.size > 1200:
                x = x[rng.choice(x.size, size=1200, replace=False)]
            if y.size > 1200:
                y = y[rng.choice(y.size, size=1200, replace=False)]
            dq = np.abs(np.quantile(x, quantiles) - np.quantile(y, quantiles)) / (iqr_gt[r] + EPS)
            d_full[s, r] = float(np.mean(dq))
            d_tail[s, r] = float(np.mean(dq[tail_mask]))
            err_rows.append(dq)
        if err_rows:
            per_quantile_err[r] = np.nanmean(np.vstack(err_rows), axis=0)
    k = np.maximum(1, np.ceil(0.25 * np.sum(np.isfinite(d_tail), axis=1)).astype(int))
    d_seq = np.full(n_seq, np.nan, dtype=np.float64)
    for s in range(n_seq):
        row = d_tail[s]
        row = row[np.isfinite(row)]
        if row.size:
            kk = min(int(k[s]), row.size)
            d_seq[s] = float(np.mean(np.sort(row)[-kk:]))
    d = float(np.nanmean(d_seq)) if np.isfinite(d_seq).any() else np.nan
    return {
        "quantiles": quantiles,
        "tail_mask": tail_mask,
        "d_tail": d_tail,
        "d_full": d_full,
        "per_quantile_err": per_quantile_err,
        "D_seq": d_seq,
        "D": d,
        "score": float(1.0 / (1.0 + d)) if np.isfinite(d) else np.nan,
    }


def extract_mean(gt: np.ndarray, pred: np.ndarray) -> dict[str, np.ndarray | float | int]:
    """Mean_score01: top-10% region sequence mean discrepancy scaled by GT IQR."""
    gt, pred = align_for_metric(gt, pred)
    mu_gt = np.nanmean(gt, axis=1)
    mu_pred = np.nanmean(pred, axis=1)
    iqr_gt = np.array([robust_iqr(gt[:, :, r].reshape(-1)) for r in range(gt.shape[2])], dtype=np.float64)
    d = np.abs(mu_gt - mu_pred) / (iqr_gt[None, :] + EPS)
    k = max(1, int(np.ceil(0.1 * gt.shape[2])))
    d_seq = np.full(gt.shape[0], np.nan, dtype=np.float64)
    for s in range(gt.shape[0]):
        row = d[s]
        row = row[np.isfinite(row)]
        if row.size:
            d_seq[s] = float(np.mean(np.sort(row)[-min(k, row.size) :]))
    d_mean = float(np.nanmean(d_seq)) if np.isfinite(d_seq).any() else np.nan
    return {"mean_delta": mu_pred - mu_gt, "scaled_abs_delta": d, "D_seq_top10": d_seq, "D": d_mean, "K": k, "score": 1.0 / (1.0 + d_mean)}


def extract_moments(gt: np.ndarray, pred: np.ndarray) -> dict[str, np.ndarray | float]:
    """MOM_score01: legacy pooled region variance/skew/kurtosis distances."""
    gt, pred = align_for_metric(gt, pred)
    rows = []
    for r in range(gt.shape[-1]):
        g = gt[:, :, r].reshape(-1)
        p = pred[:, :, r].reshape(-1)
        mask = np.isfinite(g) & np.isfinite(p)
        g = g[mask]
        p = p[mask]
        if g.size < 24:
            rows.append([np.nan] * 8)
            continue
        var_g = float(np.var(g)) + smc.EPS
        var_p = float(np.var(p)) + smc.EPS
        skew_g = float(skew(g, bias=False))
        skew_p = float(skew(p, bias=False))
        kurt_g = float(kurtosis(g, fisher=True, bias=False))
        kurt_p = float(kurtosis(p, fisher=True, bias=False))
        dist = abs(np.log(var_p / var_g)) + 0.50 * abs(skew_p - skew_g) + 0.25 * abs(kurt_p - kurt_g)
        rows.append([var_g, var_p, skew_g, skew_p, kurt_g, kurt_p, dist, 1.0 / (1.0 + dist)])
    arr = np.asarray(rows, dtype=np.float64)
    return {"components": arr, "score": float(np.nanmean(arr[:, 7]))}


# ---------------------------------------------------------------------------
# Official helper-backed metric extractors.
# ---------------------------------------------------------------------------


def extract_graph(gt: np.ndarray, pred: np.ndarray) -> dict[str, np.ndarray | float]:
    """GRAPH_score01: exact legacy top-15% corr topology, degree, clustering pieces."""
    gt, pred = align_for_metric(gt, pred)
    xg = gt.reshape(-1, gt.shape[-1])
    xp = pred.reshape(-1, pred.shape[-1])
    xg, xp = smc.finite_rows(xg, xp)
    cg = smc.safe_corrcoef(xg)
    cp = smc.safe_corrcoef(xp)
    if cg is None or cp is None:
        raise ValueError("Insufficient finite rows for graph metric.")
    ag = binary_topology_from_corr(cg)
    ap = binary_topology_from_corr(cp)
    edges_g = set(zip(*np.where(np.triu(ag, k=1) > 0)))
    edges_p = set(zip(*np.where(np.triu(ap, k=1) > 0)))
    union = edges_g | edges_p
    jaccard = float(len(edges_g & edges_p) / max(len(union), 1))
    deg_g = np.sum(np.abs(cg), axis=0)
    deg_p = np.sum(np.abs(cp), axis=0)
    clust_g = binary_clustering(ag)
    clust_p = binary_clustering(ap)
    return {
        "corr_gt": cg,
        "corr_pred": cp,
        "adj_gt": ag,
        "adj_pred": ap,
        "degree_delta": deg_p - deg_g,
        "cluster_delta": clust_p - clust_g,
        "jaccard": jaccard,
    }


def standardized_flat(gt: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gt, pred = align_for_metric(gt, pred)
    return addm._standardize_columns(addm._pooled_rows(gt)), addm._standardize_columns(addm._pooled_rows(pred))


def extract_additional(gt: np.ndarray, pred: np.ndarray) -> dict[str, object]:
    gt, pred = align_for_metric(gt, pred)
    gt_flat_raw = addm._pooled_rows(gt)
    pred_flat_raw = addm._pooled_rows(pred)
    gt_flat = addm._standardize_columns(gt_flat_raw)
    pred_flat = addm._standardize_columns(pred_flat_raw)
    gt_cov, gt_prec = addm._ledoit_cov_precision(gt_flat)
    pred_cov, pred_prec = addm._ledoit_cov_precision(pred_flat)
    gt_basis = addm._principal_subspace(gt_cov)
    pred_basis = addm._principal_subspace(pred_cov)
    angles = np.asarray([], dtype=np.float64)
    if gt_basis is not None and pred_basis is not None:
        k = min(gt_basis.shape[1], pred_basis.shape[1])
        angles = subspace_angles(gt_basis[:, :k], pred_basis[:, :k])
    return {
        "mi_gt": addm._mi_matrix(gt_flat),
        "mi_pred": addm._mi_matrix(pred_flat),
        "lagged_cov": {
            lag: (addm._lagged_covariance(gt_flat, lag), addm._lagged_covariance(pred_flat, lag))
            for lag in (1, 2, 4)
        },
        "var1_gt": addm._var1_coefficients(gt_flat),
        "var1_pred": addm._var1_coefficients(pred_flat),
        "angles": angles,
        "cos2": np.cos(angles) ** 2 if angles.size else np.asarray([], dtype=np.float64),
        "state_ref": addm._prepare_latent_state_reference(gt, pred, gt_flat_raw, pred_flat_raw),
    }


def state_bundle_from_ref(ref: dict[str, object], k: int, lag: int | None = None) -> dict[str, np.ndarray]:
    centers = addm._fit_kmeans_centers(ref["gt_proj_fit"], k, max_fit_points=10000, random_state=k)
    if lag is None:
        gt_assign = addm._assign_to_centers(ref["gt_proj_eval"], centers)
        pred_assign = addm._assign_to_centers(ref["pred_proj_eval"], centers)
        gt_hist = np.bincount(gt_assign, minlength=k).astype(np.float64)
        pred_hist = np.bincount(pred_assign, minlength=k).astype(np.float64)
        return {"gt": gt_hist / gt_hist.sum(), "pred": pred_hist / pred_hist.sum()}
    gt_seq = [addm._assign_to_centers(seq, centers) for seq in ref["gt_seq_proj"]]
    pred_seq = [addm._assign_to_centers(seq, centers) for seq in ref["pred_seq_proj"]]
    gt_mat = np.zeros((k, k), dtype=np.float64)
    pred_mat = np.zeros((k, k), dtype=np.float64)
    for g, p in zip(gt_seq, pred_seq):
        if g.size <= lag or p.size <= lag:
            continue
        np.add.at(gt_mat, (g[:-lag], g[lag:]), 1.0)
        np.add.at(pred_mat, (p[:-lag], p[lag:]), 1.0)
    gt_mat = gt_mat / max(float(gt_mat.sum()), EPS)
    pred_mat = pred_mat / max(float(pred_mat.sum()), EPS)
    return {"gt": gt_mat, "pred": pred_mat}


def extract_trajectory(gt: np.ndarray, pred: np.ndarray) -> dict[str, np.ndarray | float]:
    """TRJDIST_score01: exact GT-PCA occupancy, speed/turn, and path features."""
    gt, pred = smc.align_arrays(gt, pred)
    xg = gt.reshape(-1, gt.shape[-1])
    xp = pred.reshape(-1, pred.shape[-1])
    xg, xp = smc.finite_rows(xg, xp)
    xg_z, xp_z = smc._standardize_with_gt(xg, xp)
    k = smc._choose_k(xg_z, k_max=min(3, xg_z.shape[1]))
    from sklearn.decomposition import PCA

    pca = PCA(n_components=k, svd_solver="full", random_state=0).fit(xg_z)
    zg = pca.transform(xg_z)
    zp = pca.transform(xp_z)
    occ = []
    for dim in range(k):
        lo, hi = np.quantile(zg[:, dim], [0.02, 0.98])
        edges = np.linspace(lo, hi, 13)
        hg, _ = np.histogram(np.clip(zg[:, dim], lo, hi), bins=edges)
        hp, _ = np.histogram(np.clip(zp[:, dim], lo, hi), bins=edges)
        occ.append((hg / max(hg.sum(), 1), hp / max(hp.sum(), 1)))
    mu = np.mean(xg, axis=0, keepdims=True)
    sd = np.std(xg, axis=0, keepdims=True)
    sd = np.where(sd < smc.EPS, 1.0, sd)
    zg_seq = pca.transform((gt.reshape(-1, gt.shape[-1]) - mu) / sd).reshape(gt.shape[0], gt.shape[1], k)
    zp_seq = pca.transform((pred.reshape(-1, pred.shape[-1]) - mu) / sd).reshape(pred.shape[0], pred.shape[1], k)
    vg = np.diff(zg_seq, axis=1).reshape(-1, k)
    vp = np.diff(zp_seq, axis=1).reshape(-1, k)
    speed_g = np.linalg.norm(vg, axis=1)
    speed_p = np.linalg.norm(vp, axis=1)
    turn_g = np.sum(vg[1:] * vg[:-1], axis=1) / ((np.linalg.norm(vg[1:], axis=1) * np.linalg.norm(vg[:-1], axis=1)) + smc.EPS)
    turn_p = np.sum(vp[1:] * vp[:-1], axis=1) / ((np.linalg.norm(vp[1:], axis=1) * np.linalg.norm(vp[:-1], axis=1)) + smc.EPS)
    path_g = []
    path_p = []
    for seq_g, seq_p in zip(zg_seq, zp_seq):
        def seq_features(z: np.ndarray) -> np.ndarray:
            v = np.diff(z, axis=0)
            path = np.sum(np.linalg.norm(v, axis=1))
            disp = np.linalg.norm(z[-1] - z[0])
            radius = np.mean(np.linalg.norm(z - np.mean(z, axis=0, keepdims=True), axis=1))
            persistence = disp / (path + smc.EPS)
            speed = np.linalg.norm(v, axis=1)
            speed_lag1 = smc.correlation_score(speed[1:], speed[:-1])
            return np.asarray([path, disp, radius, persistence, speed_lag1], dtype=np.float64)

        if seq_g.shape[0] > 4:
            path_g.append(seq_features(seq_g))
            path_p.append(seq_features(seq_p))
    return {
        "occupancy": occ,
        "speed_g": speed_g,
        "speed_p": speed_p,
        "turn_g": turn_g,
        "turn_p": turn_p,
        "path_g": np.vstack(path_g) if path_g else np.empty((0, 5)),
        "path_p": np.vstack(path_p) if path_p else np.empty((0, 5)),
    }


def extract_mani(gt: np.ndarray, pred: np.ndarray) -> dict[str, object]:
    """MANI_score01: exact PH lifetime clouds plus exact kNN profile cloud."""
    out: dict[str, object] = {}
    zg_s, zp_s = smc._stratified_latent_clouds(gt, pred, k_max=3, points_per_seq=2, max_sequences=48, seed=5)
    out["stratified_cloud_gt"] = zg_s
    out["stratified_cloud_pred"] = zp_s
    out["lifetimes_gt"] = [np.asarray([]), np.asarray([])]
    out["lifetimes_pred"] = [np.asarray([]), np.asarray([])]
    if zg_s is not None and zp_s is not None:
        dg_g = smc._ripser_diagrams(zg_s, maxdim=1, n_perm=64)
        dg_p = smc._ripser_diagrams(zp_s, maxdim=1, n_perm=64)
        if dg_g is not None and dg_p is not None:
            out["lifetimes_gt"] = [smc._lifetimes(dg_g[0]), smc._lifetimes(dg_g[1])]
            out["lifetimes_pred"] = [smc._lifetimes(dg_p[0]), smc._lifetimes(dg_p[1])]
    zg_k, zp_k = smc._pooled_latent_clouds(gt, pred, k_max=3, n_points=128, seed=3)
    out["knn_gt"] = smc._knn_distance_profile(zg_k, k=5) if zg_k is not None else np.asarray([])
    out["knn_pred"] = smc._knn_distance_profile(zp_k, k=5) if zp_k is not None else np.asarray([])
    return out


# ---------------------------------------------------------------------------
# Plot functions.
# ---------------------------------------------------------------------------


def plot_kl(
    model_arrays: dict[str, base.ModelArrays],
    output_dir: Path,
    official_means: dict[str, dict[str, float]],
) -> str:
    # Raw quantity: symmetric KL per sequence and region. Lower KL is better;
    # the scalar score converts KL to similarities 1/(1+KL), geometric-averages
    # regions per sequence, then averages mean and q10 sequence similarity.
    fig, axes = plt.subplots(
        1,
        len(base.MODEL_ORDER),
        figsize=(2.8 * len(base.MODEL_ORDER), 2.9),
        constrained_layout=True,
        squeeze=False,
    )
    for column, model in enumerate(base.MODEL_ORDER):
        gt, pred = align_for_metric(model_arrays[model].gt, model_arrays[model].pred)
        ext = extract_kl(gt, pred)
        ax = axes[0, column]
        vals = ext["kl_sym"].ravel()
        vals = vals[np.isfinite(vals)]
        ax.hist(vals, bins=40, color=base.MODEL_COLORS[model], alpha=0.78)
        median_kl = float(np.nanmedian(vals)) if vals.size else np.nan
        ax.axvline(median_kl, color="black", ls="--", lw=1)
        score = official_means[model]["KL_or_JSD_score01"]
        ax.set_title(f"{base.MODEL_LABELS[model]} score={score:.3f}", fontsize=9)
        ax.set_xlabel("Symmetric KL per sequence-region")
        ax.set_ylabel("Count")
        ax.set_box_aspect(0.9)
    return save(fig, output_dir, "metric_KL_or_JSD_gt_pred_histograms.svg")


def plot_qnt(
    model_arrays: dict[str, base.ModelArrays],
    output_dir: Path,
) -> str:
    """
    Single-panel QNT plot:
    one GT quantile curve + one prediction quantile curve per model.
    Assumes all models share the same GT.
    """
    fig, ax = plt.subplots(figsize=(7.4, 5.2), constrained_layout=True)

    gt_plotted = False

    for model in base.MODEL_ORDER:
        gt, pred = align_for_metric(model_arrays[model].gt, model_arrays[model].pred)
        ext = extract_quantile(gt, pred)
        q = ext["quantiles"]

        gt_vals = gt[np.isfinite(gt)]
        pred_vals = pred[np.isfinite(pred)]

        if not gt_plotted:
            ax.plot(q, np.quantile(gt_vals, q), color="black", lw=2.2, label="GT")
            gt_plotted = True

        ax.plot(
            q,
            np.quantile(pred_vals, q),
            color=base.MODEL_COLORS[model],
            lw=2.0,
            label=base.MODEL_LABELS[model],
        )

    ax.set_title("QNT_score01: pooled GT/pred quantile curves")
    ax.set_xlabel("Quantile")
    ax.set_ylabel("Pooled quantile value")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=2)

    return save(fig, output_dir, "metric_QNT_overlay_all_models.svg")


def plot_mom(model_arrays: dict[str, base.ModelArrays], output_dir: Path) -> str:
    # Raw quantity: per-region log variance ratio, skew difference, kurtosis
    # difference. The official distance is their weighted sum; lower is better.
    raw_labels = ["var_pred / var_gt", "skew_pred - skew_gt", "kurt_pred - kurt_gt"]
    dist_labels = ["|log var ratio|", "0.5 |skew diff|", "0.25 |kurt diff|"]
    raw_mat = []
    dist_mat = []
    for model in base.MODEL_ORDER:
        c = extract_moments(model_arrays[model].gt, model_arrays[model].pred)["components"]
        raw_mat.append(
            [
                np.nanmean(c[:, 1] / c[:, 0]),
                np.nanmean(c[:, 3] - c[:, 2]),
                np.nanmean(c[:, 5] - c[:, 4]),
            ]
        )
        dist_mat.append(
            [
                np.nanmean(np.abs(np.log(c[:, 1] / c[:, 0]))),
                np.nanmean(0.5 * np.abs(c[:, 3] - c[:, 2])),
                np.nanmean(0.25 * np.abs(c[:, 5] - c[:, 4])),
            ]
        )
    raw_mat = np.asarray(raw_mat, dtype=np.float64)
    dist_mat = np.asarray(dist_mat, dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.5), constrained_layout=True)
    x = np.arange(len(raw_labels))
    width = 0.12
    for idx, model in enumerate(base.MODEL_ORDER):
        axes[0].bar(x + (idx - 2.5) * width, raw_mat[idx], width=width, color=base.MODEL_COLORS[model], label=base.MODEL_LABELS[model])
        axes[1].bar(x + (idx - 2.5) * width, dist_mat[idx], width=width, color=base.MODEL_COLORS[model])
    axes[0].axhline(1.0, color="#777777", lw=0.8, ls="--")
    axes[0].axhline(0.0, color="#BBBBBB", lw=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(raw_labels, rotation=18, ha="right")
    axes[0].set_ylabel("Signed/raw mean across regions")
    axes[0].set_title("GT vs prediction moment direction")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(dist_labels, rotation=18, ha="right")
    axes[1].set_ylabel("Mean weighted component distance")
    axes[1].set_yscale("log")
    axes[1].set_title("Official MOM distance components (lower better)")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("MOM_score01: GT/pred variance, skewness, and kurtosis comparisons", y=1.02)
    return save(fig, output_dir, "metric_MOM_gt_pred_moment_components.svg")


def plot_mean(
    model_arrays: dict[str, base.ModelArrays],
    output_dir: Path,
    official_means: dict[str, dict[str, float]],
) -> str:
    """
    Single-panel version of the mean-shift plot.

    For each region, plot the IQR-scaled absolute mean shift for all models
    as grouped bars in one figure.

    This keeps the spirit of the attached plot, but merges all models into
    a single comparison figure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    n_regions = None

    for model in base.MODEL_ORDER:
        gt, pred = align_for_metric(model_arrays[model].gt, model_arrays[model].pred)
        ext = extract_mean(gt, pred)

        scaled = np.nanmean(ext["scaled_abs_delta"], axis=0)  # mean over sequences, per region

        if n_regions is None:
            n_regions = scaled.shape[0]

        rows.append(
            {
                "model": model,
                "label": base.MODEL_LABELS[model],
                "color": base.MODEL_COLORS[model],
                "scaled": scaled,
                "D": float(ext["D"]),
                "score": official_means[model]["Mean_score01"],
            }
        )

    if not rows:
        raise ValueError("No valid model data available for mean-shift plotting.")

    # Optional: keep original region order.
    region_order = np.arange(n_regions)

    # If you prefer to sort regions by average difficulty across models, use:
    # mean_across_models = np.nanmean(np.stack([r["scaled"] for r in rows], axis=0), axis=0)
    # region_order = np.argsort(-mean_across_models)

    x = np.arange(n_regions)
    n_models = len(rows)

    # Bar width chosen to fit all models per region.
    width = min(0.82 / max(n_models, 1), 0.16)
    offsets = (np.arange(n_models) - (n_models - 1) / 2.0) * width

    ymax = max(float(np.nanmax(r["scaled"][region_order])) for r in rows)
    ymax = max(ymax, 0.05)

    fig, ax = plt.subplots(figsize=(11.2, 4.8), constrained_layout=True)

    for i, row in enumerate(rows):
        vals = row["scaled"][region_order]
        ax.bar(
            x + offsets[i],
            vals,
            width=width * 0.95,
            color=row["color"],
            alpha=0.85,
            label=f"{row['label']} (D={row['D']:.3f}, score={row['score']:.3f})",
        )

    ax.set_title("Mean_score01 raw quantity: IQR-scaled region mean shift")
    ax.set_xlabel("Region")
    ax.set_ylabel("Mean scaled |mean shift|")
    ax.set_xticks(x)
    ax.set_xticklabels(region_order)
    ax.set_ylim(0, ymax * 1.15)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        frameon=False,
        fontsize=8,
        ncol=2,
        loc="upper left",
    )

    return save(fig, output_dir, "metric_Mean_scaled_region_shift_grouped.svg")


def plot_graph(model_arrays: dict[str, base.ModelArrays], output_dir: Path) -> tuple[str, str]:
    """
    Rethought GRAPH visualization.

    Saves only two figures:
    1) a single-row panel of predicted region-by-region correlation matrices
       (one matrix per model, same color scale for all);
    2) a single-panel comparison of the upper-triangular correlation entries,
       ordered by descending absolute GT correlation magnitude.

    The second plot is designed to directly reflect the quantities behind the
    GRAPH score:
    - edge-weight agreement on vec_triangle(C)
    - top-k edge preservation (via the shaded GT top-k zone)

    Returns
    -------
    matrix_file, edge_profile_file : tuple[str, str]
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Extract graph payload once for all models
    # ------------------------------------------------------------------
    graph_data = []
    for model in base.MODEL_ORDER:
        g = extract_graph(model_arrays[model].gt, model_arrays[model].pred)
        graph_data.append((model, g))

    if len(graph_data) == 0:
        raise ValueError("No graph data available.")

    # ------------------------------------------------------------------
    # Common GT reference
    # Assumes all models are compared to the same GT (or effectively the same GT).
    # ------------------------------------------------------------------
    ref_gt = np.array(graph_data[0][1]["corr_gt"], copy=True)
    tri = np.triu_indices_from(ref_gt, k=1)

    gt_vec = ref_gt[tri]
    valid_gt = np.isfinite(gt_vec)
    if not np.any(valid_gt):
        raise ValueError("GT upper-triangle correlation entries are all non-finite.")

    # Sort edges by descending absolute GT correlation
    order = np.argsort(-np.abs(gt_vec))
    gt_ranked = gt_vec[order]

    # Top-k cutoff from the thresholded GT graph used in GRAPH
    # (number of active upper-triangle edges in adj_gt)
    ref_adj_gt = np.array(graph_data[0][1]["adj_gt"], copy=True)
    topk = int(np.sum(np.triu(ref_adj_gt > 0, k=1)))

    # ------------------------------------------------------------------
    # Figure 1: predicted correlation matrices on one line
    # ------------------------------------------------------------------
    n_models = len(graph_data)
    fig, axes = plt.subplots(
        1,
        n_models,
        figsize=(2.25 * n_models, 2.7),
        constrained_layout=True,
    )

    if n_models == 1:
        axes = [axes]

    corr_im = None
    for ax, (model, g) in zip(axes, graph_data):
        mat = np.array(g["corr_pred"], copy=True)
        corr_im = ax.imshow(mat, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_title(base.MODEL_LABELS[model], fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.colorbar(
        corr_im,
        ax=axes,
        fraction=0.020,
        pad=0.02,
        label="Correlation",
    )
    fig.suptitle(
        "GRAPH_score01: predicted region-by-region correlation matrices",
        y=1.03,
    )

    matrix_file = save(
        fig,
        output_dir,
        "metric_GRAPH_prediction_correlation_matrices_row.svg",
    )

    # ------------------------------------------------------------------
    # Figure 2: ranked upper-triangle edge comparison + signed-error heatmap
    # ------------------------------------------------------------------
    def _binned_profile(x, y, n_bins=24):
        """
        Bin ranked edges into contiguous rank bins.
        Returns bin centers and mean y per bin.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]

        if x.size == 0:
            return np.array([]), np.array([])

        bins = np.array_split(np.arange(x.size), min(n_bins, x.size))
        bx = np.array([np.nanmean(x[b]) for b in bins])
        by = np.array([np.nanmean(y[b]) for b in bins])
        return bx, by


    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.2, 6.4),
        gridspec_kw={"height_ratios": [2.2, 1.0]},
        constrained_layout=True,
    )

    ax = axes[0]
    ax_delta = axes[1]

    # x-axis: upper-triangle edge rank, sorted by |GT correlation|
    x = np.arange(len(gt_ranked))

    # ------------------------------------------------------------------
    # Top panel: ranked correlation profiles
    # ------------------------------------------------------------------
    if topk > 0:
        ax.axvspan(
            0,
            topk - 1,
            color="#EFEFEF",
            alpha=0.9,
            zorder=0,
            label=f"Top-k GT edges (k={topk})",
        )

    # GT raw + binned trend
    ax.plot(
        x,
        gt_ranked,
        color="black",
        lw=1.8,
        alpha=0.55,
        label="GT raw",
        zorder=4,
    )

    bx_gt, by_gt = _binned_profile(x, gt_ranked, n_bins=24)
    ax.plot(
        bx_gt,
        by_gt,
        color="black",
        lw=3.0,
        label="GT binned trend",
        zorder=6,
    )

    delta_rows = []
    delta_labels = []

    for model, g in graph_data:
        pred_vec = np.array(g["corr_pred"], copy=True)[tri]
        pred_ranked = pred_vec[order]

        delta_ranked = pred_ranked - gt_ranked
        delta_rows.append(delta_ranked)
        delta_labels.append(base.MODEL_LABELS[model])

        finite = np.isfinite(gt_vec) & np.isfinite(pred_vec)

        if np.sum(finite) >= 2:
            weight_r = float(np.corrcoef(gt_vec[finite], pred_vec[finite])[0, 1])
        else:
            weight_r = np.nan

        # Raw model profile, faint
        ax.plot(
            x,
            pred_ranked,
            color=base.MODEL_COLORS[model],
            lw=1.0,
            alpha=0.25,
            zorder=2,
        )

        # Binned model trend, emphasized
        bx, by = _binned_profile(x, pred_ranked, n_bins=24)

        label = base.MODEL_LABELS[model]
        if np.isfinite(weight_r):
            label = f"{label} upper-tri r={weight_r:.2f}"

        # If extract_graph stores GRAPH score or sub-scores, add them here.
        # For example:
        # if "score" in g:
        #     label = f"{label}; GRAPH={g['score']:.2f}"
        # if "jaccard" in g:
        #     label = f"{label}; J={g['jaccard']:.2f}"

        ax.plot(
            bx,
            by,
            color=base.MODEL_COLORS[model],
            lw=2.3,
            alpha=0.95,
            label=label,
            zorder=5,
        )

    # Reference lines

    if topk > 0:
        ax.axvline(topk - 0.5, color="#999999", lw=1.0, ls=":", zorder=3)

    ax.set_ylabel("Correlation value")
    ax.set_title("GRAPH_score01: upper-triangle correlation profiles in GT-defined order")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, ncol=2)

    edge_profile_file = save(
        fig,
        output_dir,
        "metric_GRAPH_ranked_upper_triangle_profiles_with_delta_heatmap.svg",
    )

    return matrix_file, edge_profile_file


def _binned_profile_from_ranked(y: np.ndarray, n_bins: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """
    Bin a ranked 1D profile into contiguous rank bins and return
    bin centers + mean value per bin.
    """
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(y)
    y = y[valid]

    if y.size == 0:
        return np.array([]), np.array([])

    idx = np.arange(y.size)
    bins = np.array_split(idx, min(n_bins, y.size))
    x_bin = np.array([np.mean(b) for b in bins], dtype=np.float64)
    y_bin = np.array([np.nanmean(y[b]) for b in bins], dtype=np.float64)
    return x_bin, y_bin


def _vector_corr(a: np.ndarray, b: np.ndarray) -> float:
    """
    Pearson correlation between two 1D arrays, ignoring non-finite values.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if np.sum(valid) < 2:
        return np.nan
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def plot_ranked_upper_triangle_profiles(
    model_arrays: dict[str, base.ModelArrays],
    output_dir: Path,
    filename: str,
    title: str,
    getter,
    value_label: str,
    *,
    sort_by_abs_gt: bool = False,
    top_frac_visual: float | None = 0.20,
    n_bins: int = 24,
) -> str:
    """
    Generic plot for metrics that compare upper-triangular matrix entries.

    The figure has two panels:
    1) ranked GT/pred upper-triangle profiles (GT-defined order)
    2) signed error heatmap (pred - GT)

    Parameters
    ----------
    getter
        Function (gt, pred) -> (matrix_gt, matrix_pred)
    value_label
        Label for the top-panel y axis and heatmap colorbar
    sort_by_abs_gt
        If True, sort by descending |GT entry|. For MI this should be False.
    top_frac_visual
        Optional visual guide: lightly shade the strongest GT edges.
        This is NOT part of the CrossMI score; it is only a visual aid.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = []
    ref_gt = None

    # --------------------------------------------------------------
    # Extract upper-triangle vectors
    # --------------------------------------------------------------
    for model in base.MODEL_ORDER:
        mat_gt, mat_pred = getter(model_arrays[model].gt, model_arrays[model].pred)

        mat_gt = np.asarray(mat_gt, dtype=np.float64)
        mat_pred = np.asarray(mat_pred, dtype=np.float64)

        if ref_gt is None:
            ref_gt = mat_gt.copy()

        tri = np.triu_indices_from(mat_gt, k=1)
        gt_vec = mat_gt[tri]
        pred_vec = mat_pred[tri]

        r = _vector_corr(gt_vec, pred_vec)
        mad = float(np.nanmean(np.abs(pred_vec - gt_vec)))

        payload.append(
            {
                "model": model,
                "label": base.MODEL_LABELS[model],
                "color": base.MODEL_COLORS[model],
                "gt_vec": gt_vec,
                "pred_vec": pred_vec,
                "r": r,
                "mad": mad,
            }
        )

    if not payload:
        raise ValueError("No valid matrix data available.")

    # --------------------------------------------------------------
    # Define common GT edge ordering
    # --------------------------------------------------------------
    gt_ref_vec = payload[0]["gt_vec"]

    if sort_by_abs_gt:
        order = np.argsort(-np.abs(gt_ref_vec))
    else:
        order = np.argsort(-gt_ref_vec)

    gt_ranked = gt_ref_vec[order]
    n_edges = len(gt_ranked)
    x = np.arange(n_edges)

    # Visual guide only: strongest GT edges
    if top_frac_visual is not None:
        n_top = max(1, int(np.ceil(top_frac_visual * n_edges)))
    else:
        n_top = 0

    # --------------------------------------------------------------
    # Prepare error matrix for heatmap
    # --------------------------------------------------------------
    delta_rows = []
    delta_labels = []
    pooled_values = [gt_ranked]

    for row in payload:
        pred_ranked = row["pred_vec"][order]
        row["pred_ranked"] = pred_ranked
        row["delta_ranked"] = pred_ranked - gt_ranked
        delta_rows.append(row["delta_ranked"])
        delta_labels.append(
            f"{row['label']} (r={row['r']:.2f}, MA|Δ|={row['mad']:.3g})"
            if np.isfinite(row["r"])
            else f"{row['label']} (MA|Δ|={row['mad']:.3g})"
        )
        pooled_values.append(pred_ranked)

    delta_mat = np.asarray(delta_rows, dtype=np.float64)
    pooled_values = np.concatenate([v[np.isfinite(v)] for v in pooled_values if np.any(np.isfinite(v))])

    delta_lim = max(
        1e-6,
        float(np.nanpercentile(np.abs(delta_mat), 98)),
    )
    y_max = max(
        1e-6,
        float(np.nanpercentile(pooled_values, 99.5)),
    )

    # --------------------------------------------------------------
    # Plot
    # --------------------------------------------------------------
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.4, 6.6),
        gridspec_kw={"height_ratios": [2.2, 1.15]},
        constrained_layout=True,
    )

    ax_top, ax_bot = axes

    # -----------------------------
    # Top panel: ranked profiles
    # -----------------------------
    if n_top > 0:
        ax_top.axvspan(
            -0.5,
            n_top - 0.5,
            color="#EFEFEF",
            alpha=0.9,
            zorder=0,
            label=f"Top {int(round(100 * top_frac_visual))}% GT MI edges",
        )

    # GT: faint raw line + strong binned trend
    ax_top.plot(
        x,
        gt_ranked,
        color="black",
        lw=1.6,
        alpha=0.35,
        zorder=3,
    )
    bx_gt, by_gt = _binned_profile_from_ranked(gt_ranked, n_bins=n_bins)
    ax_top.plot(
        bx_gt,
        by_gt,
        color="black",
        lw=3.0,
        label="GT",
        zorder=6,
    )

    # Model profiles
    for row in payload:
        pred_ranked = row["pred_ranked"]

        # faint raw profile
        ax_top.plot(
            x,
            pred_ranked,
            color=row["color"],
            lw=1.0,
            alpha=0.22,
            zorder=1,
        )

        # emphasized binned profile
        bx, by = _binned_profile_from_ranked(pred_ranked, n_bins=n_bins)
        label = (
            f"{row['label']} (r={row['r']:.2f})"
            if np.isfinite(row["r"])
            else row["label"]
        )
        ax_top.plot(
            bx,
            by,
            color=row["color"],
            lw=2.2,
            alpha=0.98,
            label=label,
            zorder=5,
        )

    if n_top > 0:
        ax_top.axvline(n_top - 0.5, color="#999999", lw=1.0, ls=":")

    ax_top.set_xlim(-0.5, n_edges - 0.5)
    ax_top.set_ylim(0.0, y_max * 1.05)
    ax_top.set_ylabel(value_label)
    ax_top.set_title(title)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")

    # -----------------------------
    # Bottom panel: signed error heatmap
    # -----------------------------
    im = ax_bot.imshow(
        delta_mat,
        aspect="auto",
        cmap="coolwarm",
        vmin=-delta_lim,
        vmax=delta_lim,
        interpolation="nearest",
    )

    if n_top > 0:
        ax_bot.axvline(n_top - 0.5, color="black", lw=0.8, ls=":")

    ax_bot.set_yticks(np.arange(len(delta_labels)))
    ax_bot.set_yticklabels(delta_labels, fontsize=8)
    ax_bot.set_xlabel("Upper-triangle edge rank (sorted by GT mutual information)")
    ax_bot.set_ylabel("Model")
    ax_bot.set_title("Signed upper-triangle error: prediction - GT", fontsize=9)
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)

    cbar = fig.colorbar(
        im,
        ax=ax_bot,
        fraction=0.035,
        pad=0.02,
    )
    cbar.set_label(f"{value_label} pred - GT")

    return save(fig, output_dir, filename)


def plot_cross_mi(model_arrays: dict[str, base.ModelArrays], output_dir: Path) -> str:
    """
    CrossRegionMI visualization based directly on the score definition.

    Instead of showing GT/pred/delta matrices for each model, this plot compares
    the upper-triangular mutual-information entries in GT-defined rank order,
    which is exactly the quantity the score uses.

    Top panel:
        GT MI profile + all model prediction profiles
    Bottom panel:
        signed error heatmap (MI_pred - MI_GT)
    """
    return plot_ranked_upper_triangle_profiles(
        model_arrays,
        output_dir,
        "metric_CrossRegionMI_ranked_upper_triangle_profiles.svg",
        "CrossRegionMI_score01: upper-triangle mutual-information profiles in GT-defined order",
        lambda gt, pred: (
            extract_additional(gt, pred)["mi_gt"],
            extract_additional(gt, pred)["mi_pred"],
        ),
        "Mutual information",
        sort_by_abs_gt=False,   # MI is nonnegative, so sort by GT MI directly
        top_frac_visual=0.20,   # visual aid only, not part of the score
        n_bins=24,
    )


def plot_lagged_cov(model_arrays: dict[str, base.ModelArrays], output_dir: Path) -> tuple[str, str]:
    # Raw quantity: lagged covariance matrices at official lags 1, 2, and 4
    # on standardized pooled rows. Smaller GT/pred matrix differences are better.
    rows = []
    lag_payload = {}
    for model in base.MODEL_ORDER:
        ext = extract_additional(model_arrays[model].gt, model_arrays[model].pred)
        lag_payload[model] = ext["lagged_cov"]
        rows.append([np.nanmean(np.abs(b - a)) for a, b in ext["lagged_cov"].values()])
    fig, ax = plt.subplots(figsize=(7.2, 4.1), constrained_layout=True)
    mat = np.asarray(rows, dtype=np.float64)
    shown = np.log1p(mat)
    im = ax.imshow(shown, cmap="magma", aspect="auto")
    ax.set_yticks(np.arange(len(base.MODEL_ORDER)))
    ax.set_yticklabels([base.MODEL_LABELS[m] for m in base.MODEL_ORDER])
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["lag 1", "lag 2", "lag 4"])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.3g}", ha="center", va="center", fontsize=7, color="white")
    ax.set_title("LaggedCovariance_score01 summary: mean |lagged covariance delta|")
    fig.colorbar(im, ax=ax, label="log1p(mean absolute matrix delta)")
    summary_file = save(fig, output_dir, "metric_LaggedCovariance_summary_by_lag.svg")

    matrices = []
    for model in base.MODEL_ORDER:
        gt_lag, pred_lag = lag_payload[model][1]
        mad = float(np.nanmean(np.abs(pred_lag - gt_lag)))
        corr = matrix_corr(gt_lag, pred_lag)
        matrices.append((base.MODEL_LABELS[model], gt_lag, pred_lag, f"lag=1; MA|delta|={mad:.3g}; r={corr:.2f}"))
    triptych_file = plot_matrix_triptych(
        matrices,
        output_dir,
        "metric_LaggedCovariance_gt_pred_delta_lag1.svg",
        "LaggedCovariance_score01: GT/pred lag-1 covariance matrices and signed delta",
        gt_pred_cmap="coolwarm",
        gt_pred_diverging=True,
        delta_label="Lagged covariance pred - GT",
    )
    return summary_file, triptych_file


def _binned_profile_from_ranked(y: np.ndarray, n_bins: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """
    Bin a ranked 1D profile into contiguous bins and return
    bin centers + mean value per bin.
    """
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(y)
    y = y[valid]

    if y.size == 0:
        return np.array([]), np.array([])

    idx = np.arange(y.size)
    bins = np.array_split(idx, min(n_bins, y.size))
    x_bin = np.array([np.mean(b) for b in bins], dtype=np.float64)
    y_bin = np.array([np.nanmean(y[b]) for b in bins], dtype=np.float64)
    return x_bin, y_bin


def _vector_corr(a: np.ndarray, b: np.ndarray) -> float:
    """
    Pearson correlation between two 1D arrays, ignoring non-finite values.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    valid = np.isfinite(a) & np.isfinite(b)
    if np.sum(valid) < 2:
        return np.nan
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def plot_ranked_operator_profiles(
    model_arrays: dict[str, base.ModelArrays],
    output_dir: Path,
    filename: str,
    title: str,
    getter,
    value_label: str,
    *,
    n_bins: int = 24,
    top_frac_visual: float | None = 0.15,
) -> str:
    """
    Generic visualization for metrics based on comparing a full operator/matrix
    through its vectorized entries.

    Produces a two-panel figure:
    1) ranked GT/pred coefficient profiles in GT-defined order
    2) signed error heatmap (pred - GT)

    The coefficients are ranked by descending absolute GT value so the strongest
    linear influences appear first.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = []

    # --------------------------------------------------------------
    # Extract and flatten matrices
    # --------------------------------------------------------------
    for model in base.MODEL_ORDER:
        A_gt, A_pred = getter(model_arrays[model].gt, model_arrays[model].pred)

        A_gt = np.asarray(A_gt, dtype=np.float64)
        A_pred = np.asarray(A_pred, dtype=np.float64)

        gt_vec = A_gt.ravel()
        pred_vec = A_pred.ravel()

        r = _vector_corr(gt_vec, pred_vec)
        mad = float(np.nanmean(np.abs(pred_vec - gt_vec)))
        fro = float(np.linalg.norm(A_pred - A_gt))

        payload.append(
            {
                "model": model,
                "label": base.MODEL_LABELS[model],
                "color": base.MODEL_COLORS[model],
                "gt_vec": gt_vec,
                "pred_vec": pred_vec,
                "r": r,
                "mad": mad,
                "fro": fro,
            }
        )

    if not payload:
        raise ValueError("No valid operator data available.")

    # --------------------------------------------------------------
    # Define common GT coefficient ordering
    # --------------------------------------------------------------
    gt_ref_vec = payload[0]["gt_vec"]
    order = np.argsort(-np.abs(gt_ref_vec))  # strongest GT coefficients first
    gt_ranked = gt_ref_vec[order]

    n_coeff = len(gt_ranked)
    x = np.arange(n_coeff)

    if top_frac_visual is not None:
        n_top = max(1, int(np.ceil(top_frac_visual * n_coeff)))
    else:
        n_top = 0

    # --------------------------------------------------------------
    # Prepare signed-error heatmap
    # --------------------------------------------------------------
    delta_rows = []
    delta_labels = []
    pooled_values = [gt_ranked]

    for row in payload:
        pred_ranked = row["pred_vec"][order]
        row["pred_ranked"] = pred_ranked
        row["delta_ranked"] = pred_ranked - gt_ranked

        delta_rows.append(row["delta_ranked"])
        delta_labels.append(
            f"{row['label']} (r={row['r']:.2f}, MA|Δ|={row['mad']:.3g})"
            if np.isfinite(row["r"])
            else f"{row['label']} (MA|Δ|={row['mad']:.3g})"
        )

        pooled_values.append(pred_ranked)

    delta_mat = np.asarray(delta_rows, dtype=np.float64)

    coeff_lim = max(
        1e-6,
        float(np.nanpercentile(np.abs(np.concatenate([v[np.isfinite(v)] for v in pooled_values])), 99.0)),
    )
    delta_lim = max(
        1e-6,
        float(np.nanpercentile(np.abs(delta_mat), 98)),
    )

    # --------------------------------------------------------------
    # Plot
    # --------------------------------------------------------------
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.6, 6.6),
        gridspec_kw={"height_ratios": [2.25, 1.15]},
        constrained_layout=True,
    )

    ax_top, ax_bot = axes

    # -----------------------------
    # Top panel: ranked coefficient profiles
    # -----------------------------
    if n_top > 0:
        ax_top.axvspan(
            -0.5,
            n_top - 0.5,
            color="#EFEFEF",
            alpha=0.9,
            zorder=0,
            label=f"Top {int(round(100 * top_frac_visual))}% |GT| coefficients",
        )

    # GT: faint raw + stronger binned trend
    ax_top.plot(
        x,
        gt_ranked,
        color="black",
        lw=1.5,
        alpha=0.40,
        zorder=3,
    )
    bx_gt, by_gt = _binned_profile_from_ranked(gt_ranked, n_bins=n_bins)
    ax_top.plot(
        bx_gt,
        by_gt,
        color="black",
        lw=3.0,
        label="GT",
        zorder=6,
    )

    for row in payload:
        pred_ranked = row["pred_ranked"]

        # Raw profile
        ax_top.plot(
            x,
            pred_ranked,
            color=row["color"],
            lw=1.0,
            alpha=0.20,
            zorder=1,
        )

        # Binned trend
        bx, by = _binned_profile_from_ranked(pred_ranked, n_bins=n_bins)
        label = (
            f"{row['label']} (r={row['r']:.2f})"
            if np.isfinite(row["r"])
            else row["label"]
        )
        ax_top.plot(
            bx,
            by,
            color=row["color"],
            lw=2.2,
            alpha=0.98,
            label=label,
            zorder=5,
        )

    ax_top.axhline(0.0, color="#666666", lw=0.9, ls="--")
    if n_top > 0:
        ax_top.axvline(n_top - 0.5, color="#999999", lw=1.0, ls=":")

    ax_top.set_xlim(-0.5, n_coeff - 0.5)
    ax_top.set_ylim(-coeff_lim * 1.05, coeff_lim * 1.05)
    ax_top.set_ylabel(value_label)
    ax_top.set_title(title)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")

    # -----------------------------
    # Bottom panel: signed error heatmap
    # -----------------------------
    im = ax_bot.imshow(
        delta_mat,
        aspect="auto",
        cmap="coolwarm",
        vmin=-delta_lim,
        vmax=delta_lim,
        interpolation="nearest",
    )

    if n_top > 0:
        ax_bot.axvline(n_top - 0.5, color="black", lw=0.8, ls=":")

    ax_bot.set_yticks(np.arange(len(delta_labels)))
    ax_bot.set_yticklabels(delta_labels, fontsize=8)
    ax_bot.set_xlabel("Coefficient rank (sorted by |GT VAR(1) coefficient|)")
    ax_bot.set_ylabel("Model")
    ax_bot.set_title("Signed operator error: prediction - GT", fontsize=9)
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)

    cbar = fig.colorbar(
        im,
        ax=ax_bot,
        fraction=0.035,
        pad=0.02,
    )
    cbar.set_label(f"{value_label} pred - GT")

    return save(fig, output_dir, filename)


def plot_impulse_response(model_arrays: dict[str, base.ModelArrays], output_dir: Path) -> str:
    """
    Improved visualization for the ImpulseResponse / VAR(1) score.

    Instead of showing GT/pred/delta matrices for each model, this function
    visualizes the vectorized VAR(1) operators directly, which is exactly what
    the score compares.

    Top panel:
        ranked GT/pred operator coefficients in GT-defined order
    Bottom panel:
        signed coefficient error heatmap (A_pred - A_GT)
    """
    return plot_ranked_operator_profiles(
        model_arrays,
        output_dir,
        "metric_ImpulseResponse_ranked_var1_operator_profiles.svg",
        "ImpulseResponse_score01: ranked VAR(1) operator coefficients in GT-defined order",
        lambda gt, pred: (
            extract_additional(gt, pred)["var1_gt"],
            extract_additional(gt, pred)["var1_pred"],
        ),
        "VAR(1) coefficient",
        n_bins=24,
        top_frac_visual=0.15,  # visual aid only
    )


def plot_subspace_angles(
    model_arrays: dict[str, base.ModelArrays],
    output_dir: Path,
    official_means: dict[str, dict[str, float]],
) -> str:
    """
    Improved visualization for SubspaceAngle_score01.

    Panel A:
        Horizontal bar chart of the official subspace score
        S = mean_k cos^2(theta_k), higher is better.

    Panel B:
        Heatmap of per-angle alignment cos^2(theta_k), with GT explained
        variance ratio shown in the x-axis labels for context.

    This is more directly tied to the metric definition than plotting raw
    angles + a separate variance-context line plot.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    gt_var_by_key = {}

    # ------------------------------------------------------------------
    # Extract quantities for each model
    # ------------------------------------------------------------------
    for model in base.MODEL_ORDER:
        ext = extract_additional(model_arrays[model].gt, model_arrays[model].pred)

        angles = np.asarray(ext["angles"], dtype=np.float64)
        deg = np.degrees(angles)
        cos2 = np.cos(angles) ** 2
        # The heatmap remains a pooled diagnostic, but its displayed scalar
        # must be the four-split, per-training-seed aggregate used in the
        # official tables.  Averaging this nonlinear metric after pooling all
        # seeds produces a different number.
        score = official_means[model]["SubspaceAngle_score01"]

        # GT variance context
        gt_flat, pred_flat = standardized_flat(model_arrays[model].gt, model_arrays[model].pred)
        gt_cov, _ = addm._ledoit_cov_precision(gt_flat)

        eigvals, _ = np.linalg.eigh(gt_cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[order], 0.0)
        gt_var_ratio = eigvals / max(float(np.sum(eigvals)), EPS)

        gt_key = getattr(model_arrays[model], "gt_key", model)
        if gt_key not in gt_var_by_key:
            gt_var_by_key[gt_key] = gt_var_ratio[: len(cos2)]

        rows.append(
            {
                "model": model,
                "label": base.MODEL_LABELS[model],
                "color": base.MODEL_COLORS[model],
                "angles_deg": deg,
                "cos2": cos2,
                "score": score,
            }
        )

    if not rows:
        raise ValueError("No subspace-angle data available.")

    # ------------------------------------------------------------------
    # Common GT variance context
    # ------------------------------------------------------------------
    K = max(len(r["cos2"]) for r in rows)
    gt_var_stack = []
    for v in gt_var_by_key.values():
        vv = np.asarray(v, dtype=np.float64)
        if len(vv) < K:
            tmp = np.full(K, np.nan, dtype=np.float64)
            tmp[: len(vv)] = vv
            vv = tmp
        gt_var_stack.append(vv)

    common_gt_var = np.nanmean(np.stack(gt_var_stack, axis=0), axis=0)

    # ------------------------------------------------------------------
    # Sort models by official score (best first)
    # ------------------------------------------------------------------
    rows = sorted(rows, key=lambda r: r["score"], reverse=True)

    # ------------------------------------------------------------------
    # Build heatmap matrix
    # ------------------------------------------------------------------
    heat = np.full((len(rows), K), np.nan, dtype=np.float64)
    angle_deg_mat = np.full((len(rows), K), np.nan, dtype=np.float64)

    for i, r in enumerate(rows):
        heat[i, : len(r["cos2"])] = r["cos2"]
        angle_deg_mat[i, : len(r["angles_deg"])] = r["angles_deg"]

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.0, 4.8),
        gridspec_kw={"width_ratios": [1.0, 1.8]},
        constrained_layout=True,
    )

    ax_bar, ax_heat = axes

    # ------------------------------------------------------------------
    # Panel 1: official score
    # ------------------------------------------------------------------
    y = np.arange(len(rows))
    scores = [r["score"] for r in rows]
    colors = [r["color"] for r in rows]
    labels = [r["label"] for r in rows]

    ax_bar.barh(y, scores, color=colors, alpha=0.9)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(labels)
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, 1.0)
    ax_bar.set_xlabel(r"Mean $\cos^2(\theta_k)$")
    ax_bar.set_title("Official subspace score")

    for yi, s in zip(y, scores):
        ax_bar.text(
            min(s + 0.02, 0.98),
            yi,
            f"{s:.2f}",
            va="center",
            ha="left",
            fontsize=8,
        )

    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    # ------------------------------------------------------------------
    # Panel 2: per-angle alignment heatmap
    # ------------------------------------------------------------------
    im = ax_heat.imshow(
        heat,
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )

    ax_heat.set_yticks(np.arange(len(rows)))
    ax_heat.set_yticklabels([f"{r['label']} ({r['score']:.2f})" for r in rows])

    xticks = np.arange(K)
    xticklabels = []
    for k in range(K):
        if np.isfinite(common_gt_var[k]):
            xticklabels.append(f"PC{k+1}\n{100 * common_gt_var[k]:.1f}%")
        else:
            xticklabels.append(f"PC{k+1}")

    ax_heat.set_xticks(xticks)
    ax_heat.set_xticklabels(xticklabels)
    ax_heat.set_xlabel("Principal-angle index (GT explained variance)")
    ax_heat.set_title(r"Per-dimension alignment: $\cos^2(\theta_k)$")

    # Optional annotation if K is small
    if K <= 6:
        for i in range(heat.shape[0]):
            for j in range(heat.shape[1]):
                if np.isfinite(heat[i, j]):
                    txt_color = "white" if heat[i, j] < 0.45 else "black"
                    ax_heat.text(
                        j,
                        i,
                        f"{heat[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=txt_color,
                    )

    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.03)
    cbar.set_label(r"$\cos^2(\theta_k)$ (higher = better aligned)")

    fig.suptitle(
        "SubspaceAngle_score01: official score and per-angle subspace alignment",
        y=1.02,
    )

    return save(
        fig,
        output_dir,
        "metric_SubspaceAngle_score_and_cos2_heatmap.svg",
    )


def plot_trajectory(
    model_arrays: dict[str, base.ModelArrays],
    output_dir: Path,
) -> str:
    """
    Single-panel comparison of speed distributions for all models against a
    common ground truth.

    Visualization choice:
    - Uses ECDF curves instead of histograms, because overlaid histograms for
      many models are cluttered and bin-dependent.
    - Speed is normalized by the common GT median speed, so x=1 corresponds
      to the GT median speed.

    Assumption:
    - All models are evaluated against the same GT (or effectively the same GT).
      If multiple GTs are present, the function uses the first one as reference
      and still plots each model's prediction curve.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    def _ecdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return np.array([]), np.array([])
        x = np.sort(x)
        y = np.arange(1, x.size + 1, dtype=np.float64) / x.size
        return x, y

    # ------------------------------------------------------------------
    # Extract speed distributions
    # ------------------------------------------------------------------
    speed_payload = {}
    common_gt_speed = None
    common_gt_median = None

    for i, model in enumerate(base.MODEL_ORDER):
        ext = extract_trajectory(model_arrays[model].gt, model_arrays[model].pred)

        speed_g = np.asarray(ext["speed_g"], dtype=np.float64)
        speed_p = np.asarray(ext["speed_p"], dtype=np.float64)

        speed_g = speed_g[np.isfinite(speed_g)]
        speed_p = speed_p[np.isfinite(speed_p)]

        if speed_g.size == 0 or speed_p.size == 0:
            continue

        # Use the first GT as the common reference
        if common_gt_speed is None:
            common_gt_speed = speed_g.copy()
            common_gt_median = max(float(np.nanmedian(common_gt_speed)), EPS)

        # Normalize everything by the common GT median
        speed_payload[model] = {
            "pred_speed_norm": speed_p / common_gt_median,
        }

    if common_gt_speed is None:
        raise ValueError("No valid GT speed values found.")

    gt_speed_norm = common_gt_speed / common_gt_median

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.8, 5.2), constrained_layout=True)

    # Common GT ECDF
    x_gt, y_gt = _ecdf(gt_speed_norm)
    ax.plot(
        x_gt,
        y_gt,
        color="black",
        lw=2.4,
        label="GT",
        zorder=5,
    )

    # Predicted ECDF for each model
    for model in base.MODEL_ORDER:
        if model not in speed_payload:
            continue

        x_pred, y_pred = _ecdf(speed_payload[model]["pred_speed_norm"])
        ax.plot(
            x_pred,
            y_pred,
            color=base.MODEL_COLORS[model],
            lw=2.0,
            alpha=0.95,
            label=base.MODEL_LABELS[model],
        )

    # Reference line at GT median
    ax.axvline(1.0, color="#666666", lw=1.0, ls="--", alpha=0.9)

    # Set a sensible x-limit from the pooled distributions
    all_x = [gt_speed_norm]
    for model in base.MODEL_ORDER:
        if model in speed_payload:
            all_x.append(speed_payload[model]["pred_speed_norm"])
    all_x = np.concatenate(all_x)
    all_x = all_x[np.isfinite(all_x)]

    xmax = float(np.nanpercentile(all_x, 99.5))
    xmax = max(xmax, 1.5)
    ax.set_xlim(0, xmax)

    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Speed / common GT median speed")
    ax.set_ylabel("Empirical cumulative probability")
    ax.set_title("TRJDIST: speed distribution comparison across models")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=2)

    return save(fig, output_dir, "metric_TRJDIST_speed_overlay_ecdf.svg")


def plot_mani(model_arrays: dict[str, base.ModelArrays], output_dir: Path) -> tuple[str, bool, str]:
    # Raw quantity: PH H0/H1 lifetime distributions and kNN distance profiles
    # in the official sampled GT-PCA latent clouds. Similar distributions are better.
    has_ph = smc.ripser is not None
    payload = {model: extract_mani(model_arrays[model].gt, model_arrays[model].pred) for model in base.MODEL_ORDER}

    lifetime_files: list[str] = []
    for dim, label in enumerate(("H0", "H1")):
        fig, axes = panel_grid(len(base.MODEL_ORDER))
        for ax, model in zip(axes, base.MODEL_ORDER):
            ext = payload[model]
            if has_ph:
                gt_life = ext["lifetimes_gt"][dim]
                pred_life = ext["lifetimes_pred"][dim]
                if gt_life.size and pred_life.size:
                    hi = max(np.nanpercentile(gt_life, 95), np.nanpercentile(pred_life, 95), EPS)
                    bins = np.linspace(0, hi, 14)
                    ax.hist(
                        gt_life,
                        bins=bins,
                        alpha=0.45,
                        density=True,
                        color="#1f77b4",
                        label="GT",
                    )
                    ax.hist(
                        pred_life,
                        bins=bins,
                        alpha=0.55,
                        density=True,
                        color="#ff7f0e",
                        label="Pred",
                    )
                else:
                    ax.text(0.5, 0.5, "No finite lifetimes", ha="center", va="center", transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, "ripser unavailable", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(base.MODEL_LABELS[model], fontsize=9)
            ax.set_xlabel("Lifetime")
            ax.set_ylabel("Density")
        axes[0].legend(frameon=False, fontsize=8)
        finalize_unused_axes(axes, len(base.MODEL_ORDER))
        filename = (
            "metric_MANI_ph_lifetimes_gt_pred.svg"
            if dim == 0
            else "metric_MANI_ph_H1_lifetimes_gt_pred.svg"
        )
        lifetime_files.append(save(fig, output_dir, filename))

    fig, axes = panel_grid(len(base.MODEL_ORDER))
    for ax, model in zip(axes, base.MODEL_ORDER):
        ext = extract_mani(model_arrays[model].gt, model_arrays[model].pred)
        ax.plot(np.sort(ext["knn_gt"]), label="GT kNN", color="black")
        ax.plot(np.sort(ext["knn_pred"]), label="Pred kNN", color=base.MODEL_COLORS[model])
        ax.set_title(base.MODEL_LABELS[model])
        ax.set_xlabel("Sorted sampled latent point")
        ax.set_ylabel("5-NN distance")
    axes[0].legend(frameon=False, fontsize=8)
    finalize_unused_axes(axes, len(base.MODEL_ORDER))
    fig.suptitle("MANI_score01 local term: GT vs prediction kNN distance profile", y=1.02)
    knn_file = save(fig, output_dir, "metric_MANI_knn_profile_gt_pred.svg")
    notes = (
        "Exact PH lifetime and kNN objects were extracted with official sampling; plot files show H0/H1 lifetimes and kNN profiles separately."
        if has_ph
        else "ripser is unavailable; PH panels are unavailable and the kNN profile file shows the exact official local geometry object."
    )
    return f"{'; '.join(lifetime_files)}; {knn_file}", has_ph, notes


def plot_state_occupancy(
    model_arrays: dict[str, base.ModelArrays],
    output_dir: Path,
    k: int,
) -> str:
    """
    Single-figure visualization of LatentStateOccupancyK.

    The plot has two panels:
    1. Common GT occupancy histogram, sorted by descending GT occupancy.
    2. Signed model error heatmap: Pred occupancy - GT occupancy.

    This avoids drawing lines between K-means states, whose labels are categorical
    rather than ordered.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Extract occupancies
    # ------------------------------------------------------------------
    payload = []
    gt_by_key = {}

    for model in base.MODEL_ORDER:
        ref = extract_additional(
            model_arrays[model].gt,
            model_arrays[model].pred,
        )["state_ref"]

        occ = state_bundle_from_ref(ref, k=k)

        gt = np.asarray(occ["gt"], dtype=np.float64)
        pred = np.asarray(occ["pred"], dtype=np.float64)

        gt = gt / max(np.nansum(gt), EPS)
        pred = pred / max(np.nansum(pred), EPS)

        payload.append((model, gt, pred))

        gt_key = getattr(model_arrays[model], "gt_key", model)
        if gt_key not in gt_by_key:
            gt_by_key[gt_key] = gt

    if len(payload) == 0:
        raise ValueError("No state-occupancy data available.")

    # ------------------------------------------------------------------
    # Common GT occupancy
    # ------------------------------------------------------------------
    common_gt = np.nanmean(np.stack(list(gt_by_key.values()), axis=0), axis=0)
    common_gt = common_gt / max(np.nansum(common_gt), EPS)

    # Sort categorical states by GT occupancy, not by arbitrary K-means label.
    order = np.argsort(-common_gt)
    common_gt_sorted = common_gt[order]

    # ------------------------------------------------------------------
    # Build signed error matrix: Pred - GT
    # ------------------------------------------------------------------
    error_rows = []
    y_labels = []

    for model, _, pred in payload:
        pred_sorted = pred[order]
        err = pred_sorted - common_gt_sorted
        error_rows.append(err)

        tv = total_variation(common_gt_sorted, pred_sorted)
        h_delta = entropy_of_prob(pred_sorted) - entropy_of_prob(common_gt_sorted)

        y_labels.append(
            f"{base.MODEL_LABELS[model]}  TV={tv:.2f}, ΔH={h_delta:+.2f}"
        )

    error_mat = np.asarray(error_rows, dtype=np.float64)

    delta_lim = max(
        0.05,
        float(np.nanpercentile(np.abs(error_mat), 98)),
    )

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.2, 5.8),
        gridspec_kw={"height_ratios": [1.1, 2.2]},
        constrained_layout=True,
    )

    ax_gt, ax_err = axes

    x = np.arange(k)

    # ------------------------------------------------------------------
    # Panel 1: common GT occupancy
    # ------------------------------------------------------------------
    ax_gt.bar(
        x,
        common_gt_sorted,
        color="#BDBDBD",
        edgecolor="none",
        alpha=0.85,
    )

    ax_gt.set_title(
        f"LatentStateOccupancyK{k}_score01: common GT occupancy"
    )
    ax_gt.set_ylabel("GT occupancy")
    ax_gt.set_xticks(x)
    ax_gt.set_xticklabels([str(i) for i in order])
    ax_gt.set_xlabel("GT-defined state, sorted by GT occupancy")
    ax_gt.set_ylim(0, max(float(np.nanmax(common_gt_sorted)) * 1.25, 0.05))

    ax_gt.spines["top"].set_visible(False)
    ax_gt.spines["right"].set_visible(False)

    # ------------------------------------------------------------------
    # Panel 2: model signed errors
    # ------------------------------------------------------------------
    im = ax_err.imshow(
        error_mat,
        aspect="auto",
        cmap="coolwarm",
        vmin=-delta_lim,
        vmax=delta_lim,
        interpolation="nearest",
    )

    ax_err.axvline(-0.5, color="black", lw=0.8)
    for j in range(k + 1):
        ax_err.axvline(j - 0.5, color="white", lw=0.5, alpha=0.5)

    ax_err.set_yticks(np.arange(len(y_labels)))
    ax_err.set_yticklabels(y_labels, fontsize=8)

    ax_err.set_xticks(x)
    ax_err.set_xticklabels([str(i) for i in order])
    ax_err.set_xlabel("GT-defined state, sorted by GT occupancy")
    ax_err.set_ylabel("Model")
    ax_err.set_title("Signed occupancy error: prediction - GT")

    ax_err.spines["top"].set_visible(False)
    ax_err.spines["right"].set_visible(False)

    cbar = fig.colorbar(
        im,
        ax=ax_err,
        fraction=0.035,
        pad=0.02,
    )
    cbar.set_label("Pred occupancy - GT occupancy")

    return save(
        fig,
        output_dir,
        f"metric_LatentStateOccupancyK{k}_occupancy_common_gt_error_heatmap.svg",
    )

def plot_state_transition(model_arrays: dict[str, base.ModelArrays], output_dir: Path, lag: int) -> str:
    # Raw quantity: GT-defined K=11 state transition probability matrix at the
    # official lag. Histogram overlap/correlation/RMSE of flattened matrices drive the score.
    matrices = []
    for model in base.MODEL_ORDER:
        ref = extract_additional(model_arrays[model].gt, model_arrays[model].pred)["state_ref"]
        tr = state_bundle_from_ref(ref, k=11, lag=lag)
        tv = total_variation(tr["gt"], tr["pred"])
        mad = float(np.nanmean(np.abs(tr["pred"] - tr["gt"])))
        gt_self = float(np.trace(tr["gt"]))
        pred_self = float(np.trace(tr["pred"]))
        summary = f"TV={tv:.2f}; MA|d|={mad:.3f}; self {gt_self:.2f}->{pred_self:.2f}"
        matrices.append((base.MODEL_LABELS[model], tr["gt"], tr["pred"], summary))
    return plot_matrix_triptych(
        matrices,
        output_dir,
        f"metric_LatentStateTransitionLag{lag}K11_gt_pred_delta.svg",
        f"LatentStateTransitionLag{lag}K11_score01: GT/pred transition matrices and signed delta",
        delta_label="Pred transition prob - GT",
        sequential_floor=0.0,
    )


def write_manifest(rows: list[dict[str, object]], output_dir: Path) -> None:
    fieldnames = [
        "score_key",
        "family",
        "official_score_available",
        "raw_quantity_visualized",
        "plot_filename",
        "exact_implementation_match",
        "notes",
    ]
    with (output_dir / "metric_visualization_manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_manifest(
    rows: list[dict[str, object]],
    score_key: str,
    raw: str,
    filename: str,
    *,
    exact: bool = True,
    notes: str = "",
) -> None:
    rows.append(
        {
            "score_key": score_key,
            "family": family_for_key(score_key),
            "official_score_available": True,
            "raw_quantity_visualized": raw,
            "plot_filename": filename,
            "exact_implementation_match": bool(exact),
            "notes": notes,
        }
    )


def filename_text(filename: str | tuple[str, ...]) -> str:
    if isinstance(filename, tuple):
        return "; ".join(filename)
    return filename


def export_official_score_tables(
    model_arrays: dict[str, base.ModelArrays],
    output_dir: Path,
    score_cache: Path,
    region_names: list[str],
    n_splits: int,
    force_scores: bool,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    if force_scores:
        means, sems, _ = base.compute_or_load_scores(
            model_arrays=model_arrays,
            output_dir=output_dir,
            region_names=region_names,
            n_splits=n_splits,
            force_scores=True,
        )
    else:
        means, sems = load_three_seed_score_cache(score_cache, n_splits)
    metric_df, metric_sem_df, family_df, family_sem_df = base.score_tables(means, sems)
    metric_df.to_csv(output_dir / "selected_nethobench_submetrics_mean.csv", float_format="%.6f")
    metric_sem_df.to_csv(output_dir / "selected_nethobench_submetrics_sem.csv", float_format="%.6f")
    family_df.to_csv(output_dir / "nethobench_family_scores_mean.csv", float_format="%.6f")
    family_sem_df.to_csv(output_dir / "nethobench_family_scores_sem.csv", float_format="%.6f")
    return means, sems


def load_three_seed_score_cache(
    score_cache: Path,
    expected_splits: int,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Aggregate the existing cache using the original nested seed design."""
    if not score_cache.is_file():
        raise FileNotFoundError(f"Official score cache not found: {score_cache}")
    payload = json.loads(score_cache.read_text(encoding="utf-8"))
    signature = payload.get("_cache_signature", {})
    if int(signature.get("n_splits", -1)) != expected_splits:
        raise ValueError(
            f"{score_cache} declares n_splits={signature.get('n_splits')}; "
            f"expected {expected_splits}"
        )
    scores = payload.get("scores")
    if not isinstance(scores, dict) or not scores:
        raise ValueError(f"{score_cache} has no non-empty 'scores' mapping")

    per_seed_means: dict[str, dict[str, dict[str, float]]] = {}
    for training_seed, model_scores in scores.items():
        if not isinstance(model_scores, dict):
            raise ValueError(f"Invalid model mapping for training seed {training_seed}")
        per_seed_means[training_seed] = {}
        for model in base.MODEL_ORDER:
            splits = model_scores.get(model)
            if not isinstance(splits, list) or len(splits) != expected_splits:
                raise ValueError(
                    f"{model} at training seed {training_seed} does not have "
                    f"exactly {expected_splits} cached splits"
                )
            per_seed_means[training_seed][model], _ = base.aggregate_split_scores(
                splits
            )

    means: dict[str, dict[str, float]] = {}
    sems: dict[str, dict[str, float]] = {}
    for model in base.MODEL_ORDER:
        score_names = sorted(
            {
                score_name
                for seed_means in per_seed_means.values()
                for score_name in seed_means[model]
            }
        )
        means[model] = {}
        sems[model] = {}
        for score_name in score_names:
            values = np.asarray(
                [
                    seed_means[model].get(score_name, np.nan)
                    for seed_means in per_seed_means.values()
                ],
                dtype=float,
            )
            means[model][score_name] = float(np.nanmean(values))
            sems[model][score_name] = base.nansem(values)

    print(
        f"Loaded official scores for {len(scores)} training seeds from "
        f"{score_cache}; no scores recomputed"
    )
    return means, sems


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base.setup_plot_style()

    model_arrays = load_three_seed_model_arrays(args.data_dir, args.score_cache)
    n_reg = next(iter(model_arrays.values())).gt.shape[2]
    region_names = [f"R{i}" for i in range(n_reg)]

    # Load the official split-then-seed aggregates even when CSV export is
    # disabled: figure annotations must never be recomputed on the pooled
    # plotting tensor.  ``--force-scores`` retains its explicit recomputation
    # behavior and uses the now forecast-aligned plotting arrays.
    if args.force_scores:
        official_means, official_sems = export_official_score_tables(
            model_arrays,
            args.output_dir,
            args.score_cache,
            region_names,
            args.n_splits,
            args.force_scores,
        )
    else:
        official_means, official_sems = load_three_seed_score_cache(
            args.score_cache, args.n_splits
        )
        if not args.skip_score_cache:
            metric_df, metric_sem_df, family_df, family_sem_df = base.score_tables(
                official_means, official_sems
            )
            metric_df.to_csv(args.output_dir / "selected_nethobench_submetrics_mean.csv", float_format="%.6f")
            metric_sem_df.to_csv(args.output_dir / "selected_nethobench_submetrics_sem.csv", float_format="%.6f")
            family_df.to_csv(args.output_dir / "nethobench_family_scores_mean.csv", float_format="%.6f")
            family_sem_df.to_csv(args.output_dir / "nethobench_family_scores_sem.csv", float_format="%.6f")

    manifest: list[dict[str, object]] = []

    add_manifest(manifest, "KL_or_JSD_score01", "Pooled forecast-only GT activity density, pooled prediction activity density, and per-sequence/region trimmed-histogram symmetric KL distribution.", filename_text(plot_kl(model_arrays, args.output_dir, official_means)), notes="Raw distributions use the forecast-only window. Displayed scores are the official four-split means averaged within training seed and then across training seeds.")
    add_manifest(manifest, "QNT_score01", "Pooled GT quantile curve, pooled prediction quantile curve, and official normalized tail quantile error curve.", filename_text(plot_qnt(model_arrays, args.output_dir)), notes="Mirrors q=0.01..0.99, n_q=99, max_time=1200, rng_seed=0; plot includes GT and prediction quantile objects plus error.")
    add_manifest(manifest, "MOM_score01", "GT/pred variance ratio, signed skewness/kurtosis differences, and official weighted component distances.", filename_text(plot_mom(model_arrays, args.output_dir)), notes="Uses the legacy perfected moment score path called by compute_moment_score01; plot includes signed GT-pred moment context and distance components.")
    add_manifest(manifest, "Mean_score01", "GT regional means, prediction regional means, signed Pred-GT regional mean shifts, and IQR-scaled absolute mean shifts.", filename_text(plot_mean(model_arrays, args.output_dir, official_means)), notes="Raw quantities use the forecast-only window; displayed scores use the official nested four-split aggregation.")
    add_manifest(manifest, "TRJDIST_score01", "Official GT-PCA component distances plus GT/pred speed distributions, turn distributions, and path feature relative differences.", filename_text(plot_trajectory(model_arrays, args.output_dir)), notes="Mirrors trajectory_occupancy_velocity_v4 and trajectory_path_features_v3; outputs include component mismatch and GT/pred latent-dynamics objects.")
    add_manifest(manifest, "GRAPH_score01", "GT correlation matrix, prediction correlation matrix, signed correlation delta, and GT/pred top-edge masks.", filename_text(plot_graph(model_arrays, args.output_dir)), notes="Uses the legacy perfected graph score path called by compute_graph_score01; triptych includes edge-overlap summaries.")
    add_manifest(manifest, "CrossRegionMI_score01", "GT mutual-information matrix, prediction mutual-information matrix, and signed MI delta.", filename_text(plot_cross_mi(model_arrays, args.output_dir)), notes="Uses _mi_matrix and standardization from compute_additional_structural_metrics; triptych includes GT, prediction, and delta.")
    add_manifest(manifest, "LaggedCovariance_score01", "GT lagged covariance matrices, prediction lagged covariance matrices, signed lagged-covariance deltas, and lag-wise mean absolute delta summary.", filename_text(plot_lagged_cov(model_arrays, args.output_dir)), notes="Uses official _lagged_covariance lags 1, 2, and 4; outputs include log-scaled lag summary and lag-1 GT/pred/delta triptych.")
    add_manifest(manifest, "ImpulseResponse_score01", "GT VAR(1) operator matrix, prediction VAR(1) operator matrix, and signed operator delta.", filename_text(plot_impulse_response(model_arrays, args.output_dir)), notes="Uses _var1_coefficients from compute_additional_structural_metrics; triptych includes GT, prediction, delta, Frobenius norm, and matrix correlation.")

    mani_file, mani_exact, mani_notes = plot_mani(model_arrays, args.output_dir)
    add_manifest(manifest, "MANI_score01", "GT and prediction H0/H1 persistent-homology lifetime distributions plus GT and prediction local 5-NN distance profiles.", filename_text(mani_file), exact=mani_exact, notes=mani_notes)

    add_manifest(manifest, "SubspaceAngle_score01", "Principal angles between GT and prediction subspaces plus GT explained variance and prediction variance along GT PCs.", filename_text(plot_subspace_angles(model_arrays, args.output_dir, official_means)), notes="Raw quantities use the forecast-only window; displayed scores use the official nested four-split aggregation.")
    add_manifest(manifest, "LatentStateOccupancyK11_score01", "GT-defined K=11 latent-state occupancy histogram and prediction occupancy histogram with TV and entropy-delta summaries.", filename_text(plot_state_occupancy(model_arrays, args.output_dir, k=11)), notes="Uses _prepare_latent_state_reference and KMeans random_state=K; plot includes GT and prediction state occupancy objects.")
    add_manifest(manifest, "LatentStateOccupancyK12_score01", "GT-defined K=12 latent-state occupancy histogram and prediction occupancy histogram with TV and entropy-delta summaries.", filename_text(plot_state_occupancy(model_arrays, args.output_dir, k=12)), notes="Uses _prepare_latent_state_reference and KMeans random_state=K; plot includes GT and prediction state occupancy objects.")
    for lag in (1, 2, 3):
        add_manifest(
            manifest,
            f"LatentStateTransitionLag{lag}K11_score01",
            f"GT-defined K=11 latent-state transition probability matrix, prediction transition matrix, and signed transition delta at lag {lag}.",
            filename_text(plot_state_transition(model_arrays, args.output_dir, lag=lag)),
            notes="Uses _prepare_latent_state_reference, K=11, and the official transition lag; triptych includes TV, mean absolute delta, and self-transition summaries.",
        )

    write_manifest(manifest, args.output_dir)
    print(f"Saved metric-specific visualizations and manifest to: {args.output_dir}")


if __name__ == "__main__":
    main()
