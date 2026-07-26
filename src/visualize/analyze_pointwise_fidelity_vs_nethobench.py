#!/usr/bin/env python3
"""
Compare pointwise fidelity with NethoBench structural realism.

This analysis evaluates the held-out widefield rollout suite in
``evaluation_results/90_810/seed_102``:

    VAR, RNN, SSM, 1_step, Sequifier, AR, TF, TF_QL_KL

For 810-step arrays, both prediction and ground truth are sliced at ``90:810``.
Arrays that are already forecast-only (such as the 720-step Sequifier arrays)
are scored from timestep zero. Pointwise Error, MI, and Fidelity scores are
computed on the same four sequence splits as the structural NethoBench cache,
using the existing functions in
``nethobench.neuro.metrics.composites`` and
``nethobench.neuro.metrics.definitions``.

Reproduce from the repository root:

    python src/visualize/analyze_pointwise_fidelity_vs_nethobench.py

Use ``--recompute-structural`` to calculate the five families and composite
directly from the arrays instead of loading the existing four-split cache.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_REPO_ROOT = Path(__file__).resolve().parents[2]
_NETHOBENCH_ROOT = _REPO_ROOT / "nethobench"
if str(_NETHOBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_NETHOBENCH_ROOT))

from nethobench.neuro.metrics.composites import (
    calculate_neuro_composites,
    compute_error_score,
    compute_mi_score,
)
from nethobench.neuro.metrics.definitions import compute_fidelity_composite
from neuro_scoring_windows import (
    align_forecast_only,
    scoring_start_for_prediction,
)


FULL_SEQUENCE_LENGTH = 810
CONTEXT_STEPS = 90
DEFAULT_N_SPLITS = 4


@dataclass(frozen=True)
class ModelSpec:
    prediction_file: str
    ground_truth_file: str
    structural_cache_name: str
    color: str


# The order and display names follow the requested widefield AR suite.
MODEL_SPECS: "OrderedDict[str, ModelSpec]" = OrderedDict(
    [
        (
            "VAR",
            ModelSpec(
                "long_predictions_90_810_VAR_BASELINE.npy",
                "long_ground_truth_90_810.npy",
                "VAR",
                "#3E7CB1",
            ),
        ),
        (
            "RNN",
            ModelSpec(
                "long_predictions_90_810_GRU_AR.npy",
                "long_ground_truth_90_810.npy",
                "RNN",
                "#E6AB02",
            ),
        ),
        (
            "SSM",
            ModelSpec(
                "long_predictions_90_810_cDMM_SSM.npy",
                "long_ground_truth_90_810.npy",
                "SMM",
                "#D95F02",
            ),
        ),
        (
            "1_step",
            ModelSpec(
                "long_predictions_90_810_1_step.npy",
                "long_ground_truth_90_810.npy",
                "1_step",
                "#7A7A7A",
            ),
        ),
        (
            "Sequifier",
            ModelSpec(
                "long_predictions_sequifier_last100.npy",
                "long_ground_truth_sequifier_last100.npy",
                "sequifier",
                "#C46410",
            ),
        ),
        (
            "AR",
            ModelSpec(
                "long_predictions_90_810_AR_KV.npy",
                "long_ground_truth_90_810.npy",
                "AR",
                "#2AA876",
            ),
        ),
        (
            "TF",
            ModelSpec(
                "long_predictions_90_810_TF.npy",
                "long_ground_truth_90_810.npy",
                "TF",
                "#A23B72",
            ),
        ),
        (
            "TF_QL_KL",
            ModelSpec(
                "long_predictions_90_810_TF_QTL_0.08_KL_0.02.npy",
                "long_ground_truth_90_810.npy",
                "TF_QL_0.08_KL_0.02",
                "#6A4C93",
            ),
        ),
    ]
)

STRUCTURAL_SCORES: "OrderedDict[str, str]" = OrderedDict(
    [
        ("FINAL_COMPOSITE_SCORE", "Composite"),
        ("family_distribution", "Distribution"),
        ("family_temporal_spectral", "Temporal"),
        ("family_relational", "Relational"),
        ("family_geometry", "Geometry"),
        ("family_state_dynamics", "State dynamics"),
    ]
)
FAMILY_KEYS = [key for key in STRUCTURAL_SCORES if key.startswith("family_")]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_REPO_ROOT / "evaluation_results" / "90_810" / "seed_102",
    )
    parser.add_argument(
        "--structural-cache",
        type=Path,
        default=(
            _REPO_ROOT
            / "output"
            / "neuro_subscores_from_npy_merged_4split"
            / "scores_cache_90_810_4split.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "output" / "pointwise_fidelity_vs_nethobench",
    )
    parser.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS)
    parser.add_argument(
        "--recompute-structural",
        action="store_true",
        help="Ignore the existing structural cache and recompute NethoBench scores.",
    )
    return parser.parse_args(argv)


def sequence_split_indices(n_sequences: int, n_splits: int) -> list[np.ndarray]:
    if n_splits < 1:
        raise ValueError("n_splits must be positive")
    if n_sequences < n_splits:
        raise ValueError(
            f"Need at least {n_splits} sequences, got {n_sequences}"
        )
    return list(np.array_split(np.arange(n_sequences), n_splits))


def mean_sem(values: list[float]) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(finite))
    sem = (
        float(np.std(finite, ddof=1) / np.sqrt(finite.size))
        if finite.size >= 2
        else float("nan")
    )
    return mean, sem


def compute_pointwise_fidelity(
    gt: np.ndarray,
    pred: np.ndarray,
    n_splits: int,
) -> dict[str, object]:
    """Compute existing sidecar Error/MI/Fidelity scores over sequence splits."""
    per_split: list[dict[str, float]] = []
    for indices in sequence_split_indices(gt.shape[0], n_splits):
        error_score = float(compute_error_score(gt[indices], pred[indices]))
        mi_score = float(compute_mi_score(gt[indices], pred[indices]))
        fidelity_score = float(
            compute_fidelity_composite(
                {"Error_score": error_score, "MI_score": mi_score}
            )
        )
        per_split.append(
            {
                "Error_score": error_score,
                "MI_score": mi_score,
                "FIDELITY_SCORE": fidelity_score,
            }
        )

    means: dict[str, float] = {}
    sems: dict[str, float] = {}
    for key in ("Error_score", "MI_score", "FIDELITY_SCORE"):
        means[key], sems[key] = mean_sem(
            [float(scores[key]) for scores in per_split]
        )
    return {"mean": means, "sem": sems, "per_split": per_split}


def aggregate_structural_splits(
    split_scores: list[Mapping[str, float]],
) -> tuple[dict[str, float], dict[str, float]]:
    means: dict[str, float] = {}
    sems: dict[str, float] = {}
    for key in STRUCTURAL_SCORES:
        means[key], sems[key] = mean_sem(
            [float(scores.get(key, np.nan)) for scores in split_scores]
        )
    return means, sems


def compute_structural_scores(
    arrays: Mapping[str, tuple[np.ndarray, np.ndarray]],
    n_splits: int,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Recompute NethoBench scores with the existing in-memory evaluator."""
    means: dict[str, dict[str, float]] = {}
    sems: dict[str, dict[str, float]] = {}
    for model_name, (gt, pred) in arrays.items():
        print(f"Recomputing structural NethoBench scores for {model_name}")
        split_scores = [
            calculate_neuro_composites(gt[indices], pred[indices])
            for indices in sequence_split_indices(gt.shape[0], n_splits)
        ]
        means[model_name], sems[model_name] = aggregate_structural_splits(
            split_scores
        )
    return means, sems


def load_structural_scores(
    cache_path: Path,
    n_splits: int,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    signature = payload.get("_cache_signature", {})
    expected_rule = {
        "full_sequence_length": FULL_SEQUENCE_LENGTH,
        "context_steps_dropped": CONTEXT_STEPS,
    }
    if int(signature.get("n_splits", -1)) != n_splits:
        raise ValueError(
            f"Structural cache uses n_splits={signature.get('n_splits')}, "
            f"requested {n_splits}"
        )
    if signature.get("forecast_scoring_rule") != expected_rule:
        raise ValueError(
            "Structural cache does not use the required forecast-only rule: "
            f"{signature.get('forecast_scoring_rule')}"
        )

    cached_means = payload["means"]
    cached_sems = payload.get("sem", payload.get("sems", {}))
    means: dict[str, dict[str, float]] = {}
    sems: dict[str, dict[str, float]] = {}
    for model_name, spec in MODEL_SPECS.items():
        cache_name = spec.structural_cache_name
        if cache_name not in cached_means:
            raise KeyError(
                f"Structural cache lacks {cache_name!r} for {model_name}"
            )
        means[model_name] = {
            key: float(cached_means[cache_name].get(key, np.nan))
            for key in STRUCTURAL_SCORES
        }
        sems[model_name] = {
            key: float(cached_sems.get(cache_name, {}).get(key, np.nan))
            for key in STRUCTURAL_SCORES
        }
    return means, sems


def rank_correlations(
    table: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    """Spearman correlations between fidelity and each structural ranking."""
    results: dict[str, dict[str, object]] = {}
    fidelity = table["FIDELITY_SCORE"].to_numpy(dtype=float)
    fidelity_ranks = table["FIDELITY_SCORE"].rank(
        ascending=False, method="average"
    )
    for key, label in STRUCTURAL_SCORES.items():
        structural = table[key].to_numpy(dtype=float)
        valid = np.isfinite(fidelity) & np.isfinite(structural)
        rho, p_value = spearmanr(fidelity[valid], structural[valid])
        structural_ranks = table[key].rank(ascending=False, method="average")
        results[key] = {
            "label": label,
            "spearman_rho": float(rho),
            "p_value": float(p_value),
            "n_models": int(valid.sum()),
            "fidelity_ranking": {
                model: float(fidelity_ranks.loc[model]) for model in table.index
            },
            "structural_ranking": {
                model: float(structural_ranks.loc[model]) for model in table.index
            },
        }
    return results


def pairwise_fidelity_structure_test(table: pd.DataFrame) -> dict[str, object]:
    """
    Quantify whether small fidelity gaps can coexist with distinct family profiles.

    Structural distance is the RMS difference over the five family scores after
    each family is min-max scaled across the evaluated models.
    """
    profiles = table[FAMILY_KEYS].astype(float)
    spans = profiles.max(axis=0) - profiles.min(axis=0)
    spans = spans.where(spans > 0.0, 1.0)
    normalized = (profiles - profiles.min(axis=0)) / spans

    rows: list[dict[str, object]] = []
    for model_a, model_b in combinations(table.index, 2):
        fidelity_gap = abs(
            float(table.loc[model_a, "FIDELITY_SCORE"])
            - float(table.loc[model_b, "FIDELITY_SCORE"])
        )
        profile_delta = (
            normalized.loc[model_a].to_numpy(dtype=float)
            - normalized.loc[model_b].to_numpy(dtype=float)
        )
        structural_distance = float(np.sqrt(np.nanmean(profile_delta**2)))
        rows.append(
            {
                "model_a": str(model_a),
                "model_b": str(model_b),
                "absolute_fidelity_gap": fidelity_gap,
                "structural_profile_distance": structural_distance,
                "absolute_composite_gap": abs(
                    float(table.loc[model_a, "FINAL_COMPOSITE_SCORE"])
                    - float(table.loc[model_b, "FINAL_COMPOSITE_SCORE"])
                ),
            }
        )

    gaps = np.asarray([row["absolute_fidelity_gap"] for row in rows], dtype=float)
    distances = np.asarray(
        [row["structural_profile_distance"] for row in rows], dtype=float
    )
    rho, p_value = spearmanr(gaps, distances)
    threshold = float(np.quantile(gaps, 0.25))
    similar = [
        row for row in rows if float(row["absolute_fidelity_gap"]) <= threshold
    ]
    similar.sort(
        key=lambda row: float(row["structural_profile_distance"]), reverse=True
    )
    return {
        "definition": (
            "Similar fidelity means an absolute fidelity gap at or below the "
            "first quartile across all model pairs. Structural distance is RMS "
            "distance across min-max-scaled five-family profiles."
        ),
        "similar_fidelity_threshold": threshold,
        "fidelity_gap_vs_structural_distance_spearman_rho": float(rho),
        "p_value": float(p_value),
        "similar_fidelity_structural_contrasts": similar,
        "all_pairs": rows,
    }


def setup_plot_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "svg.fonttype": "none",
            "axes.linewidth": 0.8,
            "font.size": 9,
        }
    )


def plot_model_panels(table: pd.DataFrame, output_path: Path) -> None:
    setup_plot_style()
    keys = ["FIDELITY_SCORE", *STRUCTURAL_SCORES.keys()]
    labels = [
        "Pointwise\nfidelity",
        *[STRUCTURAL_SCORES[key] for key in STRUCTURAL_SCORES],
    ]
    fig, axes = plt.subplots(4, 2, figsize=(12.0, 12.5), sharey=True)
    for ax, model_name in zip(axes.flat, table.index):
        values = table.loc[model_name, keys].to_numpy(dtype=float)
        color = MODEL_SPECS[model_name].color
        colors = ["#222222", *([color] * len(STRUCTURAL_SCORES))]
        bars = ax.bar(
            np.arange(len(keys)),
            values,
            color=colors,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.9,
        )
        bars[0].set_hatch("//")
        ax.set_title(model_name, color=color, fontweight="bold")
        ax.set_xticks(np.arange(len(keys)))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(value + 0.025, 0.98),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    for ax in axes[:, 0]:
        ax.set_ylabel("Score (higher is better)")
    fig.suptitle(
        "Pointwise predictive fidelity versus NethoBench structural realism",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def plot_rank_correlations(
    correlations: Mapping[str, Mapping[str, object]],
    output_path: Path,
) -> None:
    setup_plot_style()
    keys = list(STRUCTURAL_SCORES)
    labels = [str(correlations[key]["label"]) for key in keys]
    values = [float(correlations[key]["spearman_rho"]) for key in keys]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(
        np.arange(len(keys)),
        values,
        color=["#4C78A8", "#3E7CB1", "#A23B72", "#2AA876", "#6A4C93", "#D95F02"],
        edgecolor="black",
        linewidth=0.6,
    )
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_ylim(-1.0, 1.0)
    ax.set_ylabel("Spearman rank correlation with fidelity")
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + (0.04 if value >= 0 else -0.07),
            f"{value:.2f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
        )
    ax.set_title("Do pointwise-fidelity and structural-realism rankings agree?")
    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def plot_pairwise_test(pairwise: Mapping[str, object], output_path: Path) -> None:
    setup_plot_style()
    pairs = list(pairwise["all_pairs"])
    x = np.asarray([row["absolute_fidelity_gap"] for row in pairs], dtype=float)
    y = np.asarray(
        [row["structural_profile_distance"] for row in pairs], dtype=float
    )
    threshold = float(pairwise["similar_fidelity_threshold"])

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.scatter(x, y, s=38, color="#4C78A8", edgecolor="black", linewidth=0.4)
    ax.axvspan(0.0, threshold, color="#D95F02", alpha=0.12)
    for row in pairwise["similar_fidelity_structural_contrasts"][:3]:
        ax.annotate(
            f"{row['model_a']}–{row['model_b']}",
            (
                float(row["absolute_fidelity_gap"]),
                float(row["structural_profile_distance"]),
            ),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Absolute pointwise-fidelity difference")
    ax.set_ylabel("Five-family structural-profile distance")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_title("Similar prediction fidelity can mask structural differences")
    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_reproduction_document(
    path: Path,
    args: argparse.Namespace,
    input_metadata: Mapping[str, Mapping[str, object]],
    structural_source: str,
) -> None:
    lines = [
        "# Pointwise fidelity versus NethoBench structural realism",
        "",
        "Run from the repository root:",
        "",
        "```bash",
        "python src/visualize/analyze_pointwise_fidelity_vs_nethobench.py",
        "```",
        "",
        "Scoring rule: predictions with 810 time steps and their matching ground "
        "truth are scored over source slice `90:810`. Arrays already containing "
        "only the 720-step forecast are scored over `0:720`.",
        "",
        "Pointwise fidelity uses the existing NethoBench sidecar:",
        "",
        "`Fidelity = 0.65 * Error_score + 0.35 * MI_score`.",
        "",
        f"Structural score source: `{structural_source}`.",
        f"Sequence splits: `{args.n_splits}`.",
        "",
        "## Exact inputs",
        "",
    ]
    for model_name, metadata in input_metadata.items():
        lines.extend(
            [
                f"- **{model_name}**",
                f"  - prediction: `{metadata['prediction_path']}`",
                f"  - ground truth: `{metadata['ground_truth_path']}`",
                f"  - scored source slice: `{metadata['source_start']}:{metadata['source_stop']}`",
                f"  - scored shape: `{tuple(metadata['scored_shape'])}`",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> dict[str, Path]:
    args = parse_args(argv)
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    input_metadata: dict[str, dict[str, object]] = {}
    pointwise: dict[str, dict[str, object]] = {}

    for model_name, spec in MODEL_SPECS.items():
        pred_path = data_dir / spec.prediction_file
        gt_path = data_dir / spec.ground_truth_file
        if not pred_path.is_file() or not gt_path.is_file():
            raise FileNotFoundError(
                f"{model_name}: missing prediction or GT: {pred_path}, {gt_path}"
            )
        pred_raw = np.load(pred_path, allow_pickle=False)
        gt_raw = np.load(gt_path, allow_pickle=False)
        gt, pred = align_forecast_only(
            gt_raw,
            pred_raw,
            full_sequence_length=FULL_SEQUENCE_LENGTH,
            context_steps=CONTEXT_STEPS,
        )
        if gt.shape != pred.shape:
            raise ValueError(f"{model_name}: aligned shape mismatch {gt.shape} vs {pred.shape}")
        if not np.isfinite(gt).all() or not np.isfinite(pred).all():
            raise ValueError(f"{model_name}: non-finite values in aligned arrays")

        arrays[model_name] = (
            gt.astype(np.float64, copy=False),
            pred.astype(np.float64, copy=False),
        )
        source_start = scoring_start_for_prediction(
            pred_raw,
            full_sequence_length=FULL_SEQUENCE_LENGTH,
            context_steps=CONTEXT_STEPS,
        )
        input_metadata[model_name] = {
            "prediction_path": pred_path,
            "ground_truth_path": gt_path,
            "prediction_raw_shape": list(pred_raw.shape),
            "ground_truth_raw_shape": list(gt_raw.shape),
            "source_start": source_start,
            "source_stop": source_start + gt.shape[1],
            "scored_shape": list(gt.shape),
        }
        print(
            f"{model_name}: scoring {gt.shape} from source "
            f"{source_start}:{source_start + gt.shape[1]}"
        )
        pointwise[model_name] = compute_pointwise_fidelity(
            arrays[model_name][0], arrays[model_name][1], args.n_splits
        )

    structural_source: str
    if args.recompute_structural:
        structural_means, structural_sems = compute_structural_scores(
            arrays, args.n_splits
        )
        structural_source = "recomputed from aligned arrays"
    else:
        if not args.structural_cache.is_file():
            raise FileNotFoundError(
                f"Structural cache not found: {args.structural_cache}. "
                "Pass --recompute-structural to calculate it."
            )
        structural_means, structural_sems = load_structural_scores(
            args.structural_cache, args.n_splits
        )
        structural_source = str(args.structural_cache.resolve())

    rows: list[dict[str, object]] = []
    per_model: dict[str, dict[str, object]] = {}
    for model_name in MODEL_SPECS:
        fidelity_mean = pointwise[model_name]["mean"]
        fidelity_sem = pointwise[model_name]["sem"]
        row: dict[str, object] = {
            "model": model_name,
            **fidelity_mean,
            **structural_means[model_name],
        }
        rows.append(row)
        per_model[model_name] = {
            "pointwise_fidelity": pointwise[model_name],
            "nethobench_structural_mean": structural_means[model_name],
            "nethobench_structural_sem": structural_sems[model_name],
            "input": input_metadata[model_name],
            "fidelity_formula_check": (
                0.65 * float(fidelity_mean["Error_score"])
                + 0.35 * float(fidelity_mean["MI_score"])
            ),
            "fidelity_sem": fidelity_sem,
        }

    table = pd.DataFrame(rows).set_index("model").loc[list(MODEL_SPECS)]
    correlations = rank_correlations(table)
    pairwise = pairwise_fidelity_structure_test(table)

    json_path = output_dir / "pointwise_fidelity_vs_nethobench.json"
    csv_path = output_dir / "pointwise_fidelity_vs_nethobench.csv"
    panels_path = output_dir / "pointwise_fidelity_vs_nethobench_panels.svg"
    correlation_path = output_dir / "pointwise_fidelity_rank_correlations.svg"
    pairwise_path = output_dir / "similar_fidelity_structural_profiles.svg"
    reproduction_path = output_dir / "REPRODUCE.md"

    payload = {
        "analysis": "pointwise_fidelity_vs_nethobench_structural_realism",
        "model_order": list(MODEL_SPECS),
        "pointwise_fidelity_formula": (
            "FIDELITY_SCORE = 0.65 * Error_score + 0.35 * MI_score"
        ),
        "pointwise_implementation": {
            "Error_score": (
                "nethobench.neuro.metrics.composites.compute_error_score"
            ),
            "MI_score": "nethobench.neuro.metrics.composites.compute_mi_score",
            "composite": (
                "nethobench.neuro.metrics.definitions.compute_fidelity_composite"
            ),
        },
        "scoring_rule": {
            "full_sequence_length": FULL_SEQUENCE_LENGTH,
            "context_steps_dropped": CONTEXT_STEPS,
            "already_forecast_only_start": 0,
            "n_sequence_splits": args.n_splits,
        },
        "structural_score_source": structural_source,
        "pointwise_fidelity": {
            model: float(pointwise[model]["mean"]["FIDELITY_SCORE"])
            for model in MODEL_SPECS
        },
        "models": per_model,
        "ranking_correlations": correlations,
        "similar_fidelity_test": pairwise,
        "reproduction_command": (
            "python src/visualize/analyze_pointwise_fidelity_vs_nethobench.py"
        ),
    }
    json_path.write_text(
        json.dumps(json_safe(payload), indent=2), encoding="utf-8"
    )
    table.to_csv(csv_path, float_format="%.8f")
    plot_model_panels(table, panels_path)
    plot_rank_correlations(correlations, correlation_path)
    plot_pairwise_test(pairwise, pairwise_path)
    write_reproduction_document(
        reproduction_path, args, input_metadata, structural_source
    )

    print("\nPointwise fidelity and structural scores:")
    print(table.to_string(float_format=lambda value: f"{value:.4f}"))
    print("\nRank correlations with pointwise fidelity:")
    for result in correlations.values():
        print(
            f"  {result['label']}: rho={result['spearman_rho']:.3f}, "
            f"p={result['p_value']:.3g}"
        )
    print(f"\nSaved analysis to: {output_dir}")
    return {
        "json": json_path,
        "csv": csv_path,
        "panels": panels_path,
        "rank_correlations": correlation_path,
        "similar_fidelity": pairwise_path,
        "reproduction": reproduction_path,
    }


if __name__ == "__main__":
    main()
