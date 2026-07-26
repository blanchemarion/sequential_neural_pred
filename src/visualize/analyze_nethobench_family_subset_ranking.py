#!/usr/bin/env python3
"""
Find the smallest Nethobench family subset reproducing the default model ranking.

The five family names and weights are imported from Nethobench's canonical
``NEURO_FAMILY_WEIGHTS`` definition. For every non-empty family subset, the
script evaluates the deterministic ranking and a paired bootstrap over all
cached sequence splits. The split count is inferred from the cache and may be
four, eight, or another common count shared by every model.

Example:

    python src/visualize/analyze_nethobench_family_subset_ranking.py

Use more bootstrap draws or different stability thresholds:

    python src/visualize/analyze_nethobench_family_subset_ranking.py \
        --bootstrap-count 50000 \
        --min-median-spearman 0.95 \
        --min-top-preservation 0.95 \
        --min-pairwise-agreement 0.95
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata


REPO_ROOT = Path(__file__).resolve().parents[2]
NETHOBENCH_ROOT = REPO_ROOT / "nethobench"
if (NETHOBENCH_ROOT / "nethobench" / "__init__.py").is_file():
    nb_path = str(NETHOBENCH_ROOT.resolve())
    if nb_path not in sys.path:
        sys.path.insert(0, nb_path)

from nethobench.neuro.metrics.definitions import NEURO_FAMILY_WEIGHTS


DEFAULT_CACHE = (
    REPO_ROOT
    / "output"
    / "neuro_subscores_from_npy_merged_4split"
    / "scores_cache_90_810_4split.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "output" / "nethobench_family_subset_ranking"
DEFAULT_COMPOSITE_KEY = "FINAL_COMPOSITE_SCORE"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exhaustively identify the smallest Nethobench family subset that "
            "reproduces the default model ranking."
        )
    )
    parser.add_argument(
        "--input-cache",
        type=Path,
        default=DEFAULT_CACHE,
        help="Four-split Nethobench score-cache JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for CSV, JSON, and diagnostic plots.",
    )
    parser.add_argument("--bootstrap-count", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--min-median-spearman", type=float, default=0.90)
    parser.add_argument("--min-top-preservation", type=float, default=0.90)
    parser.add_argument("--min-pairwise-agreement", type=float, default=0.90)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.bootstrap_count <= 0:
        raise ValueError("--bootstrap-count must be positive")
    for name in (
        "min_median_spearman",
        "min_top_preservation",
        "min_pairwise_agreement",
    ):
        value = float(getattr(args, name))
        if not -1.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be between -1 and 1")
    if not 0.0 <= args.min_top_preservation <= 1.0:
        raise ValueError("--min-top-preservation must be between 0 and 1")
    if not 0.0 <= args.min_pairwise_agreement <= 1.0:
        raise ValueError("--min-pairwise-agreement must be between 0 and 1")


def _as_finite_float(
    value: Any, *, model: str, split: int, score_name: str
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{model} split {split + 1}: {score_name} is not numeric: {value!r}"
        ) from exc
    if not np.isfinite(result):
        raise ValueError(
            f"{model} split {split + 1}: {score_name} is missing or non-finite"
        )
    return result


def load_score_cache(
    path: Path,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input score cache not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")

    per_split = payload.get("per_split")
    if not isinstance(per_split, dict) or not per_split:
        raise ValueError(
            f"{path} is missing a non-empty object at key 'per_split'"
        )

    models = list(per_split)
    family_names = list(NEURO_FAMILY_WEIGHTS)
    if len(family_names) != 5:
        raise ValueError(
            "Expected exactly five canonical Nethobench families, found "
            f"{len(family_names)}: {family_names}"
        )

    split_counts: dict[str, int] = {}
    for model, splits in per_split.items():
        if not isinstance(splits, list):
            raise ValueError(f"per_split[{model!r}] must be a list")
        split_counts[model] = len(splits)
    unique_counts = set(split_counts.values())
    if len(unique_counts) != 1:
        raise ValueError(
            "Every model must have the same number of split-level score "
            f"dictionaries; found {split_counts}"
        )
    n_splits = next(iter(unique_counts))
    if n_splits < 2:
        raise ValueError(
            "This analysis requires at least two split-level score "
            f"dictionaries per model; found {split_counts}"
        )

    family_values = np.empty(
        (len(models), n_splits, len(family_names)), dtype=float
    )
    composite_values = np.empty((len(models), n_splits), dtype=float)
    for model_index, model in enumerate(models):
        for split_index, split_scores in enumerate(per_split[model]):
            if not isinstance(split_scores, dict):
                raise ValueError(
                    f"{model} split {split_index + 1} must be a score dictionary"
                )
            for family_index, family in enumerate(family_names):
                key = f"family_{family}"
                if key not in split_scores:
                    raise ValueError(
                        f"{model} split {split_index + 1} is missing {key!r}"
                    )
                family_values[model_index, split_index, family_index] = (
                    _as_finite_float(
                        split_scores[key],
                        model=model,
                        split=split_index,
                        score_name=key,
                    )
                )
            if DEFAULT_COMPOSITE_KEY not in split_scores:
                raise ValueError(
                    f"{model} split {split_index + 1} is missing "
                    f"{DEFAULT_COMPOSITE_KEY!r}"
                )
            composite_values[model_index, split_index] = _as_finite_float(
                split_scores[DEFAULT_COMPOSITE_KEY],
                model=model,
                split=split_index,
                score_name=DEFAULT_COMPOSITE_KEY,
            )

    weights = np.asarray(
        [float(NEURO_FAMILY_WEIGHTS[name]) for name in family_names],
        dtype=float,
    )
    reconstructed = np.sum(family_values * weights[None, None, :], axis=2)
    reconstructed /= float(weights.sum())
    if not np.allclose(reconstructed, composite_values, atol=1e-10, rtol=1e-10):
        max_error = float(np.max(np.abs(reconstructed - composite_values)))
        raise ValueError(
            "Cached default composites do not match the canonical weighted "
            f"family composite; maximum absolute difference={max_error:.3g}"
        )
    return models, family_names, family_values, composite_values, payload


def average_ranks_descending(scores: np.ndarray, axis: int = -1) -> np.ndarray:
    """Rank higher scores first, assigning average ranks to ties."""

    return rankdata(-np.asarray(scores, dtype=float), method="average", axis=axis)


def rowwise_spearman_from_ranks(
    reference_ranks: np.ndarray, candidate_ranks: np.ndarray
) -> np.ndarray:
    x = np.asarray(reference_ranks, dtype=float)
    y = np.asarray(candidate_ranks, dtype=float)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError(f"Expected matching 2D rank arrays, got {x.shape}, {y.shape}")
    x_centered = x - x.mean(axis=1, keepdims=True)
    y_centered = y - y.mean(axis=1, keepdims=True)
    numerator = np.sum(x_centered * y_centered, axis=1)
    denominator = np.sqrt(
        np.sum(x_centered**2, axis=1) * np.sum(y_centered**2, axis=1)
    )
    result = np.full(x.shape[0], np.nan, dtype=float)
    valid = denominator > 0
    result[valid] = numerator[valid] / denominator[valid]
    exact_constant = (~valid) & np.all(np.isclose(x, y), axis=1)
    result[exact_constant] = 1.0
    return result


def scalar_spearman_from_ranks(
    reference_ranks: np.ndarray, candidate_ranks: np.ndarray
) -> float:
    values = rowwise_spearman_from_ranks(
        np.asarray(reference_ranks, dtype=float)[None, :],
        np.asarray(candidate_ranks, dtype=float)[None, :],
    )
    return float(values[0])


def pair_indices(n_models: int) -> tuple[np.ndarray, np.ndarray]:
    pairs = list(itertools.combinations(range(n_models), 2))
    return (
        np.asarray([pair[0] for pair in pairs], dtype=np.int64),
        np.asarray([pair[1] for pair in pairs], dtype=np.int64),
    )


def pair_order_signs(
    ranks: np.ndarray, pair_i: np.ndarray, pair_j: np.ndarray
) -> np.ndarray:
    ranks = np.asarray(ranks, dtype=float)
    return np.sign(ranks[..., pair_i] - ranks[..., pair_j])


def top_overlap(
    reference_ranks: np.ndarray, candidate_ranks: np.ndarray
) -> np.ndarray:
    x = np.asarray(reference_ranks, dtype=float)
    y = np.asarray(candidate_ranks, dtype=float)
    x_top = np.isclose(x, np.min(x, axis=1, keepdims=True))
    y_top = np.isclose(y, np.min(y, axis=1, keepdims=True))
    return np.any(x_top & y_top, axis=1)


def subset_specs(
    family_names: list[str],
) -> list[tuple[int, tuple[int, ...], tuple[str, ...]]]:
    specs: list[tuple[int, tuple[int, ...], tuple[str, ...]]] = []
    subset_id = 0
    for size in range(1, len(family_names) + 1):
        for indices in itertools.combinations(range(len(family_names)), size):
            specs.append(
                (
                    subset_id,
                    indices,
                    tuple(family_names[index] for index in indices),
                )
            )
            subset_id += 1
    return specs


def _subset_label(families: tuple[str, ...]) -> str:
    return " + ".join(name.replace("_", " ").title() for name in families)


def _subset_key(families: tuple[str, ...]) -> str:
    return "|".join(families)


def _finite_summary(values: np.ndarray) -> tuple[float, float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.median(finite)),
        float(np.quantile(finite, 0.025)),
        float(np.quantile(finite, 0.975)),
    )


def analyze_subsets(
    *,
    models: list[str],
    family_names: list[str],
    family_values: np.ndarray,
    composite_values: np.ndarray,
    bootstrap_count: int,
    seed: int,
    min_median_spearman: float,
    min_top_preservation: float,
    min_pairwise_agreement: float,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, np.ndarray],
    np.ndarray,
]:
    n_models, n_splits, _ = family_values.shape
    weights = np.asarray(
        [float(NEURO_FAMILY_WEIGHTS[name]) for name in family_names],
        dtype=float,
    )
    default_means = composite_values.mean(axis=1)
    default_ranks = average_ranks_descending(default_means)
    pair_i, pair_j = pair_indices(n_models)
    default_pair_sign = pair_order_signs(default_ranks, pair_i, pair_j)

    rng = np.random.default_rng(seed)
    bootstrap_indices = rng.integers(
        0, n_splits, size=(bootstrap_count, n_splits), dtype=np.int64
    )
    bootstrap_family_means = np.take(
        family_values, bootstrap_indices, axis=1
    ).mean(axis=2).transpose(1, 0, 2)
    bootstrap_reference_ranks = np.broadcast_to(
        default_ranks, (bootstrap_count, n_models)
    )
    bootstrap_reference_pair_signs = np.broadcast_to(
        default_pair_sign, (bootstrap_count, default_pair_sign.size)
    )

    subset_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    pair_agreements: dict[str, np.ndarray] = {}

    for subset_id, indices_tuple, families_tuple in subset_specs(family_names):
        indices = np.asarray(indices_tuple, dtype=np.int64)
        subset_weights = weights[indices]
        weight_sum = float(subset_weights.sum())
        normalized_weights = subset_weights / weight_sum

        reduced_per_split = np.tensordot(
            family_values[:, :, indices],
            normalized_weights,
            axes=([2], [0]),
        )
        reduced_means = reduced_per_split.mean(axis=1)
        reduced_ranks = average_ranks_descending(reduced_means)

        deterministic_spearman = scalar_spearman_from_ranks(
            default_ranks, reduced_ranks
        )
        kendall = kendalltau(
            default_ranks, reduced_ranks, variant="b", nan_policy="omit"
        ).statistic
        deterministic_kendall = float(kendall) if np.isfinite(kendall) else np.nan
        reduced_pair_sign = pair_order_signs(reduced_ranks, pair_i, pair_j)
        deterministic_pairwise = float(
            np.mean(reduced_pair_sign == default_pair_sign)
        )
        deterministic_top = bool(
            top_overlap(default_ranks[None, :], reduced_ranks[None, :])[0]
        )
        deterministic_exact = bool(np.all(np.isclose(default_ranks, reduced_ranks)))

        bootstrap_reduced_scores = np.tensordot(
            bootstrap_family_means[:, :, indices],
            normalized_weights,
            axes=([2], [0]),
        )
        bootstrap_reduced_ranks = average_ranks_descending(
            bootstrap_reduced_scores, axis=1
        )
        bootstrap_spearman = rowwise_spearman_from_ranks(
            bootstrap_reference_ranks, bootstrap_reduced_ranks
        )
        median_rho, rho_low, rho_high = _finite_summary(bootstrap_spearman)

        bootstrap_subset_pair_signs = pair_order_signs(
            bootstrap_reduced_ranks, pair_i, pair_j
        )
        per_draw_pair_agreement = np.mean(
            bootstrap_subset_pair_signs == bootstrap_reference_pair_signs,
            axis=1,
        )
        pair_agreement_by_pair = np.mean(
            bootstrap_subset_pair_signs == bootstrap_reference_pair_signs,
            axis=0,
        )
        strict_reversal_by_pair = np.mean(
            (bootstrap_subset_pair_signs * bootstrap_reference_pair_signs) < 0,
            axis=0,
        )
        bootstrap_top = top_overlap(
            bootstrap_reference_ranks, bootstrap_reduced_ranks
        )
        bootstrap_exact = np.all(
            np.isclose(bootstrap_reference_ranks, bootstrap_reduced_ranks), axis=1
        )

        top_probability = float(np.mean(bootstrap_top))
        mean_pairwise = float(np.mean(per_draw_pair_agreement))
        exact_probability = float(np.mean(bootstrap_exact))
        passes = bool(
            median_rho >= min_median_spearman
            and top_probability >= min_top_preservation
            and mean_pairwise >= min_pairwise_agreement
        )
        subset_key = _subset_key(families_tuple)
        subset_label = _subset_label(families_tuple)
        pair_agreements[subset_key] = pair_agreement_by_pair

        subset_rows.append(
            {
                "subset_id": subset_id,
                "subset_key": subset_key,
                "subset_label": subset_label,
                "subset_families": json.dumps(list(families_tuple)),
                "subset_size": len(indices_tuple),
                "included_weight_sum": weight_sum,
                "spearman": deterministic_spearman,
                "kendall": deterministic_kendall,
                "pairwise_order_agreement": deterministic_pairwise,
                "top_model_preserved": deterministic_top,
                "exact_order_preserved": deterministic_exact,
                "bootstrap_median_spearman": median_rho,
                "bootstrap_spearman_ci95_low": rho_low,
                "bootstrap_spearman_ci95_high": rho_high,
                "bootstrap_top_model_probability": top_probability,
                "bootstrap_mean_pairwise_agreement": mean_pairwise,
                "bootstrap_exact_order_probability": exact_probability,
                "passes_stability": passes,
            }
        )

        for model_index, model in enumerate(models):
            model_row: dict[str, Any] = {
                "subset_id": subset_id,
                "subset_key": subset_key,
                "subset_label": subset_label,
                "subset_families": json.dumps(list(families_tuple)),
                "subset_size": len(indices_tuple),
                "model": model,
                "reduced_mean_score": float(reduced_means[model_index]),
                "reduced_rank": float(reduced_ranks[model_index]),
                "default_mean_composite": float(default_means[model_index]),
                "default_rank": float(default_ranks[model_index]),
            }
            for split_index in range(n_splits):
                model_row[f"split_{split_index + 1}_reduced_score"] = float(
                    reduced_per_split[model_index, split_index]
                )
                model_row[f"split_{split_index + 1}_default_composite"] = float(
                    composite_values[model_index, split_index]
                )
            model_rows.append(model_row)

        for pair_index, (left_index, right_index) in enumerate(
            zip(pair_i, pair_j)
        ):
            pair_rows.append(
                {
                    "subset_id": subset_id,
                    "subset_key": subset_key,
                    "subset_label": subset_label,
                    "subset_size": len(indices_tuple),
                    "model_a": models[int(left_index)],
                    "model_b": models[int(right_index)],
                    "bootstrap_order_agreement": float(
                        pair_agreement_by_pair[pair_index]
                    ),
                    "bootstrap_strict_reversal_probability": float(
                        strict_reversal_by_pair[pair_index]
                    ),
                }
            )

    subset_results = pd.DataFrame(subset_rows)
    stable = subset_results[subset_results["passes_stability"]]
    minimum_size = int(stable["subset_size"].min()) if not stable.empty else None
    subset_results["smallest_stable_subset"] = (
        subset_results["passes_stability"]
        & (subset_results["subset_size"] == minimum_size)
        if minimum_size is not None
        else False
    )
    return (
        subset_results,
        pd.DataFrame(model_rows),
        pd.DataFrame(pair_rows),
        pair_agreements,
        bootstrap_indices,
    )


def stability_sort(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        [
            "bootstrap_median_spearman",
            "bootstrap_mean_pairwise_agreement",
            "bootstrap_top_model_probability",
            "bootstrap_exact_order_probability",
            "bootstrap_spearman_ci95_low",
            "subset_size",
        ],
        ascending=[False, False, False, False, False, True],
        kind="stable",
    )


def build_default_ranking(
    models: list[str], composite_values: np.ndarray
) -> pd.DataFrame:
    means = composite_values.mean(axis=1)
    ranks = average_ranks_descending(means)
    rows = []
    for model_index, model in enumerate(models):
        row: dict[str, Any] = {
            "model": model,
            "default_mean_composite": float(means[model_index]),
            "default_rank": float(ranks[model_index]),
        }
        for split_index in range(composite_values.shape[1]):
            row[f"split_{split_index + 1}_default_composite"] = float(
                composite_values[model_index, split_index]
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["default_rank", "model"], kind="stable"
    )


def build_input_family_scores(
    models: list[str],
    family_names: list[str],
    family_values: np.ndarray,
    composite_values: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_index, model in enumerate(models):
        for split_index in range(family_values.shape[1]):
            row: dict[str, Any] = {
                "model": model,
                "split": split_index + 1,
                DEFAULT_COMPOSITE_KEY: float(
                    composite_values[model_index, split_index]
                ),
            }
            for family_index, family in enumerate(family_names):
                row[f"family_{family}"] = float(
                    family_values[model_index, split_index, family_index]
                )
            rows.append(row)
    return pd.DataFrame(rows)


def build_pairwise_matrix(
    models: list[str],
    selected_subset_keys: list[str],
    pair_agreements: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    if not selected_subset_keys:
        raise ValueError("At least one subset is required for a pairwise matrix")
    pair_i, pair_j = pair_indices(len(models))
    selected = np.stack(
        [pair_agreements[key] for key in selected_subset_keys], axis=0
    )
    values = selected.mean(axis=0)
    matrix = np.eye(len(models), dtype=float)
    matrix[pair_i, pair_j] = values
    matrix[pair_j, pair_i] = values
    return pd.DataFrame(matrix, index=models, columns=models)


def build_size_summary(subset_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for subset_size, group in subset_results.groupby("subset_size", sort=True):
        spearman_row = group.loc[group["bootstrap_median_spearman"].idxmax()]
        pairwise_row = group.loc[
            group["bootstrap_mean_pairwise_agreement"].idxmax()
        ]
        rows.append(
            {
                "subset_size": int(subset_size),
                "best_bootstrap_median_spearman": float(
                    spearman_row["bootstrap_median_spearman"]
                ),
                "best_spearman_subset": str(spearman_row["subset_key"]),
                "best_bootstrap_mean_pairwise_agreement": float(
                    pairwise_row["bootstrap_mean_pairwise_agreement"]
                ),
                "best_pairwise_subset": str(pairwise_row["subset_key"]),
            }
        )
    return pd.DataFrame(rows)


def plot_subset_size_scatter(
    subset_results: pd.DataFrame,
    output_dir: Path,
    min_median_spearman: float,
) -> tuple[Path, Path]:
    mpl.rcParams["svg.fonttype"] = "none"
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for subset_size, group in subset_results.groupby("subset_size", sort=True):
        offsets = np.linspace(-0.16, 0.16, len(group)) if len(group) > 1 else [0.0]
        colors = np.where(group["passes_stability"], "#B22222", "#7A7A7A")
        ax.scatter(
            subset_size + np.asarray(offsets),
            group["bootstrap_median_spearman"],
            c=colors,
            edgecolors="black",
            linewidths=0.45,
            s=45,
            alpha=0.9,
        )
    smallest = stability_sort(
        subset_results[subset_results["smallest_stable_subset"]]
    )
    for _, row in smallest.head(5).iterrows():
        ax.annotate(
            str(row["subset_label"]),
            (float(row["subset_size"]), float(row["bootstrap_median_spearman"])),
            xytext=(5, -18),
            textcoords="offset points",
            fontsize=7,
            va="top",
        )
    ax.axhline(
        min_median_spearman,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        label="Stability threshold",
    )
    ax.set_xticks(range(1, 6))
    ax.set_xlabel("Number of Nethobench families")
    ax.set_ylabel("Bootstrap median Spearman correlation")
    observed_min = float(subset_results["bootstrap_median_spearman"].min())
    lower_limit = max(-1.05, min(0.0, observed_min - 0.08))
    ax.set_ylim(lower_limit, 1.05)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#B22222",
            markeredgecolor="black",
            markersize=6,
            label="Passes all stability criteria",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#7A7A7A",
            markeredgecolor="black",
            markersize=6,
            label="Does not pass",
        ),
    ]
    threshold_handles, threshold_labels = ax.get_legend_handles_labels()
    ax.legend(
        legend_handles + threshold_handles,
        [item.get_label() for item in legend_handles] + threshold_labels,
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout()
    svg = output_dir / "subset_size_vs_median_spearman.svg"
    png = output_dir / "subset_size_vs_median_spearman.png"
    fig.savefig(svg, format="svg", bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return svg, png

def plot_ranked_subset_bars(
    subset_results: pd.DataFrame, output_dir: Path
) -> tuple[Path, Path]:
    ordered = stability_sort(subset_results).iloc[::-1].reset_index(drop=True)
    y = np.arange(len(ordered))
    median = ordered["bootstrap_median_spearman"].to_numpy(dtype=float)
    low = ordered["bootstrap_spearman_ci95_low"].to_numpy(dtype=float)
    high = ordered["bootstrap_spearman_ci95_high"].to_numpy(dtype=float)
    xerr = np.vstack([median - low, high - median])
    colors = np.where(ordered["passes_stability"], "#B22222", "#7A7A7A")

    fig, ax = plt.subplots(figsize=(9.0, 10.5))
    ax.barh(
        y,
        median,
        xerr=xerr,
        color=colors,
        edgecolor="black",
        linewidth=0.45,
        alpha=0.9,
        capsize=2,
        error_kw={"elinewidth": 0.75, "capthick": 0.75},
    )
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["subset_label"], fontsize=7.5)
    ax.set_xlabel("Bootstrap median Spearman correlation (95% interval)")
    ax.set_xlim(-0.25, 1.05)
    ax.axvline(0.0, color="gray", linewidth=0.8)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    svg = output_dir / "all_family_subsets_ranked_spearman.svg"
    png = output_dir / "all_family_subsets_ranked_spearman.png"
    fig.savefig(svg, format="svg", bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return svg, png


def _format_rank(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.1f}"


def plot_model_rank_heatmap(
    default_ranking: pd.DataFrame,
    model_scores: pd.DataFrame,
    selected_subsets: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    model_order = default_ranking["model"].tolist()
    row_labels = ["Default composite"]
    rows = [
        default_ranking.set_index("model").loc[model_order, "default_rank"].to_numpy(
            dtype=float
        )
    ]
    for _, subset in selected_subsets.iterrows():
        subset_scores = model_scores[
            model_scores["subset_key"] == subset["subset_key"]
        ].set_index("model")
        rows.append(
            subset_scores.loc[model_order, "reduced_rank"].to_numpy(dtype=float)
        )
        row_labels.append(str(subset["subset_label"]))
    matrix = np.vstack(rows)

    fig_height = max(3.2, 0.48 * matrix.shape[0] + 1.8)
    fig, ax = plt.subplots(figsize=(9.0, fig_height))
    image = ax.imshow(
        matrix,
        cmap="viridis_r",
        vmin=1,
        vmax=len(model_order),
        aspect="auto",
    )
    ax.set_xticks(np.arange(len(model_order)))
    ax.set_xticklabels(model_order, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column,
                row,
                _format_rank(float(matrix[row, column])),
                ha="center",
                va="center",
                fontsize=8,
                color=(
                    "white"
                    if matrix[row, column] >= len(model_order) / 2
                    else "black"
                ),
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    colorbar.set_label("Model rank (1 = best)")
    ax.set_title("Model ranks under the best-performing family subsets")
    fig.tight_layout()
    svg = output_dir / "best_subset_model_rank_heatmap.svg"
    png = output_dir / "best_subset_model_rank_heatmap.png"
    fig.savefig(svg, format="svg", bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return svg, png


def plot_pairwise_agreement_heatmap(
    matrix: pd.DataFrame,
    output_dir: Path,
    subtitle: str,
) -> tuple[Path, Path]:
    values = matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    image = ax.imshow(values, cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=40, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(
                column,
                row,
                f"{values[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if values[row, column] < 0.55 else "black",
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    colorbar.set_label("Bootstrap pairwise-order agreement")
    ax.set_title(f"Pairwise ordering stability\n{subtitle}")
    fig.tight_layout()
    svg = output_dir / "smallest_stable_pairwise_agreement_heatmap.svg"
    png = output_dir / "smallest_stable_pairwise_agreement_heatmap.png"
    fig.savefig(svg, format="svg", bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return svg, png


def plot_best_by_size(
    size_summary: pd.DataFrame, output_dir: Path
) -> tuple[Path, Path]:
    x = size_summary["subset_size"].to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    ax.plot(
        x,
        size_summary["best_bootstrap_median_spearman"],
        marker="o",
        linewidth=2,
        color="#3E7CB1",
        label="Best median Spearman",
    )
    ax.plot(
        x,
        size_summary["best_bootstrap_mean_pairwise_agreement"],
        marker="s",
        linewidth=2,
        color="#D95F02",
        label="Best pairwise agreement",
    )
    ax.set_xticks(range(1, 6))
    ax.set_xlabel("Number of Nethobench families")
    ax.set_ylabel("Best achievable bootstrap statistic")
    ax.set_ylim(0.0, 1.03)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.tight_layout()
    svg = output_dir / "best_stability_by_subset_size.svg"
    png = output_dir / "best_stability_by_subset_size.png"
    fig.savefig(svg, format="svg", bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return svg, png


def select_rank_heatmap_subsets(subset_results: pd.DataFrame) -> pd.DataFrame:
    ordered = stability_sort(subset_results)
    smallest = stability_sort(
        subset_results[subset_results["smallest_stable_subset"]]
    )
    selected_ids: list[int] = []
    for subset_id in list(smallest["subset_id"]) + list(ordered["subset_id"]):
        subset_id = int(subset_id)
        if subset_id not in selected_ids:
            selected_ids.append(subset_id)
        if len(selected_ids) >= 10:
            break
    return subset_results.set_index("subset_id").loc[selected_ids].reset_index()


def most_reversed_pairs(
    pair_results: pd.DataFrame, primary_subset_key: str
) -> pd.DataFrame:
    selected = pair_results[
        pair_results["subset_key"] == primary_subset_key
    ].copy()
    return selected.sort_values(
        [
            "bootstrap_strict_reversal_probability",
            "bootstrap_order_agreement",
            "model_a",
            "model_b",
        ],
        ascending=[False, True, True, True],
        kind="stable",
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return _json_safe(frame.to_dict(orient="records"))


def print_terminal_summary(
    default_ranking: pd.DataFrame,
    smallest_stable: pd.DataFrame,
    primary_subset: pd.Series,
    reversed_pairs: pd.DataFrame,
    n_splits: int,
) -> None:
    print(
        f"\nDefault model ranking (mean composite over {n_splits} splits):"
    )
    for _, row in default_ranking.iterrows():
        print(
            f"  {_format_rank(float(row['default_rank']))}. "
            f"{row['model']}: {float(row['default_mean_composite']):.6f}"
        )

    print("\nSmallest stable family subset(s):")
    if smallest_stable.empty:
        print("  None met all configured thresholds.")
    else:
        size = int(smallest_stable["subset_size"].iloc[0])
        print(f"  Minimum size: {size}")
        for _, row in smallest_stable.iterrows():
            print(
                f"  - {row['subset_label']}: median rho="
                f"{float(row['bootstrap_median_spearman']):.3f}, "
                f"top={float(row['bootstrap_top_model_probability']):.3f}, "
                f"pairwise={float(row['bootstrap_mean_pairwise_agreement']):.3f}"
            )

    print(
        "\nMost frequently reversed model pairs for "
        f"{primary_subset['subset_label']}:"
    )
    max_probability = float(
        reversed_pairs["bootstrap_strict_reversal_probability"].max()
    )
    if max_probability <= 0:
        print("  No strict pairwise reversals occurred in the bootstrap.")
    else:
        for _, row in reversed_pairs.head(5).iterrows():
            probability = float(row["bootstrap_strict_reversal_probability"])
            if probability <= 0:
                break
            print(
                f"  - {row['model_a']} vs {row['model_b']}: "
                f"{probability:.3%} reversed"
            )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    cache_path = args.input_cache.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        models,
        family_names,
        family_values,
        composite_values,
        cache_payload,
    ) = load_score_cache(cache_path)
    n_splits = int(family_values.shape[1])
    print(f"Input cache: {cache_path}")
    print(f"Models detected ({len(models)}): {', '.join(models)}")
    print(f"Splits detected per model: {n_splits}")
    print(
        "Canonical family weights: "
        + ", ".join(
            f"{name}={float(NEURO_FAMILY_WEIGHTS[name]):.2f}"
            for name in family_names
        )
    )

    (
        subset_results,
        model_scores,
        pair_results,
        pair_agreements,
        bootstrap_indices,
    ) = analyze_subsets(
        models=models,
        family_names=family_names,
        family_values=family_values,
        composite_values=composite_values,
        bootstrap_count=args.bootstrap_count,
        seed=args.seed,
        min_median_spearman=args.min_median_spearman,
        min_top_preservation=args.min_top_preservation,
        min_pairwise_agreement=args.min_pairwise_agreement,
    )
    default_ranking = build_default_ranking(models, composite_values)
    input_scores = build_input_family_scores(
        models, family_names, family_values, composite_values
    )
    ordered_all = stability_sort(subset_results)
    stability_rank = {
        int(subset_id): rank
        for rank, subset_id in enumerate(ordered_all["subset_id"], start=1)
    }
    subset_results["bootstrap_stability_rank"] = subset_results["subset_id"].map(
        stability_rank
    )
    ordered_all = stability_sort(subset_results)
    smallest_stable = stability_sort(
        subset_results[subset_results["smallest_stable_subset"]]
    )
    primary_subset = (
        smallest_stable.iloc[0] if not smallest_stable.empty else ordered_all.iloc[0]
    )
    selected_pairwise_subsets = (
        smallest_stable["subset_key"].tolist()
        if not smallest_stable.empty
        else [str(primary_subset["subset_key"])]
    )
    pairwise_matrix = build_pairwise_matrix(
        models, selected_pairwise_subsets, pair_agreements
    )
    model_order = default_ranking["model"].tolist()
    pairwise_matrix = pairwise_matrix.loc[model_order, model_order]
    size_summary = build_size_summary(subset_results)
    reversed_pairs = most_reversed_pairs(
        pair_results, str(primary_subset["subset_key"])
    )
    heatmap_subsets = select_rank_heatmap_subsets(subset_results)

    subset_results_path = output_dir / "family_subset_results.csv"
    model_scores_path = output_dir / "family_subset_model_scores.csv"
    pair_results_path = output_dir / "family_subset_pairwise_bootstrap.csv"
    default_ranking_path = output_dir / "default_model_ranking.csv"
    input_scores_path = output_dir / "input_family_split_scores.csv"
    pairwise_matrix_path = output_dir / "smallest_stable_pairwise_matrix.csv"
    size_summary_path = output_dir / "best_by_subset_size.csv"
    reversed_pairs_path = output_dir / "most_reversed_model_pairs.csv"
    smallest_stable_path = output_dir / "smallest_stable_subsets.csv"

    ordered_all.to_csv(subset_results_path, index=False)
    model_scores.to_csv(model_scores_path, index=False)
    pair_results.to_csv(pair_results_path, index=False)
    default_ranking.to_csv(default_ranking_path, index=False)
    input_scores.to_csv(input_scores_path, index=False)
    pairwise_matrix.to_csv(pairwise_matrix_path)
    size_summary.to_csv(size_summary_path, index=False)
    reversed_pairs.to_csv(reversed_pairs_path, index=False)
    smallest_stable.to_csv(smallest_stable_path, index=False)

    scatter_svg, scatter_png = plot_subset_size_scatter(
        subset_results, output_dir, args.min_median_spearman
    )
    bars_svg, bars_png = plot_ranked_subset_bars(subset_results, output_dir)
    ranks_svg, ranks_png = plot_model_rank_heatmap(
        default_ranking, model_scores, heatmap_subsets, output_dir
    )
    pair_svg, pair_png = plot_pairwise_agreement_heatmap(
        pairwise_matrix,
        output_dir,
        (
            "Mean across smallest stable subsets"
            if len(selected_pairwise_subsets) > 1
            else str(primary_subset["subset_label"])
        ),
    )
    size_svg, size_png = plot_best_by_size(size_summary, output_dir)

    results_json_path = output_dir / "family_subset_ranking_results.json"
    results_payload = {
        "config": {
            "input_cache": str(cache_path),
            "output_dir": str(output_dir),
            "bootstrap_count": args.bootstrap_count,
            "seed": args.seed,
            "n_splits": n_splits,
            "tie_method": "average ranks",
            "bootstrap_method": (
                f"resample the {n_splits} split indices with replacement for each "
                "reduced subset, then compare its ranking with the fixed default "
                f"ranking defined by the original {n_splits}-split mean composite"
            ),
            "stability_thresholds": {
                "min_median_spearman": args.min_median_spearman,
                "min_top_preservation": args.min_top_preservation,
                "min_pairwise_agreement": args.min_pairwise_agreement,
            },
        },
        "models": models,
        "family_names": family_names,
        "family_weights": {
            name: float(NEURO_FAMILY_WEIGHTS[name]) for name in family_names
        },
        "cache_signature": cache_payload.get("_cache_signature"),
        "bootstrap_split_indices": bootstrap_indices.tolist(),
        "default_ranking": _records(default_ranking),
        "subset_results": _records(ordered_all),
        "subset_model_scores": _records(model_scores),
        "subset_pairwise_bootstrap": _records(pair_results),
        "smallest_stable_subsets": _records(smallest_stable),
        "primary_subset": _json_safe(primary_subset.to_dict()),
        "most_reversed_pairs": _records(reversed_pairs),
        "smallest_stable_pairwise_matrix": _json_safe(
            pairwise_matrix.to_dict(orient="index")
        ),
        "best_by_subset_size": _records(size_summary),
        "outputs": {
            "subset_results_csv": str(subset_results_path),
            "model_scores_csv": str(model_scores_path),
            "pairwise_bootstrap_csv": str(pair_results_path),
            "default_ranking_csv": str(default_ranking_path),
            "input_family_scores_csv": str(input_scores_path),
            "pairwise_matrix_csv": str(pairwise_matrix_path),
            "best_by_size_csv": str(size_summary_path),
            "reversed_pairs_csv": str(reversed_pairs_path),
            "smallest_stable_subsets_csv": str(smallest_stable_path),
            "subset_size_scatter_svg": str(scatter_svg),
            "subset_size_scatter_png": str(scatter_png),
            "ranked_subsets_svg": str(bars_svg),
            "ranked_subsets_png": str(bars_png),
            "rank_heatmap_svg": str(ranks_svg),
            "rank_heatmap_png": str(ranks_png),
            "pairwise_heatmap_svg": str(pair_svg),
            "pairwise_heatmap_png": str(pair_png),
            "best_by_size_svg": str(size_svg),
            "best_by_size_png": str(size_png),
        },
    }
    results_json_path.write_text(
        json.dumps(
            _json_safe(results_payload),
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print_terminal_summary(
        default_ranking,
        smallest_stable,
        primary_subset,
        reversed_pairs,
        n_splits,
    )
    print(f"\nDetailed results: {results_json_path}")
    print(f"Plots and CSVs: {output_dir}")


if __name__ == "__main__":
    main()
