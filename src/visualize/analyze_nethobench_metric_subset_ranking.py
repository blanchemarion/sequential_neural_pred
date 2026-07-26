#!/usr/bin/env python3
"""
Rank every NethoBench metric combination by recovery of the default model ranking.

The analysis uses the same fixed-reference, paired-split bootstrap statistics
as ``analyze_nethobench_family_subset_ranking.py``. All 2^16 - 1 non-empty
canonical metric combinations are evaluated and ranked by bootstrap median
Spearman correlation, pairwise-order agreement, top-model preservation, and
exact-order recovery. Combination size is reported but is not a ranking
criterion.

Reduced composites follow NethoBench's missing-metric semantics:

1. selected metrics are renormalized within each represented family;
2. represented families are combined using the canonical family weights,
   renormalized over the represented families.

Example:

    python src/visualize/analyze_nethobench_metric_subset_ranking.py

Use more bootstrap draws or stricter informational stability thresholds:

    python src/visualize/analyze_nethobench_metric_subset_ranking.py \
        --bootstrap-count 50000 \
        --min-median-spearman 0.95 \
        --min-top-preservation 0.95 \
        --min-pairwise-agreement 0.95

Exclude one or several models from the reference and reduced rankings:

    python src/visualize/analyze_nethobench_metric_subset_ranking.py \
        --exclude-model VAR

    python src/visualize/analyze_nethobench_metric_subset_ranking.py \
        --exclude-model VAR SMM \
        --exclude-model RNN
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
NETHOBENCH_ROOT = REPO_ROOT / "nethobench"
if (NETHOBENCH_ROOT / "nethobench" / "__init__.py").is_file():
    nb_path = str(NETHOBENCH_ROOT.resolve())
    if nb_path not in sys.path:
        sys.path.insert(0, nb_path)

from nethobench.neuro.metrics.definitions import (
    NEURO_FAMILY_METRICS,
    NEURO_FAMILY_WEIGHTS,
)

try:
    from .analyze_nethobench_family_subset_ranking import (
        DEFAULT_CACHE,
        DEFAULT_COMPOSITE_KEY,
        _format_rank,
        _json_safe,
        _records,
        average_ranks_descending,
        pair_indices,
        pair_order_signs,
        rowwise_spearman_from_ranks,
        top_overlap,
    )
except ImportError:
    from analyze_nethobench_family_subset_ranking import (
        DEFAULT_CACHE,
        DEFAULT_COMPOSITE_KEY,
        _format_rank,
        _json_safe,
        _records,
        average_ranks_descending,
        pair_indices,
        pair_order_signs,
        rowwise_spearman_from_ranks,
        top_overlap,
    )


DEFAULT_OUTPUT = REPO_ROOT / "output" / "nethobench_metric_combination_ranking"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank every non-empty canonical NethoBench metric combination by "
            "how well it recovers the default model ranking."
        )
    )
    parser.add_argument(
        "--input-cache",
        type=Path,
        default=DEFAULT_CACHE,
        help="Four-split NethoBench score-cache JSON containing metric scores.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for CSV, JSON, and diagnostic plots.",
    )
    parser.add_argument("--bootstrap-count", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--min-median-spearman", type=float, default=0.80)
    parser.add_argument("--min-top-preservation", type=float, default=0.80)
    parser.add_argument("--min-pairwise-agreement", type=float, default=0.80)
    parser.add_argument(
        "--top-combinations",
        type=int,
        default=20,
        help=(
            "Number of top-ranked combinations to include in detailed model, "
            "family-breakdown, terminal, and JSON outputs."
        ),
    )
    parser.add_argument(
        "--exclude-model",
        "--exclude-models",
        dest="exclude_models",
        nargs="+",
        action="extend",
        default=["SMM"],
        metavar="MODEL",
        help=(
            "One or more cache model names to omit from the ranking analysis. "
            "The option may also be repeated."
        ),
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.bootstrap_count <= 0:
        raise ValueError("--bootstrap-count must be positive")
    if args.top_combinations <= 0:
        raise ValueError("--top-combinations must be positive")
    if not -1.0 <= args.min_median_spearman <= 1.0:
        raise ValueError("--min-median-spearman must be between -1 and 1")
    if not 0.0 <= args.min_top_preservation <= 1.0:
        raise ValueError("--min-top-preservation must be between 0 and 1")
    if not 0.0 <= args.min_pairwise_agreement <= 1.0:
        raise ValueError("--min-pairwise-agreement must be between 0 and 1")


def canonical_metric_metadata() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_index = 0
    for family, metric_weights in NEURO_FAMILY_METRICS.items():
        for metric, metric_weight in metric_weights.items():
            rows.append(
                {
                    "metric_index": metric_index,
                    "metric": metric,
                    "family": family,
                    "metric_weight_within_family": float(metric_weight),
                    "family_weight": float(NEURO_FAMILY_WEIGHTS[family]),
                }
            )
            metric_index += 1
    return pd.DataFrame(rows)


def _finite_float(
    value: Any, *, model: str, split_index: int, score_name: str
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{model} split {split_index + 1}: {score_name} is not numeric: "
            f"{value!r}"
        ) from exc
    if not np.isfinite(result):
        raise ValueError(
            f"{model} split {split_index + 1}: {score_name} is non-finite"
        )
    return result


def load_metric_score_cache(
    path: Path,
) -> tuple[list[str], pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
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
    metadata = canonical_metric_metadata()
    metric_names = metadata["metric"].tolist()
    split_counts = {
        model: len(splits) if isinstance(splits, list) else -1
        for model, splits in per_split.items()
    }
    if set(split_counts.values()) != {4}:
        raise ValueError(
            "This analysis requires exactly four split-level score dictionaries "
            f"for every model; found {split_counts}"
        )

    metric_values = np.empty((len(models), 4, len(metric_names)), dtype=float)
    composite_values = np.empty((len(models), 4), dtype=float)
    for model_index, model in enumerate(models):
        for split_index, split_scores in enumerate(per_split[model]):
            if not isinstance(split_scores, dict):
                raise ValueError(
                    f"{model} split {split_index + 1} must be a score dictionary"
                )
            for metric_index, metric in enumerate(metric_names):
                if metric not in split_scores:
                    raise ValueError(
                        f"{model} split {split_index + 1} is missing {metric!r}"
                    )
                metric_values[model_index, split_index, metric_index] = (
                    _finite_float(
                        split_scores[metric],
                        model=model,
                        split_index=split_index,
                        score_name=metric,
                    )
                )
            if DEFAULT_COMPOSITE_KEY not in split_scores:
                raise ValueError(
                    f"{model} split {split_index + 1} is missing "
                    f"{DEFAULT_COMPOSITE_KEY!r}"
                )
            composite_values[model_index, split_index] = _finite_float(
                split_scores[DEFAULT_COMPOSITE_KEY],
                model=model,
                split_index=split_index,
                score_name=DEFAULT_COMPOSITE_KEY,
            )

    full_indices = tuple(range(len(metric_names)))
    reconstructed = reduced_composite(metric_values, full_indices, metadata)
    if not np.allclose(reconstructed, composite_values, atol=1e-10, rtol=1e-10):
        max_error = float(np.max(np.abs(reconstructed - composite_values)))
        raise ValueError(
            "Cached composites do not match the canonical hierarchical metric "
            f"composite; maximum absolute difference={max_error:.3g}"
        )
    return models, metadata, metric_values, composite_values, payload


def exclude_models(
    models: list[str],
    metric_values: np.ndarray,
    composite_values: np.ndarray,
    requested_exclusions: Iterable[str],
) -> tuple[list[str], np.ndarray, np.ndarray, list[str]]:
    """Remove requested models while preserving the cache's model order."""

    excluded = list(dict.fromkeys(requested_exclusions))
    unknown = [model for model in excluded if model not in models]
    if unknown:
        raise ValueError(
            "Unknown model name(s) passed to --exclude-model: "
            f"{', '.join(unknown)}. Available models: {', '.join(models)}"
        )

    retained_indices = [
        index for index, model in enumerate(models) if model not in excluded
    ]
    if len(retained_indices) < 2:
        raise ValueError(
            "At least two models must remain after applying --exclude-model; "
            f"{len(retained_indices)} would remain."
        )
    retained_models = [models[index] for index in retained_indices]
    return (
        retained_models,
        metric_values[retained_indices, :, :],
        composite_values[retained_indices, :],
        excluded,
    )


def reduced_metric_coefficients(
    indices: Iterable[int], metadata: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(tuple(indices), dtype=np.int64)
    if selected.size == 0:
        raise ValueError("A metric subset cannot be empty")
    selected_meta = metadata.iloc[selected]
    represented_families = list(dict.fromkeys(selected_meta["family"].tolist()))
    represented_family_weight = float(
        sum(float(NEURO_FAMILY_WEIGHTS[name]) for name in represented_families)
    )
    coefficients = np.empty(selected.size, dtype=float)
    for family in represented_families:
        locations = np.flatnonzero(
            selected_meta["family"].to_numpy(dtype=object) == family
        )
        within_weights = selected_meta.iloc[locations][
            "metric_weight_within_family"
        ].to_numpy(dtype=float)
        coefficients[locations] = (
            float(NEURO_FAMILY_WEIGHTS[family])
            / represented_family_weight
            * within_weights
            / float(within_weights.sum())
        )
    if not np.isclose(coefficients.sum(), 1.0):
        raise RuntimeError("Reduced metric coefficients do not sum to one")
    return selected, coefficients


def reduced_composite(
    values: np.ndarray, indices: Iterable[int], metadata: pd.DataFrame
) -> np.ndarray:
    selected, coefficients = reduced_metric_coefficients(indices, metadata)
    return np.tensordot(
        np.asarray(values, dtype=float)[..., selected],
        coefficients,
        axes=([-1], [0]),
    )


def _metric_label(metrics: tuple[str, ...]) -> str:
    return " + ".join(name.removesuffix("_score") for name in metrics)


def _metric_key(metrics: tuple[str, ...]) -> str:
    return "|".join(metrics)


def _subset_family_counts(
    indices: tuple[int, ...], metadata: pd.DataFrame
) -> dict[str, int]:
    selected_families = metadata.iloc[list(indices)]["family"]
    counts = selected_families.value_counts()
    return {
        family: int(counts.get(family, 0)) for family in NEURO_FAMILY_METRICS
    }


def _frequency_quantiles(
    values: np.ndarray,
    frequencies: np.ndarray,
    quantiles: tuple[float, ...] = (0.5, 0.025, 0.975),
) -> tuple[float, ...]:
    """NumPy-style linear quantiles for values represented by frequencies."""

    values = np.asarray(values, dtype=float)
    frequencies = np.asarray(frequencies, dtype=np.int64)
    finite = np.isfinite(values) & (frequencies > 0)
    if not np.any(finite):
        return tuple(float("nan") for _ in quantiles)
    order = np.argsort(values[finite], kind="stable")
    sorted_values = values[finite][order]
    sorted_frequencies = frequencies[finite][order]
    cumulative = np.cumsum(sorted_frequencies)
    total = int(cumulative[-1])
    results: list[float] = []
    for quantile in quantiles:
        position = float(quantile) * (total - 1)
        lower = int(np.floor(position))
        upper = int(np.ceil(position))
        lower_index = int(np.searchsorted(cumulative, lower, side="right"))
        upper_index = int(np.searchsorted(cumulative, upper, side="right"))
        fraction = position - lower
        results.append(
            float(
                sorted_values[lower_index] * (1.0 - fraction)
                + sorted_values[upper_index] * fraction
            )
        )
    return tuple(results)


def _bootstrap_patterns(
    bootstrap_indices: np.ndarray, n_splits: int
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.column_stack(
        [
            np.sum(bootstrap_indices == split_index, axis=1)
            for split_index in range(n_splits)
        ]
    )
    patterns, frequencies = np.unique(
        counts, axis=0, return_counts=True
    )
    return patterns.astype(float) / n_splits, frequencies.astype(np.int64)


def _all_subset_coefficients(
    metadata: pd.DataFrame,
) -> tuple[list[tuple[int, ...]], np.ndarray]:
    n_metrics = len(metadata)
    family_names = list(NEURO_FAMILY_METRICS)
    family_lookup = {family: index for index, family in enumerate(family_names)}
    metric_family = np.asarray(
        [family_lookup[name] for name in metadata["family"]], dtype=np.int64
    )
    metric_weights = metadata[
        "metric_weight_within_family"
    ].to_numpy(dtype=float)
    family_weights = np.asarray(
        [float(NEURO_FAMILY_WEIGHTS[name]) for name in family_names],
        dtype=float,
    )
    specifications = [
        indices
        for size in range(1, n_metrics + 1)
        for indices in itertools.combinations(range(n_metrics), size)
    ]
    coefficients = np.zeros((len(specifications), n_metrics), dtype=float)
    for subset_id, indices in enumerate(specifications):
        selected = np.asarray(indices, dtype=np.int64)
        selected_families = metric_family[selected]
        within_totals = np.bincount(
            selected_families,
            weights=metric_weights[selected],
            minlength=len(family_names),
        )
        represented = within_totals > 0
        represented_weight = float(family_weights[represented].sum())
        coefficients[subset_id, selected] = (
            family_weights[selected_families]
            / represented_weight
            * metric_weights[selected]
            / within_totals[selected_families]
        )
    if not np.allclose(coefficients.sum(axis=1), 1.0):
        raise RuntimeError("One or more reduced metric composites are invalid")
    return specifications, coefficients


def analyze_metric_subsets(
    *,
    models: list[str],
    metadata: pd.DataFrame,
    metric_values: np.ndarray,
    composite_values: np.ndarray,
    bootstrap_count: int,
    seed: int,
    min_median_spearman: float,
    min_top_preservation: float,
    min_pairwise_agreement: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    n_models, n_splits, n_metrics = metric_values.shape
    default_means = composite_values.mean(axis=1)
    default_ranks = average_ranks_descending(default_means)
    pair_i, pair_j = pair_indices(n_models)
    default_pair_signs = pair_order_signs(default_ranks, pair_i, pair_j)

    rng = np.random.default_rng(seed)
    bootstrap_indices = rng.integers(
        0, n_splits, size=(bootstrap_count, n_splits), dtype=np.int64
    )
    bootstrap_patterns, pattern_frequencies = _bootstrap_patterns(
        bootstrap_indices, n_splits
    )
    pattern_count = len(bootstrap_patterns)
    bootstrap_reference_ranks = np.broadcast_to(
        default_ranks, (pattern_count, n_models)
    )

    specifications, coefficients = _all_subset_coefficients(metadata)
    n_subsets = len(specifications)
    reduced_split_scores = np.tensordot(
        coefficients,
        metric_values,
        axes=([1], [2]),
    ).transpose(0, 1, 2)
    reduced_means = reduced_split_scores.mean(axis=2)
    reduced_ranks = average_ranks_descending(reduced_means, axis=1)
    deterministic_spearman = rowwise_spearman_from_ranks(
        np.broadcast_to(default_ranks, reduced_ranks.shape), reduced_ranks
    )
    reduced_pair_signs = pair_order_signs(reduced_ranks, pair_i, pair_j)
    deterministic_pairwise = np.mean(
        reduced_pair_signs == default_pair_signs[None, :], axis=1
    )
    deterministic_top = top_overlap(
        np.broadcast_to(default_ranks, reduced_ranks.shape), reduced_ranks
    )
    deterministic_exact = np.all(
        np.isclose(reduced_ranks, default_ranks[None, :]), axis=1
    )

    products = reduced_pair_signs * default_pair_signs[None, :]
    concordant = np.sum(products > 0, axis=1)
    discordant = np.sum(products < 0, axis=1)
    tied_reference_only = np.sum(
        (default_pair_signs[None, :] == 0) & (reduced_pair_signs != 0),
        axis=1,
    )
    tied_reduced_only = np.sum(
        (default_pair_signs[None, :] != 0) & (reduced_pair_signs == 0),
        axis=1,
    )
    kendall_denominator = np.sqrt(
        (concordant + discordant + tied_reference_only)
        * (concordant + discordant + tied_reduced_only)
    )
    deterministic_kendall = np.full(n_subsets, np.nan, dtype=float)
    valid_kendall = kendall_denominator > 0
    deterministic_kendall[valid_kendall] = (
        concordant[valid_kendall] - discordant[valid_kendall]
    ) / kendall_denominator[valid_kendall]

    median_rho = np.empty(n_subsets, dtype=float)
    rho_low = np.empty(n_subsets, dtype=float)
    rho_high = np.empty(n_subsets, dtype=float)
    top_probability = np.empty(n_subsets, dtype=float)
    mean_pairwise = np.empty(n_subsets, dtype=float)
    exact_probability = np.empty(n_subsets, dtype=float)
    batch_size = 4096
    frequency_weights = pattern_frequencies.astype(float)
    frequency_weights /= frequency_weights.sum()
    for start in range(0, n_subsets, batch_size):
        stop = min(start + batch_size, n_subsets)
        bootstrap_scores = np.einsum(
            "bms,ps->bpm",
            reduced_split_scores[start:stop],
            bootstrap_patterns,
            optimize=True,
        )
        bootstrap_ranks = average_ranks_descending(
            bootstrap_scores, axis=2
        )
        batch_count = stop - start
        bootstrap_spearman = rowwise_spearman_from_ranks(
            np.broadcast_to(
                default_ranks,
                (batch_count * pattern_count, n_models),
            ),
            bootstrap_ranks.reshape(-1, n_models),
        ).reshape(batch_count, pattern_count)
        for local_index in range(batch_count):
            (
                median_rho[start + local_index],
                rho_low[start + local_index],
                rho_high[start + local_index],
            ) = _frequency_quantiles(
                bootstrap_spearman[local_index], pattern_frequencies
            )

        bootstrap_pair_signs = pair_order_signs(
            bootstrap_ranks, pair_i, pair_j
        )
        per_pattern_pairwise = np.mean(
            bootstrap_pair_signs
            == default_pair_signs[None, None, :],
            axis=2,
        )
        mean_pairwise[start:stop] = (
            per_pattern_pairwise @ frequency_weights
        )
        batch_top = top_overlap(
            np.broadcast_to(
                bootstrap_reference_ranks,
                (batch_count, pattern_count, n_models),
            ).reshape(-1, n_models),
            bootstrap_ranks.reshape(-1, n_models),
        ).reshape(batch_count, pattern_count)
        top_probability[start:stop] = batch_top @ frequency_weights
        batch_exact = np.all(
            np.isclose(bootstrap_ranks, default_ranks[None, None, :]),
            axis=2,
        )
        exact_probability[start:stop] = batch_exact @ frequency_weights

    metric_names = metadata["metric"].tolist()
    subset_rows: list[dict[str, Any]] = []
    family_names = list(NEURO_FAMILY_METRICS)
    metric_families = metadata["family"].to_numpy(dtype=object)
    for subset_id, indices in enumerate(specifications):
        metrics = tuple(metric_names[index] for index in indices)
        family_counts = {
            family: sum(metric_families[index] == family for index in indices)
            for family in family_names
        }
        passes = bool(
            median_rho[subset_id] >= min_median_spearman
            and top_probability[subset_id] >= min_top_preservation
            and mean_pairwise[subset_id] >= min_pairwise_agreement
        )
        subset_rows.append(
            {
                "subset_id": subset_id,
                "subset_key": _metric_key(metrics),
                "subset_label": _metric_label(metrics),
                "subset_metrics": json.dumps(list(metrics)),
                "represented_families": json.dumps(
                    [
                        family
                        for family, count in family_counts.items()
                        if count
                    ]
                ),
                "subset_size": len(indices),
                "total_available_metrics": n_metrics,
                "percent_of_all_metrics": 100.0 * len(indices) / n_metrics,
                "spearman": float(deterministic_spearman[subset_id]),
                "kendall": float(deterministic_kendall[subset_id]),
                "pairwise_order_agreement": float(
                    deterministic_pairwise[subset_id]
                ),
                "top_model_preserved": bool(
                    deterministic_top[subset_id]
                ),
                "exact_order_preserved": bool(
                    deterministic_exact[subset_id]
                ),
                "bootstrap_median_spearman": float(median_rho[subset_id]),
                "bootstrap_spearman_ci95_low": float(rho_low[subset_id]),
                "bootstrap_spearman_ci95_high": float(rho_high[subset_id]),
                "bootstrap_top_model_probability": float(
                    top_probability[subset_id]
                ),
                "bootstrap_mean_pairwise_agreement": float(
                    mean_pairwise[subset_id]
                ),
                "bootstrap_exact_order_probability": float(
                    exact_probability[subset_id]
                ),
                "passes_stability": passes,
            }
        )
    return pd.DataFrame(subset_rows), bootstrap_indices


def stability_sort(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        [
            "bootstrap_median_spearman",
            "bootstrap_mean_pairwise_agreement",
            "bootstrap_top_model_probability",
            "bootstrap_exact_order_probability",
            "bootstrap_spearman_ci95_low",
            "subset_key",
        ],
        ascending=[False, False, False, False, False, True],
        kind="stable",
    )


def build_default_ranking(
    models: list[str], composite_values: np.ndarray
) -> pd.DataFrame:
    means = composite_values.mean(axis=1)
    ranks = average_ranks_descending(means)
    rows: list[dict[str, Any]] = []
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


def build_combination_model_scores(
    combinations: pd.DataFrame,
    models: list[str],
    metadata: pd.DataFrame,
    metric_values: np.ndarray,
    composite_values: np.ndarray,
) -> pd.DataFrame:
    metric_lookup = {
        metric: index for index, metric in enumerate(metadata["metric"])
    }
    default_means = composite_values.mean(axis=1)
    default_ranks = average_ranks_descending(default_means)
    rows: list[dict[str, Any]] = []
    for _, combination in combinations.iterrows():
        metrics = json.loads(str(combination["subset_metrics"]))
        indices = tuple(metric_lookup[metric] for metric in metrics)
        reduced_per_split = reduced_composite(
            metric_values, indices, metadata
        )
        reduced_means = reduced_per_split.mean(axis=1)
        reduced_ranks = average_ranks_descending(reduced_means)
        for model_index, model in enumerate(models):
            row: dict[str, Any] = {
                "combination_rank": int(combination["combination_rank"]),
                "subset_id": int(combination["subset_id"]),
                "subset_key": combination["subset_key"],
                "subset_label": combination["subset_label"],
                "subset_size": int(combination["subset_size"]),
                "model": model,
                "reduced_mean_score": float(reduced_means[model_index]),
                "reduced_rank": float(reduced_ranks[model_index]),
                "default_mean_composite": float(default_means[model_index]),
                "default_rank": float(default_ranks[model_index]),
            }
            for split_index in range(composite_values.shape[1]):
                row[f"split_{split_index + 1}_reduced_score"] = float(
                    reduced_per_split[model_index, split_index]
                )
                row[f"split_{split_index + 1}_default_composite"] = float(
                    composite_values[model_index, split_index]
                )
            rows.append(row)
    return pd.DataFrame(rows)


def build_family_breakdown(
    combinations: pd.DataFrame, metadata: pd.DataFrame
) -> pd.DataFrame:
    columns = [
        "combination_rank",
        "subset_id",
        "subset_key",
        "subset_label",
        "family",
        "selected_metrics",
        "selected_metric_count",
        "subset_metric_count",
        "percent_of_combination",
        "family_available_metric_count",
        "percent_of_family_available",
    ]
    available_counts = metadata.groupby("family", sort=False).size().to_dict()
    metric_to_family = metadata.set_index("metric")["family"].to_dict()
    rows: list[dict[str, Any]] = []
    for _, subset in combinations.iterrows():
        metrics = tuple(json.loads(str(subset["subset_metrics"])))
        subset_size = int(subset["subset_size"])
        for family in NEURO_FAMILY_METRICS:
            selected = [
                metric for metric in metrics if metric_to_family[metric] == family
            ]
            selected_count = len(selected)
            available_count = int(available_counts[family])
            rows.append(
                {
                    "combination_rank": int(subset["combination_rank"]),
                    "subset_id": int(subset["subset_id"]),
                    "subset_key": subset["subset_key"],
                    "subset_label": subset["subset_label"],
                    "family": family,
                    "selected_metrics": json.dumps(selected),
                    "selected_metric_count": selected_count,
                    "subset_metric_count": subset_size,
                    "percent_of_combination": (
                        100.0 * selected_count / subset_size
                    ),
                    "family_available_metric_count": available_count,
                    "percent_of_family_available": (
                        100.0 * selected_count / available_count
                    ),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def plot_metric_stability(
    results: pd.DataFrame, output_dir: Path, threshold: float
) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    passed = results["passes_stability"].to_numpy(dtype=bool)
    subset_sizes = results["subset_size"].to_numpy(dtype=int)
    x = subset_sizes.astype(float)
    for subset_size in np.unique(subset_sizes):
        locations = np.flatnonzero(subset_sizes == subset_size)
        if locations.size > 1:
            x[locations] += np.linspace(-0.22, 0.22, locations.size)
    colors = np.where(passed, "#2A9D8F", "#9AA0A6")
    ax.scatter(
        x,
        results["bootstrap_median_spearman"],
        c=colors,
        s=np.where(passed, 55, 24),
        edgecolors=np.where(passed, "white", "none"),
        linewidths=np.where(passed, 0.6, 0.0),
        alpha=np.where(passed, 1.0, 0.45),
        rasterized=True,
        zorder=3,
    )
    ax.axhline(
        threshold,
        color="#C44536",
        linestyle="--",
        linewidth=1.4,
        label=f"Median Spearman threshold ({threshold:.2f})",
    )
    sizes_evaluated = sorted(set(subset_sizes))
    ax.set_xticks(sizes_evaluated)
    ax.set_xlabel("Number of metrics in candidate subset")
    ax.set_ylabel("Bootstrap median Spearman")
    ax.set_ylim(-1.03, 1.03)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.tight_layout()
    svg = output_dir / "metric_combination_stability.svg"
    png = output_dir / "metric_combination_stability.png"
    fig.savefig(svg, format="svg", bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return svg, png


def plot_family_breakdown(
    breakdown: pd.DataFrame, primary_subset_id: int, output_dir: Path
) -> tuple[Path, Path]:
    selected = breakdown[breakdown["subset_id"] == primary_subset_id]
    labels = [
        family.replace("_", " ").title() for family in selected["family"]
    ]
    x = np.arange(len(selected))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.0, 4.7))
    ax.bar(
        x - width / 2,
        selected["percent_of_combination"],
        width,
        color="#3E7CB1",
        label="% of selected combination",
    )
    ax.bar(
        x + width / 2,
        selected["percent_of_family_available"],
        width,
        color="#E07A5F",
        label="% of family metrics retained",
    )
    for position, (_, row) in enumerate(selected.iterrows()):
        ax.text(
            position - width / 2,
            float(row["percent_of_combination"]) + 1.5,
            (
                f"{int(row['selected_metric_count'])}/"
                f"{int(row['subset_metric_count'])}"
            ),
            ha="center",
            va="bottom",
            fontsize=8,
        )
        ax.text(
            position + width / 2,
            float(row["percent_of_family_available"]) + 1.5,
            (
                f"{int(row['selected_metric_count'])}/"
                f"{int(row['family_available_metric_count'])}"
            ),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Percentage")
    ax.set_ylim(0, 112)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.tight_layout()
    svg = output_dir / "best_metric_combination_family_breakdown.svg"
    png = output_dir / "best_metric_combination_family_breakdown.png"
    fig.savefig(svg, format="svg", bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return svg, png


def plot_rank_comparison(
    default_ranking: pd.DataFrame,
    model_scores: pd.DataFrame,
    primary_subset: pd.Series,
    output_dir: Path,
) -> tuple[Path, Path]:
    model_order = default_ranking["model"].tolist()
    default = (
        default_ranking.set_index("model")
        .loc[model_order, "default_rank"]
        .to_numpy(dtype=float)
    )
    reduced = (
        model_scores[
            model_scores["subset_id"] == int(primary_subset["subset_id"])
        ]
        .set_index("model")
        .loc[model_order, "reduced_rank"]
        .to_numpy(dtype=float)
    )
    matrix = np.column_stack([default, reduced])
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    image = ax.imshow(
        matrix,
        cmap="viridis_r",
        vmin=1,
        vmax=max(1, len(model_order)),
        aspect="auto",
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        [
            "Default composite",
            (
                "Best combination\n"
                f"({int(primary_subset['subset_size'])} metrics)"
            ),
        ]
    )
    ax.set_yticks(np.arange(len(model_order)))
    ax.set_yticklabels(model_order)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column,
                row,
                _format_rank(float(matrix[row, column])),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > len(model_order) / 2 else "black",
                fontsize=8,
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.04)
    colorbar.set_label("Rank (1 = best)")
    ax.set_title("Default and best-combination model ranks")
    fig.tight_layout()
    svg = output_dir / "best_metric_combination_rank_comparison.svg"
    png = output_dir / "best_metric_combination_rank_comparison.png"
    fig.savefig(svg, format="svg", bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return svg, png


def print_terminal_summary(
    default_ranking: pd.DataFrame,
    top_combinations: pd.DataFrame,
    breakdown: pd.DataFrame,
) -> None:
    print("\nDefault model ranking (mean composite over four splits):")
    for _, row in default_ranking.iterrows():
        print(
            f"  {_format_rank(float(row['default_rank']))}. "
            f"{row['model']}: {float(row['default_mean_composite']):.6f}"
        )

    print("\nBest-ranked metric combinations:")
    for _, subset in top_combinations.iterrows():
        size = int(subset["subset_size"])
        total = int(subset["total_available_metrics"])
        print(
            f"\n  {int(subset['combination_rank'])}. "
            f"{subset['subset_label']} ({size}/{total} metrics, "
            f"{float(subset['percent_of_all_metrics']):.2f}%): median rho="
            f"{float(subset['bootstrap_median_spearman']):.3f}, "
            f"top={float(subset['bootstrap_top_model_probability']):.3f}, "
            f"pairwise="
            f"{float(subset['bootstrap_mean_pairwise_agreement']):.3f}"
        )
        family_rows = breakdown[
            breakdown["subset_id"] == int(subset["subset_id"])
        ]
        for _, row in family_rows.iterrows():
            print(
                f"      {str(row['family']).replace('_', ' ').title()}: "
                f"{int(row['selected_metric_count'])}/{size} "
                f"({float(row['percent_of_combination']):.2f}% of combination); "
                f"{int(row['selected_metric_count'])}/"
                f"{int(row['family_available_metric_count'])} "
                f"({float(row['percent_of_family_available']):.2f}% of family)"
            )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    cache_path = args.input_cache.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        models,
        metadata,
        metric_values,
        composite_values,
        cache_payload,
    ) = load_metric_score_cache(cache_path)
    available_models = list(models)
    models, metric_values, composite_values, excluded_models = exclude_models(
        models,
        metric_values,
        composite_values,
        args.exclude_models,
    )
    print(f"Input cache: {cache_path}")
    print(
        f"Models available in cache ({len(available_models)}): "
        f"{', '.join(available_models)}"
    )
    if excluded_models:
        print(
            f"Models excluded ({len(excluded_models)}): "
            f"{', '.join(excluded_models)}"
        )
    print(f"Models analyzed ({len(models)}): {', '.join(models)}")
    print(
        f"Canonical metrics: {len(metadata)} across "
        f"{len(NEURO_FAMILY_METRICS)} families"
    )
    print(f"Metric combinations to evaluate: {2 ** len(metadata) - 1:,}")

    results, bootstrap_indices = analyze_metric_subsets(
        models=models,
        metadata=metadata,
        metric_values=metric_values,
        composite_values=composite_values,
        bootstrap_count=args.bootstrap_count,
        seed=args.seed,
        min_median_spearman=args.min_median_spearman,
        min_top_preservation=args.min_top_preservation,
        min_pairwise_agreement=args.min_pairwise_agreement,
    )
    ordered_results = stability_sort(results).reset_index(drop=True)
    ordered_results.insert(
        0,
        "combination_rank",
        np.arange(1, len(ordered_results) + 1, dtype=int),
    )
    top_count = min(args.top_combinations, len(ordered_results))
    top_combinations = ordered_results.head(top_count).copy()
    primary_subset = top_combinations.iloc[0]
    model_scores = build_combination_model_scores(
        top_combinations,
        models,
        metadata,
        metric_values,
        composite_values,
    )
    breakdown = build_family_breakdown(top_combinations, metadata)
    default_ranking = build_default_ranking(models, composite_values)

    results_path = output_dir / "ranked_metric_combinations.csv"
    model_scores_path = output_dir / "top_combination_model_scores.csv"
    top_path = output_dir / "top_metric_combinations.csv"
    breakdown_path = output_dir / "top_combinations_family_breakdown.csv"
    metadata_path = output_dir / "canonical_metric_metadata.csv"
    ranking_path = output_dir / "default_model_ranking.csv"
    ordered_results.to_csv(results_path, index=False)
    model_scores.to_csv(model_scores_path, index=False)
    top_combinations.to_csv(top_path, index=False)
    breakdown.to_csv(breakdown_path, index=False)
    metadata.to_csv(metadata_path, index=False)
    default_ranking.to_csv(ranking_path, index=False)

    stability_svg, stability_png = plot_metric_stability(
        results, output_dir, args.min_median_spearman
    )
    family_svg, family_png = plot_family_breakdown(
        breakdown, int(primary_subset["subset_id"]), output_dir
    )
    ranks_svg, ranks_png = plot_rank_comparison(
        default_ranking, model_scores, primary_subset, output_dir
    )

    results_json_path = output_dir / "metric_combination_ranking_results.json"
    payload = {
        "config": {
            "input_cache": str(cache_path),
            "output_dir": str(output_dir),
            "available_models": available_models,
            "excluded_models": excluded_models,
            "bootstrap_count": args.bootstrap_count,
            "seed": args.seed,
            "top_combinations_in_detailed_outputs": top_count,
            "n_splits": 4,
            "tie_method": "average ranks",
            "search_method": (
                "exhaustively evaluate and rank all 2^N - 1 non-empty metric "
                "combinations; combination size is not a ranking criterion"
            ),
            "ranking_criteria": (
                "descending bootstrap median Spearman, mean pairwise-order "
                "agreement, top-model preservation, exact-order probability, "
                "Spearman 95% CI lower bound; subset key breaks remaining ties"
            ),
            "reduced_composite_method": (
                "renormalize selected metric weights within represented families, "
                "then renormalize canonical family weights across represented "
                "families"
            ),
            "bootstrap_method": (
                "resample the four split indices with replacement for each "
                "reduced subset, then compare its ranking with the fixed default "
                "ranking defined by the original four-split mean composite; "
                "identical bootstrap split-count patterns are collapsed and "
                "weighted by their sampled frequency"
            ),
            "stability_thresholds": {
                "min_median_spearman": args.min_median_spearman,
                "min_top_preservation": args.min_top_preservation,
                "min_pairwise_agreement": args.min_pairwise_agreement,
            },
        },
        "models": models,
        "cache_signature": cache_payload.get("_cache_signature"),
        "canonical_metrics": _records(metadata),
        "bootstrap_split_indices": bootstrap_indices.tolist(),
        "default_ranking": _records(default_ranking),
        "evaluated_combination_count": len(ordered_results),
        "top_metric_combinations": _records(top_combinations),
        "best_combination": _json_safe(primary_subset.to_dict()),
        "top_combinations_family_breakdown": _records(breakdown),
        "outputs": {
            "ranked_metric_combinations_csv": str(results_path),
            "top_combination_model_scores_csv": str(model_scores_path),
            "top_metric_combinations_csv": str(top_path),
            "top_combinations_family_breakdown_csv": str(breakdown_path),
            "canonical_metric_metadata_csv": str(metadata_path),
            "default_model_ranking_csv": str(ranking_path),
            "combination_stability_svg": str(stability_svg),
            "combination_stability_png": str(stability_png),
            "family_breakdown_svg": str(family_svg),
            "family_breakdown_png": str(family_png),
            "rank_comparison_svg": str(ranks_svg),
            "rank_comparison_png": str(ranks_png),
        },
    }
    results_json_path.write_text(
        json.dumps(_json_safe(payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print_terminal_summary(
        default_ranking,
        top_combinations.head(min(10, top_count)),
        breakdown,
    )
    print(f"\nDetailed results: {results_json_path}")
    print(f"Plots and CSVs: {output_dir}")


if __name__ == "__main__":
    main()
