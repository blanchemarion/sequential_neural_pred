#!/usr/bin/env python3
"""
Four-split NethoBench scoring for MLP-2p with a fixed region exclusion mask.

The same regions are removed from ground truth and predictions before any
sequence split is scored. By default, the first 90 context steps are also
removed from both arrays, leaving the 240-step forecast window.

Run from the repository root:

    python src/visualize/neuro_subscores_from_npy_2p_4split_filtered_regions.py

Default inputs:

    evaluation_results/mlp_2p/long_ground_truth_MLP_2p.npy
    evaluation_results/mlp_2p/long_predictions_MLP_2p.npy

Default excluded zero-based region indices:

    50, 51, 52, 56, 77, 87
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_NETHOBENCH_ROOT = _REPO_ROOT / "nethobench"
if str(_NETHOBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_NETHOBENCH_ROOT))

from nethobench.neuro.metrics.composites import calculate_neuro_composites


DEFAULT_EXCLUDED_REGIONS = (50, 51, 52, 56, 77, 87)
FAMILY_SCORES = {
    "family_distribution": "Distribution",
    "family_temporal_spectral": "Temporal",
    "family_relational": "Relational",
    "family_geometry": "Geometry",
    "family_state_dynamics": "State dynamics",
    "FINAL_COMPOSITE_SCORE": "Composite",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_REPO_ROOT / "evaluation_results" / "mlp_2p",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            _REPO_ROOT
            / "output"
            / "neuro_subscores_from_npy_2p_4split_filtered_regions"
        ),
    )
    parser.add_argument(
        "--ground-truth-file",
        default="long_ground_truth_MLP_2p.npy",
    )
    parser.add_argument(
        "--prediction-file",
        default="long_predictions_MLP_2p.npy",
    )
    parser.add_argument("--context-steps", type=int, default=90)
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument(
        "--excluded-regions",
        type=int,
        nargs="+",
        default=list(DEFAULT_EXCLUDED_REGIONS),
        metavar="INDEX",
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


def apply_fixed_region_filter(
    gt: np.ndarray,
    pred: np.ndarray,
    excluded_regions: list[int] | tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove exactly the same zero-based region indices from GT and prediction."""
    if gt.ndim != 3 or pred.ndim != 3:
        raise ValueError(
            f"Expected [sequence,time,region] arrays, got {gt.shape} and {pred.shape}"
        )
    if gt.shape != pred.shape:
        raise ValueError(f"GT/prediction shape mismatch: {gt.shape} vs {pred.shape}")

    excluded = np.asarray(sorted(set(int(i) for i in excluded_regions)), dtype=int)
    if excluded.size and (
        int(excluded.min()) < 0 or int(excluded.max()) >= gt.shape[2]
    ):
        raise ValueError(
            f"Excluded indices {excluded.tolist()} outside 0..{gt.shape[2] - 1}"
        )

    keep_mask = np.ones(gt.shape[2], dtype=bool)
    keep_mask[excluded] = False
    if not keep_mask.any():
        raise ValueError("Region exclusion removed every region")
    return gt[:, :, keep_mask], pred[:, :, keep_mask], np.flatnonzero(keep_mask)


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


def aggregate_scores(
    split_scores: list[dict[str, float]],
) -> tuple[dict[str, float], dict[str, float]]:
    keys = sorted({key for scores in split_scores for key in scores})
    means: dict[str, float] = {}
    sems: dict[str, float] = {}
    for key in keys:
        means[key], sems[key] = mean_sem(
            [float(scores.get(key, np.nan)) for scores in split_scores]
        )
    return means, sems


def validate_split_variance(
    gt: np.ndarray,
    pred: np.ndarray,
    splits: list[np.ndarray],
    kept_regions: np.ndarray,
    tolerance: float = 1e-6,
) -> None:
    for split_number, indices in enumerate(splits, start=1):
        for label, array in (("GT", gt), ("prediction", pred)):
            std = np.nanstd(array[indices], axis=(0, 1))
            bad_local = np.flatnonzero(std < tolerance)
            if bad_local.size:
                bad_original = kept_regions[bad_local]
                raise ValueError(
                    f"Split {split_number} {label} still has near-zero-variance "
                    f"regions. Filtered indices={bad_local.tolist()}, original "
                    f"indices={bad_original.tolist()}"
                )


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value


def save_family_plot(
    means: dict[str, float],
    sems: dict[str, float],
    output_path: Path,
) -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "svg.fonttype": "none",
            "axes.linewidth": 0.8,
            "font.size": 9,
        }
    )
    keys = list(FAMILY_SCORES)
    labels = [FAMILY_SCORES[key] for key in keys]
    values = np.asarray([means.get(key, np.nan) for key in keys], dtype=float)
    errors = np.asarray([sems.get(key, np.nan) for key in keys], dtype=float)
    errors = np.nan_to_num(errors, nan=0.0)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bars = ax.bar(
        np.arange(len(keys)),
        values,
        yerr=errors,
        capsize=3,
        color="#A23B72",
        edgecolor="black",
        linewidth=0.6,
        alpha=0.9,
    )
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Score (mean ± SEM over four splits)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_title("MLP-2p NethoBench scores after fixed region exclusion")
    for bar, value in zip(bars, values):
        if np.isfinite(value):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(value + 0.035, 0.98),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> dict[str, Path]:
    args = parse_args(argv)
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_path = data_dir / args.ground_truth_file
    pred_path = data_dir / args.prediction_file
    gt_raw = np.load(gt_path, allow_pickle=False)
    pred_raw = np.load(pred_path, allow_pickle=False)
    if gt_raw.shape != pred_raw.shape:
        raise ValueError(
            f"GT/prediction shape mismatch: {gt_raw.shape} vs {pred_raw.shape}"
        )
    if args.context_steps < 0 or args.context_steps >= gt_raw.shape[1]:
        raise ValueError(
            f"context_steps={args.context_steps} invalid for {gt_raw.shape[1]} steps"
        )

    gt_forecast = gt_raw[:, args.context_steps :, :]
    pred_forecast = pred_raw[:, args.context_steps :, :]
    gt, pred, kept_regions = apply_fixed_region_filter(
        gt_forecast,
        pred_forecast,
        args.excluded_regions,
    )
    splits = sequence_split_indices(gt.shape[0], args.n_splits)
    validate_split_variance(gt, pred, splits, kept_regions)

    print(f"Ground truth: {gt_path}")
    print(f"Prediction:   {pred_path}")
    print(f"Raw shape: {gt_raw.shape}")
    print(f"Forecast-only shape before filtering: {gt_forecast.shape}")
    print(
        f"Excluded original regions: "
        f"{sorted(set(int(i) for i in args.excluded_regions))}"
    )
    print(f"Kept {len(kept_regions)} regions: scored shape {gt.shape}")

    split_scores: list[dict[str, float]] = []
    split_rows: list[dict[str, float | int | str]] = []
    for split_number, indices in enumerate(splits, start=1):
        print(
            f"Split {split_number}/{args.n_splits}: sequences "
            f"{int(indices[0])}..{int(indices[-1])} (n={len(indices)})"
        )
        scores = calculate_neuro_composites(gt[indices], pred[indices])
        split_scores.append(scores)
        split_rows.append(
            {
                "row": f"split_{split_number}",
                "n_sequences": int(len(indices)),
                **{key: float(scores.get(key, np.nan)) for key in FAMILY_SCORES},
            }
        )

    means, sems = aggregate_scores(split_scores)
    split_rows.extend(
        [
            {
                "row": "mean",
                "n_sequences": int(gt.shape[0]),
                **{key: means.get(key, np.nan) for key in FAMILY_SCORES},
            },
            {
                "row": "sem",
                "n_sequences": int(args.n_splits),
                **{key: sems.get(key, np.nan) for key in FAMILY_SCORES},
            },
        ]
    )

    json_path = output_dir / "scores_MLP_2p_filtered_regions_4split.json"
    family_csv_path = output_dir / "family_scores_MLP_2p_filtered_regions_4split.csv"
    submetric_csv_path = (
        output_dir / "submetric_scores_MLP_2p_filtered_regions_4split.csv"
    )
    plot_path = output_dir / "family_scores_MLP_2p_filtered_regions_4split.svg"

    payload = {
        "model": "MLP_2p",
        "input": {
            "ground_truth": gt_path,
            "prediction": pred_path,
            "raw_shape": gt_raw.shape,
        },
        "scoring": {
            "context_steps_dropped_from_both_arrays": int(args.context_steps),
            "forecast_steps": int(gt.shape[1]),
            "n_splits": int(args.n_splits),
            "split_sizes": [int(len(indices)) for indices in splits],
            "excluded_original_region_indices": sorted(
                set(int(i) for i in args.excluded_regions)
            ),
            "kept_original_region_indices": kept_regions,
            "n_regions_before": int(gt_raw.shape[2]),
            "n_regions_after": int(gt.shape[2]),
        },
        "mean": means,
        "sem": sems,
        "per_split": split_scores,
        "reproduction_command": (
            "python src/visualize/"
            "neuro_subscores_from_npy_2p_4split_filtered_regions.py"
        ),
    }
    json_path.write_text(
        json.dumps(json_safe(payload), indent=2), encoding="utf-8"
    )
    pd.DataFrame(split_rows).set_index("row").to_csv(
        family_csv_path, float_format="%.8f"
    )
    pd.DataFrame(
        {
            "mean": means,
            "sem": sems,
        }
    ).to_csv(submetric_csv_path, float_format="%.8f")
    save_family_plot(means, sems, plot_path)

    print("\nFamily/composite mean ± SEM:")
    for key, label in FAMILY_SCORES.items():
        print(
            f"  {label}: {means.get(key, np.nan):.6f} ± "
            f"{sems.get(key, np.nan):.6f}"
        )
    print(f"\nSaved outputs under: {output_dir}")
    return {
        "json": json_path,
        "family_csv": family_csv_path,
        "submetric_csv": submetric_csv_path,
        "plot": plot_path,
    }


if __name__ == "__main__":
    main()
