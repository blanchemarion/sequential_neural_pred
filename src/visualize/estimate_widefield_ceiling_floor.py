#!/usr/bin/env python3
"""
Estimate an empirical Nethobench ceiling and shuffle floors for neural recordings.

For references matched to all 222 original evaluation targets, use
``matched_widefield_references.py``. This standalone empirical sampler supports
other pair counts and is not matched to the original target set by default.

Supported inputs are:

- processed widefield NCT-block arrays from ``src/prepare/prepare.py``, with
  shape ``[sequenceID, subsequence, region, time]``;
- raw 2p trace CSVs with no header, rows = neurons, columns = timepoints, as
  documented by ``src/prepare/prepare_2p_traces.py``.

Consecutive subsequences from one sequenceID are concatenated before sampling.
Each repetition draws two disjoint real windows from the same recording. The
first window is ground truth, the second is the split-half prediction, and
temporal-only and time+region shuffles provide two matched floors.

All scores come from Nethobench's existing ``calculate_neuro_composites``
implementation, which is the in-memory scoring core called by
``compute_neuro_scores``. Calling the core directly is necessary because one
repetition contains one sequence, while the CSV input validator requires at
least two sequences.

Example (defaults: 720 steps, 222 independently sampled pairs, four batches,
seed 101). This mode matches the model evaluation's count but draws different
targets. To hold the original 222 evaluation targets fixed across Monte Carlo
draws, run ``matched_widefield_references.py`` instead. Standalone results are
saved to ``output/widefield_empirical_ceiling_floor`` by default.

    python src/visualize/estimate_widefield_ceiling_floor.py

Run the same analysis on the 2p recording:

    python src/visualize/estimate_widefield_ceiling_floor.py \
        --data-path data_raw/2p_traces/Data_Valence_FULL_A_3_1189451_Session_10_C_dec_10Hz.csv

Restrict all sampled windows to one sequenceID and one reproducible RNG seed:

    python src/visualize/estimate_widefield_ceiling_floor.py \
        --sequence-id 0 --seed 101

The paired ceiling-window variance-contraction dose-response runs by default.
Disable it when only the ceiling/floor analysis is needed:

    python src/visualize/estimate_widefield_ceiling_floor.py \
        --no-variance-contraction-dose-response

Override the retained-variance ratios (rho=1 is required for verification):

    python src/visualize/estimate_widefield_ceiling_floor.py \
        --variance-contraction-dose-response \
        --variance-contraction-rhos 1.0 0.5 0.25 0.0

The default widefield input and model-cache paths are:

    data_processed/data100_ba16.npy
    output/neuro_subscores_from_npy_merged_4split_3seeds_new/
        scores_cache_90_810_4split_3seeds.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from cns_plotting import setup_cnsplots_style


def setup_plot_style() -> None:
    """Apply the same cnsplots-based publication defaults as peer figures."""

    setup_cnsplots_style(
        {
            "svg.fonttype": "none",
            "axes.linewidth": 0.8,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )


setup_plot_style()


REPO_ROOT = Path(__file__).resolve().parents[2]
NETHOBENCH_ROOT = REPO_ROOT / "nethobench"
if not (NETHOBENCH_ROOT / "nethobench" / "__init__.py").is_file():
    raise RuntimeError(
        f"Required nethobench checkout not found: {NETHOBENCH_ROOT}"
    )
nb_path = str(NETHOBENCH_ROOT.resolve())
if nb_path in sys.path:
    sys.path.remove(nb_path)
sys.path.insert(0, nb_path)

# Avoid joblib's platform-specific physical-core probe warning; Nethobench's
# numerical results do not depend on this worker-count hint.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings(
    "ignore",
    message=r"Could not find the number of physical cores.*",
    category=UserWarning,
    module=r"joblib\.externals\.loky\.backend\.context",
)

import nethobench
from nethobench.neuro.metrics.composites import calculate_neuro_composites
from nethobench.neuro.metrics import additional as nethobench_additional_metrics

_RESOLVED_NETHOBENCH_PATH = Path(nethobench.__file__).resolve()
if NETHOBENCH_ROOT.resolve() not in _RESOLVED_NETHOBENCH_PATH.parents:
    raise RuntimeError(
        "Resolved Nethobench outside nethobench: "
        f"{_RESOLVED_NETHOBENCH_PATH}"
    )


CEILING = "split_half_ceiling"
TEMPORAL_FLOOR = "temporal_shuffle_floor"
TIME_REGION_FLOOR = "time_and_region_shuffle_floor"
CONDITIONS = (CEILING, TEMPORAL_FLOOR, TIME_REGION_FLOOR)
CONDITION_LABELS = {
    CEILING: "Split-half ceiling",
    TEMPORAL_FLOOR: "Temporal-shuffle floor",
    TIME_REGION_FLOOR: "Time + region shuffle floor",
}
CONDITION_COLORS = {
    CEILING: "#3E7CB1",
    TEMPORAL_FLOOR: "#E6AB02",
    TIME_REGION_FLOOR: "#7A7A7A",
}

DEFAULT_VARIANCE_CONTRACTION_RHOS = (1.0, 0.50, 0.32, 0.20, 0.09, 0.06, 0.0)

FAMILY_COLUMNS = [
    "family_distribution",
    "family_temporal_spectral",
    "family_relational",
    "family_geometry",
    "family_state_dynamics",
]
PLOT_SCORE_COLUMNS = FAMILY_COLUMNS + ["FINAL_COMPOSITE_SCORE"]
PLOT_SCORE_LABELS = {
    "family_distribution": "Distribution",
    "family_temporal_spectral": "Temporal\nspectral",
    "family_relational": "Relational",
    "family_geometry": "Geometry",
    "family_state_dynamics": "State\ndynamics",
    "FINAL_COMPOSITE_SCORE": "Composite",
}

# Matches the grouped-bar order and colors in the existing four-split script.
MODEL_ORDER = [
    "VAR",
    "1_step",
    "SSM",
    "RNN",
    "AR",
    "TF",
    "TF_QL_0.08_KL_0.02",
    "sequifier",
]
MODEL_COLORS = {
    "VAR": "#3E7CB1",
    "1_step": "#7A7A7A",
    "AR": "#2AA876",
    "TF": "#A23B72",
    "TF_QL_0.08_KL_0.02": "#6A4C93",
    "sequifier": "#C46410",
    "SSM": "#D95F02",
    "RNN": "#E6AB02",
}

SAMPLE_COLUMNS = [
    "repetition",
    "batch",
    "sequence_id",
    "reference_start",
    "reference_end_exclusive",
    "ceiling_start",
    "ceiling_end_exclusive",
    "temporal_shuffle_seed",
    "time_region_shuffle_seed",
]
SCORE_META_COLUMNS = SAMPLE_COLUMNS + ["condition"]


@dataclass(frozen=True)
class EligibleSession:
    """Valid window starts for one sequence axis entry.

    ``valid_starts=None`` means every start in ``[0, max_start]`` is valid.
    Otherwise only values in ``valid_starts`` contain finite data for the full
    requested window.
    """

    sequence_id: int
    total_timesteps: int
    max_start: int
    valid_starts: np.ndarray | None
    eligible_reference_starts: np.ndarray | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate an empirical Nethobench ceiling and shuffle floors for "
            "processed widefield arrays or raw 2p trace CSVs."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=REPO_ROOT /"data_processed" / "data100_ba16.npy", # / "data_raw" / "2p_traces" / "Data_Valence_FULL_A_3_1189451_Session_10_F_dff_10Hz.csv", #
        help=(
            "Processed [sequence, subsequence, region, time] NumPy array, or a "
            "headerless 2p CSV with rows=neurons and columns=timepoints."
        ),
    )
    parser.add_argument(
        "--input-format",
        choices=("auto", "widefield-npy", "2p-csv"),
        default="auto",
        help="Input layout. 'auto' infers it from the .npy or .csv suffix.",
    )
    parser.add_argument(
        "--model-cache",
        type=Path,
        default=(
            REPO_ROOT
            / "output"
            / "neuro_subscores_from_npy_merged_4split_3seeds_new"
            / "scores_cache_90_810_4split_3seeds.json"
        ),
        help=(
            "Four-split Nethobench model-score cache. Three-training-seed "
            "caches are averaged across training seeds within each split."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for CSV, JSON, and plot outputs. Defaults to the legacy "
            "widefield directory for .npy input and a recording-specific 2p "
            "directory for .csv input."
        ),
    )
    parser.add_argument("--window-length", type=int, default=720)
    parser.add_argument(
        "--repetitions",
        type=int,
        default=222,
        help="Number of real-window pairs (default: 222, matching model evaluation).",
    )
    parser.add_argument(
        "--sequence-id",
        type=int,
        default=None,
        help=(
            "Restrict every sampled ceiling/floor window to this zero-based "
            "sequenceID. By default, repetitions sample across all eligible "
            "sequences."
        ),
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=4,
        help="Contiguous repetition groups used for four-split-style error bars.",
    )
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument(
        "--expected-regions",
        type=int,
        default=None,
        help=(
            "Fail unless this many regions/neurons are present. By default, "
            "widefield .npy input expects 16 and 2p CSV input accepts any count."
        ),
    )
    parser.add_argument(
        "--clip-normalized",
        action="store_true",
        help="Clip normalized model scores to [0,1]. Raw values are always retained.",
    )
    parser.add_argument(
        "--skip-model-normalization",
        action="store_true",
        help=(
            "Compute empirical references only; do not load or plot model "
            "scores. This is already the default for 2p CSV input."
        ),
    )
    parser.add_argument(
        "--normalize-model-scores",
        action="store_true",
        help=(
            "Opt in to model-cache normalization for 2p CSV input. Use only "
            "with a model cache whose scores are scientifically comparable to "
            "this 2p recording. Widefield .npy input keeps its legacy default "
            "of normalizing unless --skip-model-normalization is set."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume missing condition/repetition rows from an existing compatible CSV.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="Rewrite repetition_scores.csv after this many completed repetitions.",
    )
    parser.add_argument(
        "--variance-contraction-dose-response",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run the paired ceiling-window retained-variance dose-response "
            "(enabled by default). Use --no-variance-contraction-dose-response "
            "to skip it."
        ),
    )
    parser.add_argument(
        "--variance-contraction-rhos",
        type=float,
        nargs="+",
        default=list(DEFAULT_VARIANCE_CONTRACTION_RHOS),
        metavar="RHO",
        help=(
            "Retained-variance ratios for the optional dose-response. Values "
            "must be unique, lie in [0,1], and include 1.0."
        ),
    )
    parser.add_argument(
        "--variance-contraction-bootstrap-count",
        type=int,
        default=10_000,
        help=(
            "Paired bootstrap draws over joint batch scores for the optional "
            "variance-contraction dose-response."
        ),
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.window_length <= 0:
        raise ValueError("--window-length must be positive")
    if args.repetitions <= 0:
        raise ValueError("--repetitions must be positive")
    if args.sequence_id is not None and args.sequence_id < 0:
        raise ValueError("--sequence-id must be a non-negative integer")
    if args.batches <= 0:
        raise ValueError("--batches must be positive")
    if args.batches > args.repetitions:
        raise ValueError("--batches cannot exceed --repetitions")
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be positive")
    if args.expected_regions is not None and args.expected_regions <= 0:
        raise ValueError("--expected-regions must be positive when provided")
    if args.skip_model_normalization and args.normalize_model_scores:
        raise ValueError(
            "--skip-model-normalization and --normalize-model-scores are "
            "mutually exclusive"
        )
    rhos = np.asarray(args.variance_contraction_rhos, dtype=float)
    if rhos.ndim != 1 or rhos.size == 0 or not np.isfinite(rhos).all():
        raise ValueError("--variance-contraction-rhos must contain finite values")
    if np.any((rhos < 0.0) | (rhos > 1.0)):
        raise ValueError("--variance-contraction-rhos values must lie in [0,1]")
    if np.unique(rhos).size != rhos.size:
        raise ValueError("--variance-contraction-rhos values must be unique")
    if not np.any(np.isclose(rhos, 1.0, atol=0.0, rtol=0.0)):
        raise ValueError(
            "--variance-contraction-rhos must include 1.0 so the original "
            "ceiling can be reproduced and checked"
        )
    if args.variance_contraction_bootstrap_count <= 0:
        raise ValueError(
            "--variance-contraction-bootstrap-count must be positive"
        )


def _resolve_input_format(data_path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    suffix = data_path.suffix.lower()
    if suffix == ".npy":
        return "widefield-npy"
    if suffix == ".csv":
        return "2p-csv"
    raise ValueError(
        f"Cannot infer input format from {data_path.name!r}; use "
        "--input-format widefield-npy or --input-format 2p-csv"
    )


def load_analysis_data(
    data_path: Path, input_format: str
) -> tuple[np.ndarray, dict[str, Any], Path | None, list[int], list[str]]:
    """Load either native NCT blocks or a raw neuron-by-time 2p CSV.

    The returned array always follows
    ``[sequence, subsequence, region_or_neuron, time]`` so the sampling and
    scoring code has one canonical representation.
    """

    if input_format == "widefield-npy":
        data = np.load(data_path, mmap_mode="r", allow_pickle=False)
        if data.ndim != 4:
            raise ValueError(
                "Expected processed data with shape "
                f"[sequence,subsequence,region,time], got {data.shape}"
            )
        metadata_path = data_path.with_name(
            f"{data_path.stem}_metadata.json"
        )
        input_metadata: dict[str, Any] = {}
        if metadata_path.is_file():
            input_metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
            metadata_shape = input_metadata.get("shape")
            if (
                metadata_shape is not None
                and list(metadata_shape) != list(data.shape)
            ):
                raise ValueError(
                    f"Metadata shape {metadata_shape} does not match data "
                    f"shape {list(data.shape)} in {metadata_path}"
                )
            region_names = input_metadata.get("region_names")
            if (
                region_names is not None
                and len(region_names) != data.shape[2]
            ):
                raise ValueError(
                    f"Metadata has {len(region_names)} region names but data "
                    f"has {data.shape[2]} regions"
                )
        return (
            data,
            input_metadata,
            metadata_path if metadata_path.is_file() else None,
            list(data.shape),
            ["sequenceID", "subsequence", "region", "time"],
        )

    if input_format == "2p-csv":
        matrix = pd.read_csv(
            data_path, header=None, dtype=np.float32
        ).to_numpy(copy=False)
        if matrix.ndim != 2:
            raise ValueError(
                f"Expected a 2D 2p trace matrix, got shape {matrix.shape}"
            )
        if matrix.shape[0] < 1 or matrix.shape[1] < 2:
            raise ValueError(
                f"Unexpectedly small 2p trace matrix: {matrix.shape}"
            )
        if not np.isfinite(matrix).all():
            nonfinite = int(matrix.size - np.isfinite(matrix).sum())
            raise ValueError(
                f"2p CSV contains {nonfinite} non-finite values. Clean or "
                "impute the raw trace matrix before this analysis."
            )
        n_neurons, n_timepoints = map(int, matrix.shape)
        input_metadata = {
            "source_type": "2p_traces_csv",
            "source_csv": str(data_path),
            "native_shape": [n_neurons, n_timepoints],
            "native_axis_order": ["neuron", "time"],
            "region_names": [
                f"neuron_{index:03d}" for index in range(n_neurons)
            ],
        }
        data = matrix[np.newaxis, np.newaxis, :, :]
        return (
            data,
            input_metadata,
            None,
            [n_neurons, n_timepoints],
            ["neuron", "time"],
        )

    raise ValueError(f"Unsupported input format: {input_format}")


def _session_time_major(data: np.ndarray, sequence_id: int) -> np.ndarray:
    """Return one recording as [consecutive_time, region]."""

    blocks = np.asarray(data[sequence_id])
    if blocks.ndim != 3:
        raise ValueError(
            f"Expected data[{sequence_id}] to have [subsequence,region,time], "
            f"got {blocks.shape}"
        )
    # [subsequence, region, time] -> [subsequence, time, region] -> [time, region]
    return blocks.transpose(0, 2, 1).reshape(
        blocks.shape[0] * blocks.shape[2], blocks.shape[1]
    )


def _valid_starts_from_finite_mask(
    finite_timestep: np.ndarray, window_length: int
) -> np.ndarray:
    if finite_timestep.ndim != 1:
        raise ValueError("finite_timestep must be one-dimensional")
    n_time = finite_timestep.size
    if n_time < window_length:
        return np.empty(0, dtype=np.int64)
    bad = (~finite_timestep).astype(np.int64, copy=False)
    cumulative = np.concatenate(([0], np.cumsum(bad, dtype=np.int64)))
    invalid_counts = cumulative[window_length:] - cumulative[:-window_length]
    return np.flatnonzero(invalid_counts == 0).astype(np.int64, copy=False)


def _eligible_reference_starts(
    starts: np.ndarray, window_length: int
) -> np.ndarray:
    """Keep starts having at least one non-overlapping partner."""

    if starts.size == 0:
        return starts
    left_counts = np.searchsorted(
        starts, starts - window_length, side="right"
    )
    right_indices = np.searchsorted(
        starts, starts + window_length, side="left"
    )
    partner_counts = left_counts + (starts.size - right_indices)
    return starts[partner_counts > 0]


def discover_eligible_sessions(
    data: np.ndarray, window_length: int
) -> list[EligibleSession]:
    eligible: list[EligibleSession] = []
    for sequence_id in range(data.shape[0]):
        session = _session_time_major(data, sequence_id)
        total = int(session.shape[0])
        if total < 2 * window_length:
            continue
        finite_timestep = np.isfinite(session).all(axis=1)
        max_start = total - window_length
        if bool(finite_timestep.all()):
            eligible.append(
                EligibleSession(
                    sequence_id=sequence_id,
                    total_timesteps=total,
                    max_start=max_start,
                    valid_starts=None,
                    eligible_reference_starts=None,
                )
            )
            continue

        starts = _valid_starts_from_finite_mask(finite_timestep, window_length)
        eligible_a = _eligible_reference_starts(starts, window_length)
        if eligible_a.size:
            eligible.append(
                EligibleSession(
                    sequence_id=sequence_id,
                    total_timesteps=total,
                    max_start=max_start,
                    valid_starts=starts,
                    eligible_reference_starts=eligible_a,
                )
            )
    if not eligible:
        raise ValueError(
            "No sequenceID contains two finite, non-overlapping windows of "
            f"{window_length} consecutive timesteps."
        )
    return eligible


def _sample_nonoverlapping_starts(
    session: EligibleSession, window_length: int, rng: np.random.Generator
) -> tuple[int, int]:
    if session.valid_starts is None:
        reference_start = int(rng.integers(0, session.max_start + 1))
        left_count = max(0, reference_start - window_length + 1)
        right_first = reference_start + window_length
        right_count = max(0, session.max_start - right_first + 1)
        partner_count = left_count + right_count
        if partner_count <= 0:
            raise RuntimeError("Eligible all-finite session unexpectedly has no partner")
        draw = int(rng.integers(0, partner_count))
        if draw < left_count:
            ceiling_start = draw
        else:
            ceiling_start = right_first + (draw - left_count)
        return reference_start, int(ceiling_start)

    starts = session.valid_starts
    eligible_a = session.eligible_reference_starts
    assert starts is not None and eligible_a is not None
    reference_start = int(eligible_a[int(rng.integers(0, eligible_a.size))])
    left_end = int(
        np.searchsorted(starts, reference_start - window_length, side="right")
    )
    right_begin = int(
        np.searchsorted(starts, reference_start + window_length, side="left")
    )
    partner_count = left_end + (starts.size - right_begin)
    draw = int(rng.integers(0, partner_count))
    if draw < left_end:
        ceiling_start = int(starts[draw])
    else:
        ceiling_start = int(starts[right_begin + draw - left_end])
    return reference_start, ceiling_start


def _batch_assignments(n_repetitions: int, n_batches: int) -> np.ndarray:
    assignments = np.empty(n_repetitions, dtype=np.int64)
    for batch, indices in enumerate(np.array_split(np.arange(n_repetitions), n_batches)):
        assignments[indices] = batch
    return assignments


def build_sample_plan(
    eligible: list[EligibleSession],
    *,
    repetitions: int,
    batches: int,
    window_length: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    batch_ids = _batch_assignments(repetitions, batches)
    rows: list[dict[str, int]] = []
    max_seed = np.iinfo(np.uint32).max
    for repetition in range(repetitions):
        session = eligible[int(rng.integers(0, len(eligible)))]
        reference_start, ceiling_start = _sample_nonoverlapping_starts(
            session, window_length, rng
        )
        rows.append(
            {
                "repetition": repetition,
                "batch": int(batch_ids[repetition]),
                "sequence_id": session.sequence_id,
                "reference_start": reference_start,
                "reference_end_exclusive": reference_start + window_length,
                "ceiling_start": ceiling_start,
                "ceiling_end_exclusive": ceiling_start + window_length,
                "temporal_shuffle_seed": int(rng.integers(0, max_seed)),
                "time_region_shuffle_seed": int(rng.integers(0, max_seed)),
            }
        )
    return pd.DataFrame(rows, columns=SAMPLE_COLUMNS)


def temporal_shuffle(
    reference: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Permute time independently within each region."""

    surrogate = np.empty_like(reference)
    for region in range(reference.shape[1]):
        surrogate[:, region] = rng.permutation(reference[:, region])
    return surrogate


def time_and_region_shuffle(
    reference: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Permute all values without replacement across time and region."""

    return rng.permutation(reference.reshape(-1)).reshape(reference.shape)


def score_window_pair(
    ground_truth: np.ndarray, prediction: np.ndarray
) -> dict[str, float]:
    if ground_truth.shape != prediction.shape:
        raise ValueError(
            f"Window shape mismatch: {ground_truth.shape} vs {prediction.shape}"
        )
    if ground_truth.ndim != 2:
        raise ValueError(f"Expected [time,region] windows, got {ground_truth.shape}")
    scores = calculate_neuro_composites(
        ground_truth[np.newaxis, :, :],
        prediction[np.newaxis, :, :],
    )
    return {
        str(key): float(value) if value is not None else float("nan")
        for key, value in scores.items()
    }


def _calculate_neuro_composites_with_constant_guard(
    ground_truth: np.ndarray, prediction: np.ndarray
) -> dict[str, Any]:
    """Run the canonical scorer, guarding its optional PSD diagnostic at rho=0.

    Nethobench's additional-metric bundle computes a non-composite PSD
    diagnostic before returning the canonical structural metrics. For an
    exactly constant prediction, its normalized PSD is empty and the current
    correlation/RMSE helpers raise on the resulting shape mismatch. Returning
    NaN for only that undefined optional comparison lets the unchanged
    canonical scorer continue. Non-constant inputs take the original path
    without monkey-patching.
    """

    prediction_array = np.asarray(prediction, dtype=np.float64)
    is_temporally_constant = bool(
        np.all(prediction_array.var(axis=1, ddof=0) <= 1e-24)
    )
    if not is_temporally_constant:
        return calculate_neuro_composites(ground_truth, prediction)

    original_correlation = nethobench_additional_metrics.correlation_score
    original_rmse = nethobench_additional_metrics.rmse_similarity

    def shape_safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
        if np.shape(left) != np.shape(right):
            return float("nan")
        return float(original_correlation(left, right))

    def shape_safe_rmse(left: np.ndarray, right: np.ndarray) -> float:
        if np.shape(left) != np.shape(right):
            return float("nan")
        return float(original_rmse(left, right))

    nethobench_additional_metrics.correlation_score = shape_safe_correlation
    nethobench_additional_metrics.rmse_similarity = shape_safe_rmse
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Precision loss occurred in moment calculation.*",
                category=RuntimeWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r"Mean of empty slice",
                category=RuntimeWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r"invalid value encountered in divide",
                category=RuntimeWarning,
            )
            return calculate_neuro_composites(ground_truth, prediction)
    finally:
        nethobench_additional_metrics.correlation_score = original_correlation
        nethobench_additional_metrics.rmse_similarity = original_rmse


def score_variance_contraction_pair(
    ground_truth: np.ndarray, prediction: np.ndarray
) -> dict[str, float]:
    if ground_truth.shape != prediction.shape:
        raise ValueError(
            f"Window shape mismatch: {ground_truth.shape} vs {prediction.shape}"
        )
    if ground_truth.ndim != 2:
        raise ValueError(f"Expected [time,region] windows, got {ground_truth.shape}")
    scores = _calculate_neuro_composites_with_constant_guard(
        ground_truth[np.newaxis, :, :],
        prediction[np.newaxis, :, :],
    )
    return {
        str(key): float(value) if value is not None else float("nan")
        for key, value in scores.items()
    }


def _score_columns(frame: pd.DataFrame) -> list[str]:
    excluded = set(SCORE_META_COLUMNS)
    return [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]


def _ordered_score_columns(columns: Iterable[str]) -> list[str]:
    columns = list(dict.fromkeys(columns))
    primary_metrics = [
        c
        for c in columns
        if c.endswith("_score")
        and not c.startswith("family_")
        and c not in {"composite_score"}
    ]
    families = [c for c in FAMILY_COLUMNS if c in columns]
    composites = [
        c
        for c in (
            "composite_score",
            "FINAL_COMPOSITE_SCORE",
            "FINAL_NEURO_COMPOSITE_SCORE",
        )
        if c in columns
    ]
    aliases = [c for c in columns if c.endswith("_score01")]
    used = set(primary_metrics + families + composites + aliases)
    remainder = [c for c in columns if c not in used]
    return primary_metrics + families + composites + aliases + remainder


def _write_repetition_checkpoint(
    records: list[dict[str, Any]], path: Path
) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    score_columns = _ordered_score_columns(_score_columns(frame))
    ordered = [c for c in SCORE_META_COLUMNS if c in frame.columns] + score_columns
    frame = frame.reindex(columns=ordered).sort_values(
        ["repetition", "condition"], kind="stable"
    )
    frame.to_csv(path, index=False)
    return frame


def _check_resume_plan(existing: pd.DataFrame, plan: pd.DataFrame) -> None:
    missing = [column for column in SAMPLE_COLUMNS if column not in existing.columns]
    if missing:
        raise ValueError(
            "--resume found an older or incompatible repetition_scores.csv; "
            f"missing sample columns: {missing}"
        )
    expected = plan[SAMPLE_COLUMNS].reset_index(drop=True)
    observed = (
        existing[SAMPLE_COLUMNS]
        .drop_duplicates()
        .sort_values("repetition")
        .reset_index(drop=True)
    )
    common = expected[expected["repetition"].isin(observed["repetition"])]
    common = common.reset_index(drop=True)
    if not observed.equals(common):
        raise ValueError(
            "--resume found repetition_scores.csv generated from a different "
            "sample plan, seed, window length, repetition count, or batch count."
        )


def run_repetition_scoring(
    data: np.ndarray,
    plan: pd.DataFrame,
    *,
    window_length: int,
    output_csv: Path,
    resume: bool,
    checkpoint_every: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    completed: set[tuple[int, str]] = set()
    if resume and output_csv.exists():
        existing = pd.read_csv(output_csv)
        _check_resume_plan(existing, plan)
        records = existing.to_dict(orient="records")
        completed = {
            (int(row["repetition"]), str(row["condition"])) for row in records
        }
        print(f"Resuming with {len(completed)} completed repetition/condition rows")

    @lru_cache(maxsize=4)
    def cached_session(sequence_id: int) -> np.ndarray:
        return np.asarray(
            _session_time_major(data, sequence_id), dtype=np.float64
        )

    for plan_row in plan.to_dict(orient="records"):
        repetition = int(plan_row["repetition"])
        pending = [
            condition
            for condition in CONDITIONS
            if (repetition, condition) not in completed
        ]
        if not pending:
            continue

        sequence_id = int(plan_row["sequence_id"])
        session = cached_session(sequence_id)
        a0 = int(plan_row["reference_start"])
        b0 = int(plan_row["ceiling_start"])
        reference = session[a0 : a0 + window_length]
        ceiling_window = session[b0 : b0 + window_length]
        if reference.shape != (window_length, data.shape[2]):
            raise RuntimeError(f"Invalid reference window shape {reference.shape}")
        if ceiling_window.shape != reference.shape:
            raise RuntimeError(f"Invalid ceiling window shape {ceiling_window.shape}")
        if not np.isfinite(reference).all() or not np.isfinite(ceiling_window).all():
            raise RuntimeError("Sample plan produced a non-finite window")
        if not (
            a0 + window_length <= b0 or b0 + window_length <= a0
        ):
            raise RuntimeError("Sample plan produced overlapping split-half windows")

        candidates: dict[str, np.ndarray] = {}
        if CEILING in pending:
            candidates[CEILING] = ceiling_window
        if TEMPORAL_FLOOR in pending:
            candidates[TEMPORAL_FLOOR] = temporal_shuffle(
                reference,
                np.random.default_rng(int(plan_row["temporal_shuffle_seed"])),
            )
        if TIME_REGION_FLOOR in pending:
            candidates[TIME_REGION_FLOOR] = time_and_region_shuffle(
                reference,
                np.random.default_rng(int(plan_row["time_region_shuffle_seed"])),
            )

        metadata = {column: int(plan_row[column]) for column in SAMPLE_COLUMNS}
        for condition in pending:
            scores = score_window_pair(reference, candidates[condition])
            records.append({**metadata, "condition": condition, **scores})
            completed.add((repetition, condition))

        done_repetitions = repetition + 1
        if (
            done_repetitions % checkpoint_every == 0
            or done_repetitions == len(plan)
        ):
            _write_repetition_checkpoint(records, output_csv)
            print(
                f"Scored {done_repetitions}/{len(plan)} repetitions "
                f"({len(completed)}/{len(plan) * len(CONDITIONS)} pairs)"
            )

    expected_count = len(plan) * len(CONDITIONS)
    if len(completed) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} scored rows, found {len(completed)}"
        )
    return _write_repetition_checkpoint(records, output_csv)


def _score_level(score_name: str) -> str:
    if score_name.startswith("family_"):
        return "family"
    if score_name in {
        "composite_score",
        "FINAL_COMPOSITE_SCORE",
        "FINAL_NEURO_COMPOSITE_SCORE",
    }:
        return "composite"
    if score_name.endswith("_score01"):
        return "legacy_alias"
    return "metric"


def summarize_values(
    frame: pd.DataFrame, group_columns: list[str], score_columns: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouper: str | list[str] = (
        group_columns[0] if len(group_columns) == 1 else group_columns
    )
    for group_key, group in frame.groupby(grouper, sort=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group_meta = dict(zip(group_columns, group_key))
        for score_name in score_columns:
            values = pd.to_numeric(group[score_name], errors="coerce").to_numpy(
                dtype=float
            )
            values = values[np.isfinite(values)]
            n = int(values.size)
            mean = float(values.mean()) if n else float("nan")
            std = float(values.std(ddof=1)) if n >= 2 else float("nan")
            sem = std / math.sqrt(n) if n >= 2 else float("nan")
            if n >= 2 and np.isfinite(sem):
                half_width = float(student_t.ppf(0.975, df=n - 1) * sem)
                ci_low, ci_high = mean - half_width, mean + half_width
            else:
                ci_low = ci_high = float("nan")
            rows.append(
                {
                    **group_meta,
                    "score_name": score_name,
                    "score_level": _score_level(score_name),
                    "n": n,
                    "mean": mean,
                    "std": std,
                    "sem": sem,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                }
            )
    return pd.DataFrame(rows)


def contract_temporal_variance(
    ceiling_window: np.ndarray,
    rho: float,
    *,
    mean_atol: float = 1e-12,
    variance_rtol: float = 1e-10,
    variance_atol: float = 1e-14,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Contract each region around its temporal mean by ``sqrt(rho)``."""

    window = np.asarray(ceiling_window, dtype=np.float64)
    if window.ndim != 2:
        raise ValueError(f"Expected [time,region] window, got {window.shape}")
    if not np.isfinite(window).all():
        raise ValueError("Variance contraction requires a finite ceiling window")
    rho = float(rho)
    if not np.isfinite(rho) or not 0.0 <= rho <= 1.0:
        raise ValueError(f"rho must lie in [0,1], got {rho!r}")

    original_means = window.mean(axis=0)
    contracted = original_means[None, :] + math.sqrt(rho) * (
        window - original_means[None, :]
    )
    contracted_means = contracted.mean(axis=0)
    original_variances = window.var(axis=0, ddof=0)
    contracted_variances = contracted.var(axis=0, ddof=0)
    expected_variances = rho * original_variances
    mean_errors = np.abs(contracted_means - original_means)
    variance_errors = np.abs(contracted_variances - expected_variances)
    positive_variance = original_variances > variance_atol
    achieved_ratios = np.full(original_variances.shape, np.nan, dtype=float)
    achieved_ratios[positive_variance] = (
        contracted_variances[positive_variance]
        / original_variances[positive_variance]
    )
    relative_errors = np.full(original_variances.shape, np.nan, dtype=float)
    relative_errors[positive_variance] = np.abs(
        achieved_ratios[positive_variance] - rho
    )

    means_preserved = bool(
        np.allclose(
            contracted_means,
            original_means,
            atol=mean_atol,
            rtol=mean_atol,
        )
    )
    variances_match = bool(
        np.allclose(
            contracted_variances,
            expected_variances,
            atol=variance_atol,
            rtol=variance_rtol,
        )
    )
    if not means_preserved:
        raise RuntimeError(
            "Variance contraction failed to preserve regional temporal means; "
            f"maximum absolute error={float(mean_errors.max()):.3g}"
        )
    if not variances_match:
        raise RuntimeError(
            "Variance contraction failed to achieve the requested regional "
            f"variance ratio rho={rho:g}; maximum absolute error="
            f"{float(variance_errors.max()):.3g}"
        )

    finite_ratios = achieved_ratios[np.isfinite(achieved_ratios)]
    finite_relative_errors = relative_errors[np.isfinite(relative_errors)]
    checks = {
        "rho": rho,
        "sqrt_rho": math.sqrt(rho),
        "means_preserved": means_preserved,
        "variances_match": variances_match,
        "max_abs_mean_error": float(mean_errors.max(initial=0.0)),
        "max_abs_variance_error": float(variance_errors.max(initial=0.0)),
        "max_abs_variance_ratio_error": (
            float(finite_relative_errors.max(initial=0.0))
            if finite_relative_errors.size
            else 0.0
        ),
        "min_achieved_variance_ratio": (
            float(finite_ratios.min()) if finite_ratios.size else float("nan")
        ),
        "max_achieved_variance_ratio": (
            float(finite_ratios.max()) if finite_ratios.size else float("nan")
        ),
        "nonzero_variance_region_count": int(positive_variance.sum()),
        "zero_variance_region_count": int((~positive_variance).sum()),
    }
    return contracted, checks


def _dose_response_score_columns(score_columns: list[str]) -> list[str]:
    metrics = [
        column
        for column in score_columns
        if column.endswith("_score")
        and not column.endswith("_score01")
        and not column.startswith("family_")
        and column != "composite_score"
    ]
    families = [column for column in FAMILY_COLUMNS if column in score_columns]
    composites = [
        column
        for column in ["FINAL_COMPOSITE_SCORE"]
        if column in score_columns
    ]
    selected = metrics + families + composites
    if not metrics or len(families) != len(FAMILY_COLUMNS) or not composites:
        raise RuntimeError(
            "Nethobench dose-response output is missing canonical metrics, "
            "one or more families, or FINAL_COMPOSITE_SCORE"
        )
    return selected


def run_variance_contraction_repetition_scoring(
    data: np.ndarray,
    plan: pd.DataFrame,
    *,
    window_length: int,
    rhos: list[float],
) -> pd.DataFrame:
    """Score every contracted ceiling against its paired reference window."""

    @lru_cache(maxsize=4)
    def cached_session(sequence_id: int) -> np.ndarray:
        return np.asarray(
            _session_time_major(data, sequence_id), dtype=np.float64
        )

    rows: list[dict[str, Any]] = []
    for plan_row in plan.to_dict(orient="records"):
        sequence_id = int(plan_row["sequence_id"])
        session = cached_session(sequence_id)
        a0 = int(plan_row["reference_start"])
        b0 = int(plan_row["ceiling_start"])
        reference = session[a0 : a0 + window_length]
        ceiling_window = session[b0 : b0 + window_length]
        metadata = {column: int(plan_row[column]) for column in SAMPLE_COLUMNS}
        for rho in rhos:
            contracted, checks = contract_temporal_variance(
                ceiling_window, rho
            )
            scores = score_variance_contraction_pair(reference, contracted)
            rows.append({**metadata, **checks, **scores})
        repetition = int(plan_row["repetition"]) + 1
        if repetition % 5 == 0 or repetition == len(plan):
            print(
                "Variance contraction: scored "
                f"{repetition}/{len(plan)} paired repetitions"
            )
    return pd.DataFrame(rows).sort_values(
        ["repetition", "rho"], ascending=[True, False], kind="stable"
    )


def score_variance_contraction_batches(
    data: np.ndarray,
    plan: pd.DataFrame,
    *,
    window_length: int,
    rhos: list[float],
    score_columns: list[str],
) -> pd.DataFrame:
    """Jointly score the same paired windows within every existing batch."""

    @lru_cache(maxsize=4)
    def cached_session(sequence_id: int) -> np.ndarray:
        return np.asarray(
            _session_time_major(data, sequence_id), dtype=np.float64
        )

    rows: list[dict[str, Any]] = []
    for batch, batch_plan in plan.groupby("batch", sort=True):
        references: list[np.ndarray] = []
        ceilings: list[np.ndarray] = []
        for plan_row in batch_plan.to_dict(orient="records"):
            session = cached_session(int(plan_row["sequence_id"]))
            a0 = int(plan_row["reference_start"])
            b0 = int(plan_row["ceiling_start"])
            references.append(session[a0 : a0 + window_length])
            ceilings.append(session[b0 : b0 + window_length])
        gt_batch = np.stack(references, axis=0)
        for rho in rhos:
            contracted_batch = np.stack(
                [
                    contract_temporal_variance(window, rho)[0]
                    for window in ceilings
                ],
                axis=0,
            )
            scores = _calculate_neuro_composites_with_constant_guard(
                gt_batch, contracted_batch
            )
            rows.append(
                {
                    "rho": float(rho),
                    "sqrt_rho": math.sqrt(float(rho)),
                    "batch": int(batch),
                    "n_repetitions": int(len(batch_plan)),
                    **{
                        score: (
                            float(scores.get(score, float("nan")))
                            if scores.get(score) is not None
                            else float("nan")
                        )
                        for score in score_columns
                    },
                }
            )
        print(
            f"Variance contraction: jointly scored batch {int(batch) + 1} "
            f"at {len(rhos)} retained-variance levels"
        )
    return pd.DataFrame(rows).sort_values(
        ["rho", "batch"], ascending=[False, True], kind="stable"
    )


def _maximum_score_difference(
    left: pd.DataFrame,
    right: pd.DataFrame,
    score_columns: list[str],
) -> float:
    left_values = left[score_columns].to_numpy(dtype=float)
    right_values = right[score_columns].to_numpy(dtype=float)
    if left_values.shape != right_values.shape:
        raise RuntimeError(
            f"Cannot compare score shapes {left_values.shape} and "
            f"{right_values.shape}"
        )
    if not np.array_equal(np.isnan(left_values), np.isnan(right_values)):
        raise RuntimeError("rho=1 and original ceiling have different NaN patterns")
    finite = np.isfinite(left_values) & np.isfinite(right_values)
    if not finite.any():
        return 0.0
    return float(np.max(np.abs(left_values[finite] - right_values[finite])))


def verify_rho_one_reproduces_ceiling(
    repetition_scores: pd.DataFrame,
    batch_scores: pd.DataFrame,
    dose_repetition_scores: pd.DataFrame,
    dose_batch_scores: pd.DataFrame,
    score_columns: list[str],
    *,
    atol: float = 1e-12,
) -> dict[str, Any]:
    original_repetition = repetition_scores[
        repetition_scores["condition"] == CEILING
    ].sort_values("repetition")
    rho_one_repetition = dose_repetition_scores[
        dose_repetition_scores["rho"] == 1.0
    ].sort_values("repetition")
    original_batch = batch_scores[
        batch_scores["condition"] == CEILING
    ].sort_values("batch")
    rho_one_batch = dose_batch_scores[
        dose_batch_scores["rho"] == 1.0
    ].sort_values("batch")
    repetition_error = _maximum_score_difference(
        original_repetition,
        rho_one_repetition,
        score_columns,
    )
    batch_error = _maximum_score_difference(
        original_batch,
        rho_one_batch,
        score_columns,
    )
    if repetition_error > atol or batch_error > atol:
        raise RuntimeError(
            "rho=1 variance contraction did not reproduce the original ceiling "
            f"(repetition max error={repetition_error:.3g}, "
            f"batch max error={batch_error:.3g})"
        )
    return {
        "verified": True,
        "absolute_tolerance": atol,
        "repetition_max_abs_score_error": repetition_error,
        "batch_max_abs_score_error": batch_error,
    }


def build_paired_bootstrap_indices(
    n_batches: int, bootstrap_count: int, seed: int
) -> np.ndarray:
    if n_batches <= 0 or bootstrap_count <= 0:
        raise ValueError("Bootstrap dimensions must be positive")
    return np.random.default_rng(seed).integers(
        0,
        n_batches,
        size=(bootstrap_count, n_batches),
        dtype=np.int64,
    )


def summarize_variance_contraction_bootstrap(
    batch_scores: pd.DataFrame,
    score_columns: list[str],
    bootstrap_indices: np.ndarray,
) -> pd.DataFrame:
    batch_ids = sorted(int(value) for value in batch_scores["batch"].unique())
    if bootstrap_indices.shape[1] != len(batch_ids):
        raise ValueError(
            "Bootstrap width does not match the number of joint batches"
        )
    rows: list[dict[str, Any]] = []
    for rho, group in batch_scores.groupby("rho", sort=False):
        indexed = group.set_index("batch").reindex(batch_ids)
        if indexed[score_columns].shape[0] != len(batch_ids):
            raise RuntimeError(f"Missing batch score at rho={rho:g}")
        for score_name in score_columns:
            values = indexed[score_name].to_numpy(dtype=float)
            sampled = values[bootstrap_indices]
            sampled_finite = np.isfinite(sampled)
            sampled_counts = sampled_finite.sum(axis=1)
            bootstrap_means = np.full(
                bootstrap_indices.shape[0], np.nan, dtype=float
            )
            usable_draws = sampled_counts > 0
            bootstrap_means[usable_draws] = (
                np.where(sampled_finite, sampled, 0.0).sum(axis=1)[usable_draws]
                / sampled_counts[usable_draws]
            )
            finite = bootstrap_means[np.isfinite(bootstrap_means)]
            finite_values = values[np.isfinite(values)]
            rows.append(
                {
                    "rho": float(rho),
                    "sqrt_rho": math.sqrt(float(rho)),
                    "score_name": score_name,
                    "score_level": _score_level(score_name),
                    "n_batches": len(batch_ids),
                    "bootstrap_count": int(bootstrap_indices.shape[0]),
                    "point_mean": (
                        float(np.mean(finite_values))
                        if finite_values.size
                        else float("nan")
                    ),
                    "point_median": (
                        float(np.median(finite_values))
                        if finite_values.size
                        else float("nan")
                    ),
                    "bootstrap_mean": (
                        float(np.mean(finite)) if finite.size else float("nan")
                    ),
                    "bootstrap_median": (
                        float(np.median(finite)) if finite.size else float("nan")
                    ),
                    "bootstrap_ci95_low": (
                        float(np.quantile(finite, 0.025))
                        if finite.size
                        else float("nan")
                    ),
                    "bootstrap_ci95_high": (
                        float(np.quantile(finite, 0.975))
                        if finite.size
                        else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["score_level", "score_name", "rho"],
        ascending=[True, True, False],
        kind="stable",
    )


def _dose_score_label(score_name: str) -> str:
    if score_name in PLOT_SCORE_LABELS:
        return PLOT_SCORE_LABELS[score_name].replace("\n", " ")
    return score_name.removesuffix("_score").replace("_", " ")


def _save_svg_png(fig: plt.Figure, svg_path: Path, png_path: Path) -> None:
    """Save both formats, retrying transient Windows overwrite failures."""
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    for attempt in range(5):
        try:
            with png_path.open("wb") as png_file:
                fig.savefig(
                    png_file,
                    format="png",
                    dpi=300,
                    bbox_inches="tight",
                )
            return
        except OSError as error:
            if error.errno != 22 or attempt == 4:
                raise
            time.sleep(0.15 * (attempt + 1))


def _draw_dose_response_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    score_name: str,
    *,
    color: str,
) -> None:
    subset = summary[summary["score_name"] == score_name].sort_values("rho")
    x = subset["rho"].to_numpy(dtype=float)
    y = subset["bootstrap_mean"].to_numpy(dtype=float)
    low = subset["bootstrap_ci95_low"].to_numpy(dtype=float)
    high = subset["bootstrap_ci95_high"].to_numpy(dtype=float)
    ax.plot(x, y, marker="o", linewidth=1.8, color=color)
    ax.fill_between(x, low, high, color=color, alpha=0.18, linewidth=0)
    ax.set_title(_dose_score_label(score_name), fontsize=9)
    ax.set_xlim(-0.025, 1.025)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)


def plot_variance_contraction_metrics(
    summary: pd.DataFrame,
    metric_columns: list[str],
    output_dir: Path,
) -> tuple[Path, Path]:
    setup_plot_style()
    n_columns = 4
    n_rows = math.ceil(len(metric_columns) / n_columns)
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(12.0, 2.65 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for ax, score_name in zip(axes.flat, metric_columns):
        _draw_dose_response_panel(
            ax, summary, score_name, color="#3E7CB1"
        )
    for ax in axes.flat[len(metric_columns) :]:
        ax.set_visible(False)
    fig.supxlabel("Retained temporal variance ratio (rho)")
    fig.supylabel("Nethobench score (bootstrap mean and 95% CI)")
    fig.suptitle("Metric dose-response to ceiling-window variance contraction")
    fig.tight_layout()
    svg = output_dir / "variance_contraction_metric_dose_response.svg"
    png = output_dir / "variance_contraction_metric_dose_response.png"
    _save_svg_png(fig, svg, png)
    plt.close(fig)
    return svg, png


def plot_variance_contraction_families_and_composite(
    summary: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    setup_plot_style()
    score_names = FAMILY_COLUMNS + ["FINAL_COMPOSITE_SCORE"]
    colors = [
        "#4C78A8",
        "#F58518",
        "#54A24B",
        "#E45756",
        "#72B7B2",
        "#6F4E7C",
    ]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for score_name, color in zip(score_names, colors):
        subset = summary[
            summary["score_name"] == score_name
        ].sort_values("rho")
        x = subset["rho"].to_numpy(dtype=float)
        y = subset["bootstrap_mean"].to_numpy(dtype=float)
        low = subset["bootstrap_ci95_low"].to_numpy(dtype=float)
        high = subset["bootstrap_ci95_high"].to_numpy(dtype=float)
        is_composite = score_name == "FINAL_COMPOSITE_SCORE"
        ax.plot(
            x,
            y,
            marker="o",
            markersize=4.0,
            linewidth=2.2 if is_composite else 1.6,
            linestyle="--" if is_composite else "-",
            color=color,
            label=_dose_score_label(score_name),
            zorder=3 if is_composite else 2,
        )
        ax.fill_between(
            x,
            low,
            high,
            color=color,
            alpha=0.10,
            linewidth=0,
            zorder=1,
        )
    ax.set_xlim(-0.025, 1.025)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Retained temporal variance ratio (rho)")
    ax.set_ylabel("Nethobench score (bootstrap mean and 95% CI)")
    ax.set_title(
        "Family and composite dose-response to variance contraction"
    )
    ax.grid(False)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    svg = output_dir / "variance_contraction_family_composite_dose_response.svg"
    png = output_dir / "variance_contraction_family_composite_dose_response.png"
    _save_svg_png(fig, svg, png)
    plt.close(fig)
    return svg, png


def load_cached_variance_contraction_results(
    output_dir: Path,
    *,
    data_path: Path,
    samples_path: Path,
    window_length: int,
    repetitions: int,
    batches: int,
    seed: int,
    rhos: list[float],
    bootstrap_count: int,
    score_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray] | None:
    """Load compatible dose-response estimates instead of rescoring arrays."""
    results_path = output_dir / "variance_contraction_dose_response_results.json"
    repetition_path = output_dir / "variance_contraction_repetition_scores.csv"
    batch_path = output_dir / "variance_contraction_batch_scores.csv"
    bootstrap_path = output_dir / "variance_contraction_bootstrap_summary.csv"
    bootstrap_indices_path = (
        output_dir / "variance_contraction_bootstrap_indices.csv"
    )
    required = (
        results_path,
        repetition_path,
        batch_path,
        bootstrap_path,
        bootstrap_indices_path,
    )
    if not all(path.is_file() for path in required):
        return None

    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        config = payload["config"]
        cached_rhos = np.asarray(
            config["retained_variance_ratios"], dtype=float
        )
        requested_rhos = np.asarray(rhos, dtype=float)
        compatible = (
            Path(config["data_path"]).resolve() == data_path.resolve()
            and Path(config["sampled_windows_csv"]).resolve()
            == samples_path.resolve()
            and int(config["window_length"]) == window_length
            and int(config["repetitions"]) == repetitions
            and int(config["batches"]) == batches
            and int(config["seed"]) == seed
            and int(config["bootstrap_count"]) == bootstrap_count
            and cached_rhos.shape == requested_rhos.shape
            and np.allclose(cached_rhos, requested_rhos, rtol=0.0, atol=1e-12)
            and list(payload["score_columns"]) == score_columns
        )
        if not compatible:
            print("Saved variance-contraction results are incompatible; rescoring")
            return None

        repetition_scores = pd.read_csv(repetition_path)
        batch_scores = pd.read_csv(batch_path)
        bootstrap_summary = pd.read_csv(bootstrap_path)
        bootstrap_frame = pd.read_csv(bootstrap_indices_path)
        expected_repetition_keys = {
            (repetition, round(rho, 12))
            for repetition in range(repetitions)
            for rho in rhos
        }
        observed_repetition_keys = {
            (int(row["repetition"]), round(float(row["rho"]), 12))
            for row in repetition_scores.to_dict(orient="records")
        }
        expected_batch_keys = {
            (batch, round(rho, 12))
            for batch in range(batches)
            for rho in rhos
        }
        observed_batch_keys = {
            (int(row["batch"]), round(float(row["rho"]), 12))
            for row in batch_scores.to_dict(orient="records")
        }
        expected_summary_keys = {
            (score, round(rho, 12))
            for score in score_columns
            for rho in rhos
        }
        observed_summary_keys = {
            (str(row["score_name"]), round(float(row["rho"]), 12))
            for row in bootstrap_summary.to_dict(orient="records")
        }
        draw_columns = [
            column
            for column in bootstrap_frame.columns
            if column.startswith("batch_draw_")
        ]
        required_score_columns = set(score_columns)
        if (
            observed_repetition_keys != expected_repetition_keys
            or observed_batch_keys != expected_batch_keys
            or observed_summary_keys != expected_summary_keys
            or not required_score_columns.issubset(repetition_scores.columns)
            or not required_score_columns.issubset(batch_scores.columns)
            or len(bootstrap_frame) != bootstrap_count
            or len(draw_columns) != batches
        ):
            print("Saved variance-contraction tables are incomplete; rescoring")
            return None
        bootstrap_indices = bootstrap_frame[draw_columns].to_numpy(
            dtype=np.int64
        )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        print(f"Could not reuse saved variance-contraction results: {error}")
        return None

    print(
        "Reusing cached variance-contraction repetition, batch, and bootstrap "
        "estimates"
    )
    return (
        repetition_scores,
        batch_scores,
        bootstrap_summary,
        bootstrap_indices,
    )


def score_batches(
    data: np.ndarray,
    plan: pd.DataFrame,
    *,
    window_length: int,
    score_columns: list[str],
) -> pd.DataFrame:
    """Score every empirical batch jointly, matching model split scoring.

    Repetition-level scores describe the sampled-pair distribution. These
    joint batch scores are the correct reference for model
    split normalization because Nethobench contains nonlinear reductions across
    sequences.
    """

    @lru_cache(maxsize=4)
    def cached_session(sequence_id: int) -> np.ndarray:
        return np.asarray(
            _session_time_major(data, sequence_id), dtype=np.float64
        )

    rows: list[dict[str, Any]] = []
    for batch, batch_plan in plan.groupby("batch", sort=True):
        references: list[np.ndarray] = []
        candidates: dict[str, list[np.ndarray]] = {
            condition: [] for condition in CONDITIONS
        }
        for plan_row in batch_plan.to_dict(orient="records"):
            session = cached_session(int(plan_row["sequence_id"]))
            a0 = int(plan_row["reference_start"])
            b0 = int(plan_row["ceiling_start"])
            reference = session[a0 : a0 + window_length]
            references.append(reference)
            candidates[CEILING].append(session[b0 : b0 + window_length])
            candidates[TEMPORAL_FLOOR].append(
                temporal_shuffle(
                    reference,
                    np.random.default_rng(
                        int(plan_row["temporal_shuffle_seed"])
                    ),
                )
            )
            candidates[TIME_REGION_FLOOR].append(
                time_and_region_shuffle(
                    reference,
                    np.random.default_rng(
                        int(plan_row["time_region_shuffle_seed"])
                    ),
                )
            )

        gt_batch = np.stack(references, axis=0)
        for condition in CONDITIONS:
            pred_batch = np.stack(candidates[condition], axis=0)
            scores = calculate_neuro_composites(gt_batch, pred_batch)
            score_row = {
                score: (
                    float(scores.get(score, float("nan")))
                    if scores.get(score) is not None
                    else float("nan")
                )
                for score in score_columns
            }
            rows.append(
                {
                    "condition": condition,
                    "batch": int(batch),
                    "n_repetitions": int(len(batch_plan)),
                    **score_row,
                }
            )
            print(
                f"Batch {int(batch) + 1}: scored "
                f"{CONDITION_LABELS[condition]} with n={len(batch_plan)}"
            )
    return pd.DataFrame(rows)


def plot_empirical_references(
    batch_summary: pd.DataFrame,
    output_dir: Path,
    *,
    output_prefix: str,
) -> tuple[Path, Path]:
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    x = np.arange(len(PLOT_SCORE_COLUMNS))
    n_conditions = len(CONDITIONS)
    group_width = 0.8
    bar_width = (group_width / n_conditions) * 0.78

    for index, condition in enumerate(CONDITIONS):
        offsets = x + (index - (n_conditions - 1) / 2) * bar_width
        subset = batch_summary[batch_summary["condition"] == condition].set_index(
            "score_name"
        )
        values = np.array(
            [subset.at[c, "mean"] if c in subset.index else np.nan for c in PLOT_SCORE_COLUMNS]
        )
        errors = np.array(
            [subset.at[c, "sem"] if c in subset.index else np.nan for c in PLOT_SCORE_COLUMNS]
        )
        ax.bar(
            offsets,
            values,
            width=bar_width,
            label=CONDITION_LABELS[condition],
            color=CONDITION_COLORS[condition],
            edgecolor="none",
            linewidth=0.0,
            alpha=0.9,
            yerr=np.nan_to_num(errors, nan=0.0),
            capsize=2.5,
            error_kw={"elinewidth": 0.9, "capthick": 0.9},
        )

    ax.set_xticks(x)
    ax.set_xticklabels([PLOT_SCORE_LABELS[c] for c in PLOT_SCORE_COLUMNS], fontsize=10)
    ax.set_ylabel("Nethobench score")
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=False, fontsize=8.5, ncol=2)
    fig.tight_layout()
    svg_path = output_dir / f"{output_prefix}_ceiling_floor_family_scores.svg"
    png_path = output_dir / f"{output_prefix}_ceiling_floor_family_scores.png"
    _save_svg_png(fig, svg_path, png_path)
    plt.close(fig)
    return svg_path, png_path


def load_model_cache(path: Path, expected_batches: int) -> tuple[dict, list[str]]:
    """Load cached model scores without invoking Nethobench scoring.

    The current three-seed cache is organized as training seed -> model ->
    split. Corresponding splits are averaged across training seeds so they
    remain aligned with the four empirical ceiling/floor batches. The legacy
    ``per_split`` cache schema remains supported for explicit overrides.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Model score cache not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    signature = payload.get("_cache_signature", {})
    declared_splits = signature.get("n_splits")
    if declared_splits is not None and int(declared_splits) != expected_batches:
        raise ValueError(
            f"{path} declares n_splits={declared_splits}; expected "
            f"{expected_batches} to align with empirical batches"
        )
    per_split = payload.get("per_split")
    if not isinstance(per_split, dict) or not per_split:
        seed_scores = payload.get("scores")
        if not isinstance(seed_scores, dict) or not seed_scores:
            raise ValueError(
                f"{path} contains neither a non-empty 'scores' nor "
                "'per_split' mapping"
            )

        training_seeds = list(seed_scores)
        declared_models = signature.get("models", [])
        available_models = list(declared_models) or list(
            next(iter(seed_scores.values()))
        )
        per_split = {}
        for model in available_models:
            split_scores_by_seed: list[list[Mapping[str, Any]]] = []
            for training_seed in training_seeds:
                seed_mapping = seed_scores[training_seed]
                if not isinstance(seed_mapping, dict) or model not in seed_mapping:
                    raise ValueError(
                        f"{path} is missing model {model!r} for training seed "
                        f"{training_seed!r}"
                    )
                splits = seed_mapping[model]
                if not isinstance(splits, list) or len(splits) != expected_batches:
                    raise ValueError(
                        f"{model} at training seed {training_seed} has "
                        f"{len(splits) if isinstance(splits, list) else 'invalid'} "
                        f"splits; expected {expected_batches}"
                    )
                split_scores_by_seed.append(splits)

            averaged_splits: list[dict[str, float]] = []
            for split_index in range(expected_batches):
                score_names = list(split_scores_by_seed[0][split_index])
                averaged_splits.append(
                    {
                        score_name: float(
                            np.mean(
                                [
                                    float(seed_splits[split_index][score_name])
                                    for seed_splits in split_scores_by_seed
                                ]
                            )
                        )
                        for score_name in score_names
                    }
                )
            per_split[model] = averaged_splits

        print(
            f"Loaded cached model scores from {len(training_seeds)} training "
            "seeds; averaged corresponding splits without recomputing scores"
        )
    models = [model for model in MODEL_ORDER if model in per_split]
    models.extend(model for model in per_split if model not in models)
    for model in models:
        splits = per_split[model]
        if not isinstance(splits, list) or len(splits) != expected_batches:
            raise ValueError(
                f"{model} has {len(splits) if isinstance(splits, list) else 'invalid'} "
                f"splits; expected {expected_batches} to align with empirical batches"
            )
    return per_split, models


def normalize_model_scores(
    per_split: Mapping[str, list[Mapping[str, Any]]],
    models: list[str],
    batch_scores: pd.DataFrame,
    *,
    floor_condition: str,
    clip: bool,
) -> pd.DataFrame:
    references: dict[tuple[str, int], dict[str, float]] = {}
    for row in batch_scores.to_dict(orient="records"):
        key = (str(row["condition"]), int(row["batch"]))
        references[key] = {
            score: float(value)
            for score, value in row.items()
            if score not in {"condition", "batch"}
        }

    rows: list[dict[str, Any]] = []
    for model in models:
        for split, model_scores in enumerate(per_split[model]):
            ceiling_scores = references[(CEILING, split)]
            floor_scores = references[(floor_condition, split)]
            common_scores = [
                score
                for score in model_scores
                if score in ceiling_scores and score in floor_scores
            ]
            for score_name in common_scores:
                model_score = float(model_scores[score_name])
                ceiling_score = float(ceiling_scores[score_name])
                floor_score = float(floor_scores[score_name])
                denominator = ceiling_score - floor_score
                if (
                    np.isfinite(model_score)
                    and np.isfinite(ceiling_score)
                    and np.isfinite(floor_score)
                    and not np.isclose(denominator, 0.0, atol=1e-12, rtol=1e-9)
                ):
                    raw = (model_score - floor_score) / denominator
                else:
                    raw = float("nan")
                normalized = (
                    float(np.clip(raw, 0.0, 1.0))
                    if clip and np.isfinite(raw)
                    else float(raw)
                )
                rows.append(
                    {
                        "model": model,
                        "split": split,
                        "score_name": score_name,
                        "score_level": _score_level(score_name),
                        "model_score": model_score,
                        "ceiling_score": ceiling_score,
                        "floor_score": floor_score,
                        "denominator": denominator,
                        "valid_denominator": bool(
                            np.isfinite(denominator)
                            and not np.isclose(
                                denominator, 0.0, atol=1e-12, rtol=1e-9
                            )
                        ),
                        "inverted_reference": bool(
                            np.isfinite(denominator) and denominator < 0.0
                        ),
                        "normalized_unclipped": raw,
                        "normalized_score": normalized,
                        "below_zero": bool(np.isfinite(raw) and raw < 0.0),
                        "above_one": bool(np.isfinite(raw) and raw > 1.0),
                        "clipped": bool(clip and np.isfinite(raw) and raw != normalized),
                        "floor_condition": floor_condition,
                    }
                )
    return pd.DataFrame(rows)


def summarize_normalized_scores(normalized: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, score_name), group in normalized.groupby(
        ["model", "score_name"], sort=False
    ):
        values = group["normalized_score"].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        n = int(finite.size)
        mean = float(finite.mean()) if n else float("nan")
        std = float(finite.std(ddof=1)) if n >= 2 else float("nan")
        sem = std / math.sqrt(n) if n >= 2 else float("nan")
        if n >= 2 and np.isfinite(sem):
            half_width = float(student_t.ppf(0.975, df=n - 1) * sem)
            ci_low, ci_high = mean - half_width, mean + half_width
        else:
            ci_low = ci_high = float("nan")
        rows.append(
            {
                "model": model,
                "score_name": score_name,
                "score_level": _score_level(score_name),
                "n_splits": n,
                "mean": mean,
                "std": std,
                "sem": sem,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "n_below_zero": int(group["below_zero"].sum()),
                "n_above_one": int(group["above_one"].sum()),
                "n_inverted_reference": int(
                    group["inverted_reference"].sum()
                ),
                "floor_condition": str(group["floor_condition"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def plot_normalized_models(
    normalized: pd.DataFrame,
    normalized_summary: pd.DataFrame,
    models: list[str],
    output_dir: Path,
    *,
    floor_condition: str,
    clipped: bool,
) -> tuple[Path, Path]:
    setup_plot_style()
    plot_models = [model for model in MODEL_ORDER if model in models]
    plot_models.extend(model for model in models if model not in plot_models)
    x = np.arange(len(PLOT_SCORE_COLUMNS))
    n_models = len(plot_models)
    group_width = 0.82
    bar_width = (group_width / n_models) * 0.72

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    plotted_values: list[float] = [0.0, 1.0]

    summary_index = normalized_summary.set_index(["model", "score_name"])
    for index, model in enumerate(plot_models):
        offsets = x + (index - (n_models - 1) / 2) * bar_width
        values = np.array(
            [
                summary_index.at[(model, score), "mean"]
                if (model, score) in summary_index.index
                else np.nan
                for score in PLOT_SCORE_COLUMNS
            ],
            dtype=float,
        )
        errors = np.array(
            [
                summary_index.at[(model, score), "sem"]
                if (model, score) in summary_index.index
                else np.nan
                for score in PLOT_SCORE_COLUMNS
            ],
            dtype=float,
        )
        ax.bar(
            offsets,
            values,
            width=bar_width,
            label=model,
            color=MODEL_COLORS.get(model),
            edgecolor="none",
            linewidth=0.0,
            alpha=0.9,
            yerr=np.nan_to_num(errors, nan=0.0),
            capsize=1.8,
            error_kw={"elinewidth": 0.75, "capthick": 0.75},
        )

        for score_index, score_name in enumerate(PLOT_SCORE_COLUMNS):
            split_rows = normalized[
                (normalized["model"] == model)
                & (normalized["score_name"] == score_name)
            ]
            for _, row in split_rows.iterrows():
                value = float(row["normalized_score"])
                if not np.isfinite(value):
                    continue
                plotted_values.append(value)
                ax.scatter(
                    offsets[score_index],
                    value,
                    marker="o",
                    s=12,
                    facecolors="none",
                    edgecolors="black",
                    linewidths=0.55,
                    alpha=0.7,
                    zorder=4,
                )

    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.9, alpha=0.8)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.1, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([PLOT_SCORE_LABELS[c] for c in PLOT_SCORE_COLUMNS], fontsize=10)
    ax.set_ylabel("Ceiling/floor-normalized score")
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)

    finite_plot = np.asarray(plotted_values, dtype=float)
    finite_plot = finite_plot[np.isfinite(finite_plot)]
    ymin, ymax = float(finite_plot.min()), float(finite_plot.max())
    margin = 0.08 * max(ymax - ymin, 0.25)
    ax.set_ylim(ymin - margin, ymax + margin)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=False, fontsize=8, ncol=2)
    floor_label = CONDITION_LABELS[floor_condition]
    clip_label = "clipped" if clipped else "unclipped"
    ax.set_title(f"Model scores normalized to empirical ceiling and {floor_label} ({clip_label})")
    fig.tight_layout()
    if floor_condition != TIME_REGION_FLOOR:
        raise ValueError(f"Unsupported normalization floor: {floor_condition}")
    tag = "time_region"
    clip_tag = "_clipped" if clipped else "_unclipped"
    svg_path = output_dir / f"normalized_model_family_scores_{tag}{clip_tag}.svg"
    png_path = output_dir / f"normalized_model_family_scores_{tag}{clip_tag}.png"
    _save_svg_png(fig, svg_path, png_path)
    plt.close(fig)
    return svg_path, png_path


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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_args(args)
    data_path = args.data_path.resolve()
    model_cache_path = args.model_cache.resolve()

    if not data_path.is_file():
        raise FileNotFoundError(f"Input data not found: {data_path}")
    input_format = _resolve_input_format(data_path, args.input_format)
    (
        data,
        input_metadata,
        metadata_path,
        native_shape,
        native_axis_order,
    ) = load_analysis_data(data_path, input_format)

    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    elif input_format == "widefield-npy":
        output_dir = (
            REPO_ROOT / "output" / "widefield_empirical_ceiling_floor"
        )
    else:
        output_dir = (
            REPO_ROOT
            / "output"
            / "2p_empirical_ceiling_floor"
            / data_path.stem
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_regions = args.expected_regions
    if expected_regions is None and input_format == "widefield-npy":
        expected_regions = 16
    if expected_regions is not None and data.shape[2] != expected_regions:
        raise ValueError(
            f"Expected {expected_regions} regions/neurons, found "
            f"{data.shape[2]}"
        )

    should_normalize_models = (
        not args.skip_model_normalization
        and (
            input_format == "widefield-npy"
            or args.normalize_model_scores
        )
    )
    if (
        input_format == "2p-csv"
        and not args.skip_model_normalization
        and not args.normalize_model_scores
    ):
        print(
            "2p CSV input: model-cache normalization is disabled by default. "
            "Use --normalize-model-scores to opt in with a comparable cache."
        )

    total_timesteps = int(data.shape[1] * data.shape[3])
    if total_timesteps < 2 * args.window_length:
        raise ValueError(
            f"Each sequence has at most {total_timesteps} concatenated timesteps, "
            f"fewer than two non-overlapping {args.window_length}-step windows."
        )

    print(f"Input: {data_path}")
    print(f"Input format: {input_format}")
    print(
        f"Canonical shape: {tuple(data.shape)} "
        "[sequence, subsequence, region/neuron, time]"
    )
    print(
        f"Concatenated session length: {total_timesteps}; "
        f"window length: {args.window_length}"
    )
    eligible = discover_eligible_sessions(data, args.window_length)
    print(f"Eligible sequenceIDs: {len(eligible)}/{data.shape[0]}")
    if args.sequence_id is not None:
        if args.sequence_id >= data.shape[0]:
            raise ValueError(
                f"--sequence-id {args.sequence_id} is out of range for "
                f"{data.shape[0]} sequences (valid IDs: 0-{data.shape[0] - 1})"
            )
        eligible = [
            session
            for session in eligible
            if session.sequence_id == args.sequence_id
        ]
        if not eligible:
            raise ValueError(
                f"SequenceID {args.sequence_id} cannot provide two finite, "
                f"non-overlapping windows of {args.window_length} timesteps"
            )
        print(
            f"Restricting all repetitions to sequenceID {args.sequence_id} "
            f"with RNG seed {args.seed}"
        )

    plan = build_sample_plan(
        eligible,
        repetitions=args.repetitions,
        batches=args.batches,
        window_length=args.window_length,
        seed=args.seed,
    )
    samples_path = output_dir / "sampled_windows.csv"
    if args.resume and samples_path.exists():
        existing_plan = pd.read_csv(samples_path)
        if not existing_plan.equals(plan):
            raise ValueError(
                "--resume found sampled_windows.csv incompatible with current arguments"
            )
    plan.to_csv(samples_path, index=False)

    repetition_scores_path = output_dir / "repetition_scores.csv"
    repetition_scores = run_repetition_scoring(
        data,
        plan,
        window_length=args.window_length,
        output_csv=repetition_scores_path,
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
    )
    score_columns = _ordered_score_columns(_score_columns(repetition_scores))
    if not score_columns:
        raise RuntimeError("Nethobench returned no numeric score columns")

    repetition_summary = summarize_values(
        repetition_scores, ["condition"], score_columns
    )
    repetition_summary_path = output_dir / "repetition_summary.csv"
    repetition_summary.to_csv(repetition_summary_path, index=False)

    batch_scores_path = output_dir / "batch_scores.csv"
    batch_scores = pd.DataFrame()
    if args.resume and batch_scores_path.exists():
        cached_batch_scores = pd.read_csv(batch_scores_path)
        expected_keys = {
            (condition, batch)
            for condition in CONDITIONS
            for batch in range(args.batches)
        }
        observed_keys = {
            (str(row["condition"]), int(row["batch"]))
            for row in cached_batch_scores.to_dict(orient="records")
        }
        if expected_keys == observed_keys and set(score_columns).issubset(
            cached_batch_scores.columns
        ):
            batch_scores = cached_batch_scores
            print(f"Reusing {len(batch_scores)} cached joint batch scores")
    if batch_scores.empty:
        batch_scores = score_batches(
            data,
            plan,
            window_length=args.window_length,
            score_columns=score_columns,
        )
    batch_scores.to_csv(batch_scores_path, index=False)
    batch_summary = summarize_values(batch_scores, ["condition"], score_columns)
    batch_summary_path = output_dir / "batch_summary.csv"
    batch_summary.to_csv(batch_summary_path, index=False)

    empirical_svg, empirical_png = plot_empirical_references(
        batch_summary,
        output_dir,
        output_prefix=(
            "widefield" if input_format == "widefield-npy" else "2p"
        ),
    )

    variance_contraction_repetition_scores = pd.DataFrame()
    variance_contraction_repetition_summary = pd.DataFrame()
    variance_contraction_batch_scores = pd.DataFrame()
    variance_contraction_batch_summary = pd.DataFrame()
    variance_contraction_bootstrap_summary = pd.DataFrame()
    variance_contraction_validation = pd.DataFrame()
    variance_contraction_bootstrap_indices = np.empty((0, 0), dtype=np.int64)
    variance_contraction_verification: dict[str, Any] = {}
    variance_contraction_paths: dict[str, str] = {}
    dose_score_columns: list[str] = []
    if args.variance_contraction_dose_response:
        rhos = [float(value) for value in args.variance_contraction_rhos]
        print(
            "\nRunning paired variance-contraction dose-response at rho="
            + ", ".join(f"{rho:g}" for rho in rhos)
        )
        dose_score_columns = _dose_response_score_columns(score_columns)
        cached_variance_results = load_cached_variance_contraction_results(
            output_dir,
            data_path=data_path,
            samples_path=samples_path,
            window_length=args.window_length,
            repetitions=args.repetitions,
            batches=args.batches,
            seed=args.seed,
            rhos=rhos,
            bootstrap_count=args.variance_contraction_bootstrap_count,
            score_columns=dose_score_columns,
        )
        if cached_variance_results is None:
            variance_contraction_repetition_scores = (
                run_variance_contraction_repetition_scoring(
                    data,
                    plan,
                    window_length=args.window_length,
                    rhos=rhos,
                )
            )
            variance_contraction_batch_scores = (
                score_variance_contraction_batches(
                    data,
                    plan,
                    window_length=args.window_length,
                    rhos=rhos,
                    score_columns=score_columns,
                )
            )
        else:
            (
                variance_contraction_repetition_scores,
                variance_contraction_batch_scores,
                variance_contraction_bootstrap_summary,
                variance_contraction_bootstrap_indices,
            ) = cached_variance_results
        variance_contraction_repetition_summary = summarize_values(
            variance_contraction_repetition_scores,
            ["rho"],
            dose_score_columns,
        )
        variance_contraction_batch_summary = summarize_values(
            variance_contraction_batch_scores,
            ["rho"],
            dose_score_columns,
        )
        variance_contraction_verification = (
            verify_rho_one_reproduces_ceiling(
                repetition_scores,
                batch_scores,
                variance_contraction_repetition_scores,
                variance_contraction_batch_scores,
                score_columns,
            )
        )
        validation_rows: list[dict[str, Any]] = []
        for rho, group in variance_contraction_repetition_scores.groupby(
            "rho", sort=False
        ):
            validation_rows.append(
                {
                    "rho": float(rho),
                    "sqrt_rho": math.sqrt(float(rho)),
                    "n_windows": int(len(group)),
                    "all_means_preserved": bool(
                        group["means_preserved"].astype(bool).all()
                    ),
                    "all_variances_match": bool(
                        group["variances_match"].astype(bool).all()
                    ),
                    "max_abs_mean_error": float(
                        group["max_abs_mean_error"].max()
                    ),
                    "max_abs_variance_error": float(
                        group["max_abs_variance_error"].max()
                    ),
                    "max_abs_variance_ratio_error": float(
                        group["max_abs_variance_ratio_error"].max()
                    ),
                    "min_achieved_variance_ratio": float(
                        group["min_achieved_variance_ratio"].min()
                    ),
                    "max_achieved_variance_ratio": float(
                        group["max_achieved_variance_ratio"].max()
                    ),
                    "rho_one_repetition_max_abs_score_error": (
                        variance_contraction_verification[
                            "repetition_max_abs_score_error"
                        ]
                        if float(rho) == 1.0
                        else np.nan
                    ),
                    "rho_one_batch_max_abs_score_error": (
                        variance_contraction_verification[
                            "batch_max_abs_score_error"
                        ]
                        if float(rho) == 1.0
                        else np.nan
                    ),
                }
            )
        variance_contraction_validation = pd.DataFrame(validation_rows)

        if cached_variance_results is None:
            variance_contraction_bootstrap_indices = (
                build_paired_bootstrap_indices(
                    args.batches,
                    args.variance_contraction_bootstrap_count,
                    args.seed,
                )
            )
            variance_contraction_bootstrap_summary = (
                summarize_variance_contraction_bootstrap(
                    variance_contraction_batch_scores,
                    dose_score_columns,
                    variance_contraction_bootstrap_indices,
                )
            )
        batch_ids = np.asarray(
            sorted(
                int(value)
                for value in variance_contraction_batch_scores["batch"].unique()
            ),
            dtype=np.int64,
        )
        sampled_batch_ids = batch_ids[variance_contraction_bootstrap_indices]
        bootstrap_index_frame = pd.DataFrame(
            sampled_batch_ids,
            columns=[
                f"batch_draw_{index + 1}" for index in range(args.batches)
            ],
        )
        bootstrap_index_frame.insert(
            0,
            "bootstrap_draw",
            np.arange(args.variance_contraction_bootstrap_count),
        )

        dose_repetition_path = (
            output_dir / "variance_contraction_repetition_scores.csv"
        )
        dose_repetition_summary_path = (
            output_dir / "variance_contraction_repetition_summary.csv"
        )
        dose_batch_path = output_dir / "variance_contraction_batch_scores.csv"
        dose_batch_summary_path = (
            output_dir / "variance_contraction_batch_summary.csv"
        )
        dose_bootstrap_summary_path = (
            output_dir / "variance_contraction_bootstrap_summary.csv"
        )
        dose_bootstrap_indices_path = (
            output_dir / "variance_contraction_bootstrap_indices.csv"
        )
        dose_validation_path = (
            output_dir / "variance_contraction_numerical_checks.csv"
        )
        variance_contraction_repetition_scores.to_csv(
            dose_repetition_path, index=False
        )
        variance_contraction_repetition_summary.to_csv(
            dose_repetition_summary_path, index=False
        )
        variance_contraction_batch_scores.to_csv(
            dose_batch_path, index=False
        )
        variance_contraction_batch_summary.to_csv(
            dose_batch_summary_path, index=False
        )
        variance_contraction_bootstrap_summary.to_csv(
            dose_bootstrap_summary_path, index=False
        )
        bootstrap_index_frame.to_csv(
            dose_bootstrap_indices_path, index=False
        )
        variance_contraction_validation.to_csv(
            dose_validation_path, index=False
        )

        metric_columns = [
            column
            for column in dose_score_columns
            if _score_level(column) == "metric"
        ]
        dose_metric_svg, dose_metric_png = (
            plot_variance_contraction_metrics(
                variance_contraction_bootstrap_summary,
                metric_columns,
                output_dir,
            )
        )
        dose_family_svg, dose_family_png = (
            plot_variance_contraction_families_and_composite(
                variance_contraction_bootstrap_summary,
                output_dir,
            )
        )
        variance_contraction_paths = {
            "repetition_scores_csv": str(dose_repetition_path),
            "repetition_summary_csv": str(dose_repetition_summary_path),
            "batch_scores_csv": str(dose_batch_path),
            "batch_summary_csv": str(dose_batch_summary_path),
            "bootstrap_summary_csv": str(dose_bootstrap_summary_path),
            "bootstrap_indices_csv": str(dose_bootstrap_indices_path),
            "numerical_checks_csv": str(dose_validation_path),
            "metric_dose_response_svg": str(dose_metric_svg),
            "metric_dose_response_png": str(dose_metric_png),
            "family_composite_dose_response_svg": str(dose_family_svg),
            "family_composite_dose_response_png": str(dose_family_png),
        }
        dose_results_json_path = (
            output_dir / "variance_contraction_dose_response_results.json"
        )
        variance_contraction_paths["results_json"] = str(
            dose_results_json_path
        )
        dose_results_payload = {
            "config": {
                "data_path": str(data_path),
                "sampled_windows_csv": str(samples_path),
                "window_length": args.window_length,
                "repetitions": args.repetitions,
                "batches": args.batches,
                "seed": args.seed,
                "retained_variance_ratios": rhos,
                "bootstrap_count": (
                    args.variance_contraction_bootstrap_count
                ),
                "bootstrap_statistic": "mean of joint batch scores",
                "bootstrap_pairing": (
                    "one shared bootstrap index matrix for every rho and score"
                ),
                "x_axis": "linear retained-variance ratio including rho=0",
            },
            "score_columns": dose_score_columns,
            "rho_one_ceiling_verification": _json_safe(
                variance_contraction_verification
            ),
            "numerical_checks": _records(
                variance_contraction_validation
            ),
            "repetition_summary": _records(
                variance_contraction_repetition_summary
            ),
            "batch_summary": _records(
                variance_contraction_batch_summary
            ),
            "bootstrap_summary": _records(
                variance_contraction_bootstrap_summary
            ),
            "bootstrap_indices": (
                variance_contraction_bootstrap_indices.tolist()
            ),
            "outputs": variance_contraction_paths,
        }
        dose_results_json_path.write_text(
            json.dumps(
                _json_safe(dose_results_payload),
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

    normalized = pd.DataFrame()
    normalized_summary = pd.DataFrame()
    normalized_paths: dict[str, str] = {}
    floor_condition = TIME_REGION_FLOOR
    if should_normalize_models:
        per_split, models = load_model_cache(model_cache_path, args.batches)
        normalized = normalize_model_scores(
            per_split,
            models,
            batch_scores,
            floor_condition=floor_condition,
            clip=args.clip_normalized,
        )
        normalized_summary = summarize_normalized_scores(normalized)
        normalized_csv = output_dir / "normalized_model_scores_per_split.csv"
        normalized_summary_csv = output_dir / "normalized_model_scores_summary.csv"
        normalized.to_csv(normalized_csv, index=False)
        normalized_summary.to_csv(normalized_summary_csv, index=False)
        normalized_svg, normalized_png = plot_normalized_models(
            normalized,
            normalized_summary,
            models,
            output_dir,
            floor_condition=floor_condition,
            clipped=args.clip_normalized,
        )
        normalized_paths = {
            "per_split_csv": str(normalized_csv),
            "summary_csv": str(normalized_summary_csv),
            "plot_svg": str(normalized_svg),
            "plot_png": str(normalized_png),
        }

    results_path = output_dir / (
        "widefield_ceiling_floor_results.json"
        if input_format == "widefield-npy"
        else "2p_ceiling_floor_results.json"
    )
    payload = {
        "config": {
            "data_path": str(data_path),
            "input_format": input_format,
            "model_cache": (
                str(model_cache_path) if should_normalize_models else None
            ),
            "model_normalization_enabled": should_normalize_models,
            "output_dir": str(output_dir),
            "window_length": args.window_length,
            "repetitions": args.repetitions,
            "batches": args.batches,
            "seed": args.seed,
            "sequence_id": args.sequence_id,
            "expected_regions": expected_regions,
            "normalization_floor": floor_condition,
            "clip_normalized": bool(args.clip_normalized),
            "variance_contraction_dose_response": {
                "enabled": bool(args.variance_contraction_dose_response),
                "retained_variance_ratios": (
                    [float(value) for value in args.variance_contraction_rhos]
                    if args.variance_contraction_dose_response
                    else []
                ),
                "bootstrap_count": (
                    args.variance_contraction_bootstrap_count
                    if args.variance_contraction_dose_response
                    else 0
                ),
                "bootstrap_seed": (
                    args.seed
                    if args.variance_contraction_dose_response
                    else None
                ),
                "transformation": (
                    "X_rho[t,r] = temporal_mean[r] + sqrt(rho) * "
                    "(X[t,r] - temporal_mean[r])"
                ),
                "pairing": (
                    "all rho values use the exact sampled reference/ceiling "
                    "window pairs in sampled_windows.csv and the same joint-batch "
                    "bootstrap index matrix"
                ),
                "bootstrap_statistic": (
                    "mean of resampled joint batch scores"
                ),
                "confidence_interval": (
                    "2.5th and 97.5th percentiles of paired bootstrap means"
                ),
                "x_axis": (
                    "linear retained-variance ratio; rho=0 is displayed at zero"
                ),
            },
            "scorer": (
                "nethobench.neuro.metrics.composites."
                "calculate_neuro_composites"
            ),
            "confidence_interval": "two-sided 95% Student-t interval",
            "batch_aggregation": (
                "joint Nethobench scoring of all repetition windows stacked "
                "within each contiguous batch"
            ),
        },
        "input": {
            "format": input_format,
            "native_shape": native_shape,
            "native_axis_order": native_axis_order,
            "shape": list(data.shape),
            "axis_order": [
                "sequenceID",
                "subsequence",
                "region_or_neuron",
                "time",
            ],
            "concatenated_timesteps_per_sequence": total_timesteps,
            "eligible_sequence_ids": [entry.sequence_id for entry in eligible],
            "metadata_path": (
                str(metadata_path) if metadata_path is not None else None
            ),
            "region_names": input_metadata.get("region_names"),
            "sequence_id_definition": (
                "the sole continuous recording represented by the 2p CSV"
                if input_format == "2p-csv"
                else
                "axis-0 index in the processed array; for data100_ba16.npy "
                "this equals the sorted source sequenceId"
            ),
            "batch_sizes": {
                str(int(batch)): int(count)
                for batch, count in plan.groupby("batch").size().items()
            },
        },
        "sampled_windows": _records(plan),
        "repetition_scores": _records(repetition_scores),
        "repetition_summary": _records(repetition_summary),
        "batch_scores": _records(batch_scores),
        "batch_summary": _records(batch_summary),
        "variance_contraction": {
            "score_columns": dose_score_columns,
            "repetition_scores": _records(
                variance_contraction_repetition_scores
            ),
            "repetition_summary": _records(
                variance_contraction_repetition_summary
            ),
            "batch_scores": _records(variance_contraction_batch_scores),
            "batch_summary": _records(
                variance_contraction_batch_summary
            ),
            "bootstrap_summary": _records(
                variance_contraction_bootstrap_summary
            ),
            "bootstrap_indices": (
                variance_contraction_bootstrap_indices.tolist()
            ),
            "numerical_checks": _records(
                variance_contraction_validation
            ),
            "rho_one_ceiling_verification": _json_safe(
                variance_contraction_verification
            ),
        },
        "normalized_model_scores_per_split": _records(normalized),
        "normalized_model_scores_summary": _records(normalized_summary),
        "outputs": {
            "sampled_windows_csv": str(samples_path),
            "repetition_scores_csv": str(repetition_scores_path),
            "repetition_summary_csv": str(repetition_summary_path),
            "batch_scores_csv": str(batch_scores_path),
            "batch_summary_csv": str(batch_summary_path),
            "empirical_plot_svg": str(empirical_svg),
            "empirical_plot_png": str(empirical_png),
            "variance_contraction": variance_contraction_paths,
            **normalized_paths,
        },
    }
    results_path.write_text(
        json.dumps(_json_safe(payload), indent=2, allow_nan=False), encoding="utf-8"
    )

    composite = batch_summary[
        batch_summary["score_name"] == "FINAL_COMPOSITE_SCORE"
    ][["condition", "mean", "sem"]]
    print("\nComposite score by empirical condition (mean +/- SEM across batches):")
    for row in composite.to_dict(orient="records"):
        print(
            f"  {CONDITION_LABELS[str(row['condition'])]}: "
            f"{float(row['mean']):.4f} +/- {float(row['sem']):.4f}"
        )
    print(f"\nResults JSON: {results_path}")
    print(f"Repetition scores: {repetition_scores_path}")
    print(f"Empirical plot: {empirical_svg}")
    if variance_contraction_paths:
        dose_composite = variance_contraction_bootstrap_summary[
            variance_contraction_bootstrap_summary["score_name"]
            == "FINAL_COMPOSITE_SCORE"
        ].sort_values("rho", ascending=False)
        print(
            "\nVariance-contraction composite dose-response "
            "(bootstrap mean [95% CI]):"
        )
        for row in dose_composite.to_dict(orient="records"):
            print(
                f"  rho={float(row['rho']):g}: "
                f"{float(row['bootstrap_mean']):.4f} "
                f"[{float(row['bootstrap_ci95_low']):.4f}, "
                f"{float(row['bootstrap_ci95_high']):.4f}]"
            )
        print(
            "Variance-contraction metric plot: "
            f"{variance_contraction_paths['metric_dose_response_svg']}"
        )
        print(
            "Variance-contraction family/composite plot: "
            f"{variance_contraction_paths['family_composite_dose_response_svg']}"
        )
    if normalized_paths:
        print(f"Normalized-model plot: {normalized_paths['plot_svg']}")


if __name__ == "__main__":
    main()
