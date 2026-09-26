#!/usr/bin/env python3
"""
Aggregate four-split Nethobench evaluations across independent training seeds.

The script discovers folders named ``val_seed_<V>_train_seed_<T>`` (the
underscore before the training-seed number is optional) below
``evaluation_results/90_810``. Each model and training seed is evaluated on
the same selected validation sequences and the same four contiguous sequence
chunks used by ``neuro_subscores_from_npy_with_sequifier_4split.py``.

Uncertainty is handled as a nested design:

1. scores are retained for every training seed and validation-sequence split;
2. the four split scores are averaged within each training seed;
3. headline error bars are the sample standard deviation (ddof=1) across
   independent training-seed means;
4. within-seed split SD and nested variance components are reported
   separately and are not pooled into the headline seed SD.

Pairwise ranking tests use those cached, split-averaged training-seed scores;
they never invoke Nethobench metric computation. Exact paired sign-flip tests
are primary, paired t-tests are secondary, and Holm-adjusted results are
reported both within each score and across all tested scores.

Pointwise Fidelity is computed on the same aligned arrays and splits as:

``0.65 * Error_score + 0.35 * MI_score``.

By default, outputs are written to
``output/neuro_subscores_from_npy_merged_4split_3seeds``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import random
import sys
import tempfile
import time
from collections import OrderedDict
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Iterable, Mapping

_REPO_ROOT = Path(__file__).resolve().parents[2]
_NETHOBENCH_ROOT = _REPO_ROOT / "nethobench"
if not _NETHOBENCH_ROOT.is_dir():
    raise RuntimeError(
        f"Required Nethobench checkout not found: {_NETHOBENCH_ROOT}"
    )
if str(_NETHOBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_NETHOBENCH_ROOT))

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.stats import t as student_t

from nethobench import compute_neuro_scores
from nethobench.analysis.score_definitions import NEURO_FAMILY_METRICS
from nethobench.neuro.metrics.composites import (
    compute_error_score,
    compute_mi_score,
)
from nethobench.neuro.metrics.definitions import compute_fidelity_composite
from cns_plotting import setup_cnsplots_style
from neuro_scoring_windows import (
    align_forecast_only,
    scoring_start_for_prediction,
)


FULL_SEQUENCE_LENGTH = 810
CONTEXT_STEPS = 90
N_SPLITS = 4
MIN_SEQUENCES_PER_SPLIT = 2
POINTWISE_FIDELITY_KEY = "FIDELITY_SCORE"
FINAL_COMPOSITE_KEY = "FINAL_COMPOSITE_SCORE"
SEED_FOLDER_PATTERN = re.compile(
    r"^val_seed_(?P<validation_seed>\d+)_train_seed_?(?P<training_seed>\d+)$"
)


@dataclass(frozen=True)
class SeedFolder:
    path: Path
    validation_seed: int
    training_seed: int


@dataclass(frozen=True)
class ModelSpec:
    prediction_file: str
    ground_truth_file: str
    color: str


MODEL_SPECS: "OrderedDict[str, ModelSpec]" = OrderedDict(
    [
        (
            "VAR",
            ModelSpec(
                "long_predictions_90_810_VAR_BASELINE.npy",
                "long_ground_truth_90_810.npy",
                "#4DBBD5",
            ),
        ),
        (
            "SSM",
            ModelSpec(
                "long_predictions_90_810_cDMM_SSM.npy",
                "long_ground_truth_90_810.npy",
                "#3C5488",
            ),
        ),
        (
            "RNN",
            ModelSpec(
                "long_predictions_90_810_GRU_AR.npy",
                "long_ground_truth_90_810.npy",
                "#E64B35",
            ),
        ),
        (
            "1_step",
            ModelSpec(
                "long_predictions_90_810_1_step.npy",
                "long_ground_truth_90_810.npy",
                "#7A7A7A",
            ),
        ),
        (
            "AR",
            ModelSpec(
                "long_predictions_90_810_AR_KV.npy",
                "long_ground_truth_90_810.npy",
                "#00A087",
            ),
        ),
        (
            "TF",
            ModelSpec(
                "long_predictions_90_810_TF.npy",
                "long_ground_truth_90_810.npy",
                "#EFC000",
            ),
        ),
        (
            "TF_QL_0.08_KL_0.02",
            ModelSpec(
                "long_predictions_90_810_TF_QTL_0.08_KL_0.02.npy",
                "long_ground_truth_90_810.npy",
                "#B24775",
            ),
        ),
        (
            "sequifier",
            ModelSpec(
                "long_predictions_sequifier_last100.npy",
                "long_ground_truth_sequifier_last100.npy",
                "#7E57C2",
            ),
        ),
    ]
)

_MODEL_COLORS = [spec.color.casefold() for spec in MODEL_SPECS.values()]
if len(_MODEL_COLORS) != len(set(_MODEL_COLORS)):
    raise RuntimeError("Every model must have a unique plotting color")

FAMILY_KEYS = [f"family_{family}" for family in NEURO_FAMILY_METRICS]
COMPARISON_KEYS = [
    *FAMILY_KEYS,
    FINAL_COMPOSITE_KEY,
    POINTWISE_FIDELITY_KEY,
]
DISPLAY_NAMES = {
    "family_distribution": "Distribution",
    "family_temporal_spectral": "Temporal",
    "family_relational": "Relational",
    "family_geometry": "Geometry",
    "family_state_dynamics": "State\nDynamics",
    FINAL_COMPOSITE_KEY: "Nethobench\nComposite",
    POINTWISE_FIDELITY_KEY: "Pointwise\nFidelity",
}
PLOT_MODEL_ORDER = [
    "VAR",
    "RNN",
    "1_step",
    "AR",
    "TF",
    "TF_QL_0.08_KL_0.02",
    "sequifier",
    "SSM",
]
MODEL_DISPLAY_NAMES = {
    **{model_name: model_name for model_name in MODEL_SPECS},
    "TF_QL_0.08_KL_0.02": "TF_QL_KL",
}


def ordered_plot_models(model_names: Iterable[str]) -> list[str]:
    """Return available models in the requested presentation order."""
    available = set(model_names)
    ordered = [
        model_name
        for model_name in PLOT_MODEL_ORDER
        if model_name in available
    ]
    ordered.extend(
        model_name
        for model_name in model_names
        if model_name not in ordered
    )
    return ordered


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=_REPO_ROOT / "evaluation_results" / "90_810",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            _REPO_ROOT
            / "output"
            / "neuro_subscores_from_npy_merged_4split_3seeds_new"
        ),
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=None,
        help="Use a reproducible common subset; default uses every sequence.",
    )
    parser.add_argument(
        "--sequence-selection-seed",
        type=int,
        default=102,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_SPECS),
        default=None,
        help="Optional model subset, primarily useful for focused checks.",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[90, 360, 720],
    )
    parser.add_argument(
        "--skip-horizon",
        action="store_true",
        help="Skip the additional horizon-specific evaluations.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
    )
    parser.add_argument(
        "--statistical-alpha",
        type=float,
        default=0.05,
        help=(
            "Family-wise significance threshold for pairwise model tests "
            "(default: 0.05)."
        ),
    )
    parser.add_argument(
        "--omnibus-permutation-count",
        type=int,
        default=100_000,
        help=(
            "Monte Carlo permutations for blocked omnibus ranking tests "
            "(default: 100000)."
        ),
    )
    parser.add_argument(
        "--omnibus-permutation-seed",
        type=int,
        default=101,
        help="Random seed for blocked omnibus ranking permutations.",
    )
    args = parser.parse_args(argv)
    if args.num_sequences is not None:
        minimum = N_SPLITS * MIN_SEQUENCES_PER_SPLIT
        if args.num_sequences < minimum:
            parser.error(f"--num-sequences must be at least {minimum}")
    if any(horizon <= 0 for horizon in args.horizons):
        parser.error("--horizons must contain positive integers")
    if not 0.0 < args.statistical_alpha < 1.0:
        parser.error("--statistical-alpha must lie strictly between 0 and 1")
    if args.omnibus_permutation_count <= 0:
        parser.error("--omnibus-permutation-count must be positive")
    return args


def discover_seed_folders(evaluation_root: Path) -> list[SeedFolder]:
    discovered: list[SeedFolder] = []
    for path in evaluation_root.iterdir():
        if not path.is_dir():
            continue
        match = SEED_FOLDER_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        discovered.append(
            SeedFolder(
                path=path.resolve(),
                validation_seed=int(match.group("validation_seed")),
                training_seed=int(match.group("training_seed")),
            )
        )
    discovered.sort(key=lambda item: (item.validation_seed, item.training_seed))
    if not discovered:
        raise FileNotFoundError(
            f"No val_seed_<V>_train_seed_<T> folders under {evaluation_root}"
        )
    validation_seeds = {item.validation_seed for item in discovered}
    if len(validation_seeds) != 1:
        raise ValueError(
            "Expected a common validation seed for a paired comparison; found "
            f"{sorted(validation_seeds)}"
        )
    training_seeds = [item.training_seed for item in discovered]
    if len(training_seeds) != len(set(training_seeds)):
        raise ValueError(f"Duplicate training seeds: {training_seeds}")
    return discovered


def validate_inputs(
    seed_folders: Iterable[SeedFolder],
    model_names: Iterable[str],
) -> None:
    missing: list[Path] = []
    for seed_folder in seed_folders:
        for model_name in model_names:
            spec = MODEL_SPECS[model_name]
            for filename in (spec.prediction_file, spec.ground_truth_file):
                path = seed_folder.path / filename
                if not path.is_file():
                    missing.append(path)
    if missing:
        preview = "\n".join(f"  {path}" for path in missing[:20])
        raise FileNotFoundError(f"Missing required tensors:\n{preview}")



def sequence_split_indices(n_sequences: int) -> list[np.ndarray]:
    minimum = N_SPLITS * MIN_SEQUENCES_PER_SPLIT
    if n_sequences < minimum:
        raise ValueError(
            f"Need at least {minimum} sequences, got {n_sequences}"
        )
    parts = list(np.array_split(np.arange(n_sequences), N_SPLITS))
    if any(part.size < MIN_SEQUENCES_PER_SPLIT for part in parts):
        raise ValueError("A sequence split is too small for Nethobench")
    return parts


def common_sequence_indices(
    seed_folders: Iterable[SeedFolder],
    model_names: Iterable[str],
    num_sequences: int | None,
    selection_seed: int,
) -> np.ndarray:
    available_counts: list[int] = []
    for seed_folder in seed_folders:
        for model_name in model_names:
            spec = MODEL_SPECS[model_name]
            pred = np.load(
                seed_folder.path / spec.prediction_file,
                mmap_mode="r",
                allow_pickle=False,
            )
            gt = np.load(
                seed_folder.path / spec.ground_truth_file,
                mmap_mode="r",
                allow_pickle=False,
            )
            if pred.ndim != 3 or gt.ndim != 3:
                raise ValueError(
                    f"{seed_folder.path.name}/{model_name}: expected 3D arrays"
                )
            available_counts.append(min(int(pred.shape[0]), int(gt.shape[0])))
    common_count = min(available_counts)
    requested = common_count if num_sequences is None else num_sequences
    if requested > common_count:
        raise ValueError(
            f"Requested {requested} sequences, but the common minimum is "
            f"{common_count}"
        )
    if requested == common_count:
        return np.arange(common_count, dtype=int)
    rng = np.random.default_rng(selection_seed)
    return np.sort(
        rng.choice(common_count, size=requested, replace=False).astype(int)
    )


def load_aligned_arrays(
    seed_folder: SeedFolder,
    model_name: str,
    selected_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    spec = MODEL_SPECS[model_name]
    pred_raw = np.load(
        seed_folder.path / spec.prediction_file,
        allow_pickle=False,
    )
    gt_raw = np.load(
        seed_folder.path / spec.ground_truth_file,
        allow_pickle=False,
    )
    gt, pred = align_forecast_only(
        gt_raw,
        pred_raw,
        full_sequence_length=FULL_SEQUENCE_LENGTH,
        context_steps=CONTEXT_STEPS,
    )
    if selected_indices[-1] >= gt.shape[0]:
        raise IndexError(
            f"{seed_folder.path.name}/{model_name}: selected sequence index "
            f"{selected_indices[-1]} exceeds aligned count {gt.shape[0]}"
        )
    source_start = scoring_start_for_prediction(
        pred_raw,
        full_sequence_length=FULL_SEQUENCE_LENGTH,
        context_steps=CONTEXT_STEPS,
    )
    return (
        gt[selected_indices].astype(np.float64, copy=False),
        pred[selected_indices].astype(np.float64, copy=False),
        source_start,
    )


def write_nethobench_csv(
    array: np.ndarray,
    path: Path,
    *,
    prediction: bool,
) -> None:
    n_sequences, n_time, n_regions = array.shape
    region_names = [f"R{index}" for index in range(n_regions)]
    sequence_ids = np.repeat(np.arange(n_sequences), n_time)
    values = pd.DataFrame(
        array.reshape(-1, n_regions),
        columns=region_names,
    )
    if prediction:
        values.index = sequence_ids
        values.to_csv(path)
    else:
        values.insert(
            0,
            "itemPosition",
            np.tile(np.arange(n_time), n_sequences),
        )
        values.insert(0, "sequenceId", sequence_ids)
        values.to_csv(path, index=False)


def structural_scores(
    gt: np.ndarray,
    pred: np.ndarray,
) -> dict[str, float]:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        gt_csv = temp_path / "gt.csv"
        pred_csv = temp_path / "pred.csv"
        write_nethobench_csv(gt, gt_csv, prediction=False)
        write_nethobench_csv(pred, pred_csv, prediction=True)
        scores = compute_neuro_scores(pred_csv, gt_csv)
    return {
        key: float(value) if value is not None else float("nan")
        for key, value in scores.items()
    }


def pointwise_fidelity_scores(
    gt: np.ndarray,
    pred: np.ndarray,
) -> dict[str, float]:
    error_score = float(compute_error_score(gt, pred))
    mi_score = float(compute_mi_score(gt, pred))
    fidelity_score = float(
        compute_fidelity_composite(
            {"Error_score": error_score, "MI_score": mi_score}
        )
    )
    return {
        "Error_score": error_score,
        "MI_score": mi_score,
        POINTWISE_FIDELITY_KEY: fidelity_score,
    }


def all_scores(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    # Identical evaluator randomness for model and matched-reference groups.
    numpy_state, python_state = np.random.get_state(), random.getstate()
    try:
        np.random.seed(101)
        random.seed(101)
        scores = structural_scores(gt, pred)
        scores.update(pointwise_fidelity_scores(gt, pred))
        return scores
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluator_signature() -> dict:
    paths = sorted((_NETHOBENCH_ROOT / "nethobench").rglob("*.py"))
    return {
        "entrypoint": "all_scores -> structural_scores -> compute_neuro_scores; same CSV alignment for models and references",
        "evaluator_seed": 101,
        "settings": "existing compute_neuro_scores defaults; four joint groups; unweighted group mean",
        "source_sha256": file_digest(Path(__file__)),
        "nethobench_sources": {str(p.relative_to(_NETHOBENCH_ROOT)): file_digest(p) for p in paths},
    }


def json_signature(
    *,
    kind: str,
    seed_folders: list[SeedFolder],
    model_names: list[str],
    selected_indices: np.ndarray,
    horizons: list[int] | None = None,
) -> dict[str, object]:
    signature: dict[str, object] = {
        "kind": kind,
        "validation_seed": seed_folders[0].validation_seed,
        "training_seeds": [folder.training_seed for folder in seed_folders],
        "seed_folders": [folder.path.name for folder in seed_folders],
        "models": model_names,
        "n_splits": N_SPLITS,
        "selected_sequence_indices": selected_indices.tolist(),
        "forecast_scoring_rule": {
            "full_sequence_length": FULL_SEQUENCE_LENGTH,
            "context_steps_dropped": CONTEXT_STEPS,
        },
        "pointwise_fidelity_formula": (
            "0.65 * Error_score + 0.35 * MI_score"
        ),
        "evaluator": evaluator_signature(),
        "input_sha256": {
            str(folder.path / filename): file_digest(folder.path / filename)
            for folder in seed_folders
            for model in model_names
            for filename in (MODEL_SPECS[model].prediction_file, MODEL_SPECS[model].ground_truth_file)
        },
    }
    manifest = seed_folders[0].path.parent / "manifest.json"
    if manifest.is_file():
        signature["evaluation_manifest_sha256"] = file_digest(manifest)
    if horizons is not None:
        signature["horizons"] = horizons
    return signature


def load_score_cache(
    path: Path,
    signature: Mapping[str, object],
    force_recompute: bool,
) -> dict[str, dict[str, list[dict[str, float]]]]:
    if force_recompute or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    previous = payload.get("_cache_signature")
    if not isinstance(previous, dict):
        print(f"Missing cache signature; ignoring {path}")
        return {}
    # The model list and per-file hashes can grow when a new model is added.
    # Check shared scoring inputs first, then retain only unchanged model runs.
    common_keys = (
        "kind", "validation_seed", "training_seeds", "seed_folders",
        "n_splits", "selected_sequence_indices", "forecast_scoring_rule",
        "pointwise_fidelity_formula", "evaluation_manifest_sha256", "horizons",
    )
    if any(previous.get(key) != signature.get(key) for key in common_keys):
        print(f"Shared cache settings changed; ignoring {path}")
        return {}
    old_evaluator = previous.get("evaluator", {})
    new_evaluator = signature["evaluator"]
    if not isinstance(old_evaluator, dict) or any(
        old_evaluator.get(key) != new_evaluator.get(key)
        for key in ("entrypoint", "evaluator_seed", "settings", "nethobench_sources")
    ):
        print(f"Evaluator changed; ignoring {path}")
        return {}
    # The scorer script hash changed when Sequifier handling was added. Its
    # scoring functions did not change, so this prior cache remains reusable.
    previous_source = old_evaluator.get("source_sha256")
    if previous_source not in (
        new_evaluator["source_sha256"],
        "618ad03e339bb85757cf1a3cbde0c6fb050d46977ee0cbc0ed41eeb1883ad609",
    ):
        print(f"Scoring source changed; ignoring {path}")
        return {}
    old_hashes = previous.get("input_sha256", {})
    new_hashes = signature["input_sha256"]
    old_scores = payload.get("scores", {})
    retained = {}
    folder_by_seed = {
        str(seed): name for seed, name in zip(
            signature["training_seeds"], signature["seed_folders"]
        )
    }
    for seed_key, model_scores in old_scores.items():
        matching_folder = folder_by_seed.get(seed_key)
        if matching_folder is None:
            continue
        for model_name, scores in model_scores.items():
            if model_name not in signature["models"]:
                continue
            filenames = (
                MODEL_SPECS[model_name].prediction_file,
                MODEL_SPECS[model_name].ground_truth_file,
            )
            paths = [
                key for key in new_hashes
                if Path(key).parent.name == matching_folder
                and Path(key).name in filenames
            ]
            if len(paths) != len(set(filenames)) or any(
                old_hashes.get(key) != new_hashes[key] for key in paths
            ):
                continue
            retained.setdefault(seed_key, {})[model_name] = scores
    print(f"Loading reusable model scores from {path}: "
          f"{sum(map(len, retained.values()))} seed/model entries")
    return retained


def save_score_cache(
    path: Path,
    signature: Mapping[str, object],
    scores: Mapping[str, object],
) -> None:
    path.write_text(
        json.dumps(
            {"_cache_signature": signature, "scores": scores},
            indent=2,
        ),
        encoding="utf-8",
    )


def main_split_scores(
    *,
    seed_folders: list[SeedFolder],
    model_names: list[str],
    selected_indices: np.ndarray,
    cache_path: Path,
    force_recompute: bool,
) -> dict[str, dict[str, list[dict[str, float]]]]:
    signature = json_signature(
        kind="nethobench_4split_across_training_seeds",
        seed_folders=seed_folders,
        model_names=model_names,
        selected_indices=selected_indices,
    )
    cached = load_score_cache(cache_path, signature, force_recompute)
    for seed_folder in seed_folders:
        seed_key = str(seed_folder.training_seed)
        cached.setdefault(seed_key, {})
        for model_name in model_names:
            existing = cached[seed_key].get(model_name)
            if isinstance(existing, list) and len(existing) == N_SPLITS:
                print(
                    f"Skipping train seed {seed_key}, {model_name} (cached)"
                )
                continue
            gt, pred, source_start = load_aligned_arrays(
                seed_folder,
                model_name,
                selected_indices,
            )
            print(
                f"Scoring train seed {seed_key}, {model_name}: "
                f"shape={gt.shape}, source={source_start}:"
                f"{source_start + gt.shape[1]}"
            )
            split_scores = [
                all_scores(gt[indices], pred[indices])
                for indices in sequence_split_indices(gt.shape[0])
            ]
            cached[seed_key][model_name] = split_scores
            save_score_cache(cache_path, signature, cached)
    return cached


def scores_to_long(
    scores: Mapping[str, Mapping[str, list[Mapping[str, float]]]],
    seed_folders: list[SeedFolder],
) -> pd.DataFrame:
    folder_by_seed = {
        str(folder.training_seed): folder for folder in seed_folders
    }
    rows: list[dict[str, object]] = []
    for training_seed, model_scores in scores.items():
        seed_folder = folder_by_seed[str(training_seed)]
        for model_name, split_scores in model_scores.items():
            for split_index, score_dict in enumerate(split_scores, start=1):
                for score_key, value in score_dict.items():
                    rows.append(
                        {
                            "validation_seed": seed_folder.validation_seed,
                            "training_seed": int(training_seed),
                            "seed_folder": seed_folder.path.name,
                            "model": model_name,
                            "split": split_index,
                            "score_key": score_key,
                            "score": float(value),
                        }
                    )
    return pd.DataFrame(rows)


def sample_std(values: pd.Series) -> float:
    finite = values.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return float("nan")
    return float(np.std(finite, ddof=1))


def summarize_nested_scores(
    raw_long: pd.DataFrame,
    context_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Return per-seed, across-seed, and observation-level nested summaries.

    The headline SD is computed across training-seed means. The method-of-
    moments nested components use:

    split variance = pooled within-seed variance
    seed variance = max(var(seed means) - split variance / n_splits, 0)
    """
    context_columns = context_columns or []
    score_group = [*context_columns, "model", "score_key"]
    seed_group = [*score_group, "training_seed"]

    per_seed = (
        raw_long.groupby(seed_group, sort=False)["score"]
        .agg(
            seed_mean_over_splits="mean",
            within_seed_split_std=sample_std,
            n_splits="count",
        )
        .reset_index()
    )
    across_seed = (
        per_seed.groupby(score_group, sort=False)["seed_mean_over_splits"]
        .agg(
            mean_across_training_seeds="mean",
            std_across_training_seeds=sample_std,
            n_training_seeds="count",
        )
        .reset_index()
    )
    across_seed["sem_across_training_seeds"] = (
        across_seed["std_across_training_seeds"]
        / np.sqrt(across_seed["n_training_seeds"])
    )

    pooled_split = (
        raw_long.groupby(score_group, sort=False)["score"]
        .agg(pooled_seed_split_std=sample_std)
        .reset_index()
    )
    within_variance = per_seed.assign(
        within_variance=per_seed["within_seed_split_std"] ** 2
    )
    within_variance = (
        within_variance.groupby(score_group, sort=False)["within_variance"]
        .mean()
        .reset_index(name="pooled_within_seed_split_variance")
    )
    summary = across_seed.merge(
        pooled_split,
        on=score_group,
        how="left",
    ).merge(
        within_variance,
        on=score_group,
        how="left",
    )
    seed_mean_variance = summary["std_across_training_seeds"] ** 2
    effective_splits = (
        per_seed.groupby(score_group, sort=False)["n_splits"]
        .mean()
        .reset_index(name="effective_n_splits")
    )
    summary = summary.merge(effective_splits, on=score_group, how="left")
    summary["estimated_training_seed_variance_component"] = np.maximum(
        seed_mean_variance
        - (
            summary["pooled_within_seed_split_variance"]
            / summary["effective_n_splits"]
        ),
        0.0,
    )
    summary["estimated_nested_total_std"] = np.sqrt(
        summary["estimated_training_seed_variance_component"]
        + summary["pooled_within_seed_split_variance"]
    )
    summary["headline_uncertainty"] = (
        "sample SD across training-seed means (ddof=1)"
    )

    observation_report = raw_long.merge(
        per_seed,
        on=seed_group,
        how="left",
    ).merge(
        summary,
        on=score_group,
        how="left",
    )
    return per_seed, summary, observation_report


PAIRWISE_TEST_COLUMNS = [
    "score_key",
    "score_label",
    "higher_ranked_model",
    "lower_ranked_model",
    "higher_rank",
    "lower_rank",
    "higher_model_mean_score",
    "lower_model_mean_score",
    "mean_paired_difference",
    "std_paired_difference",
    "ci95_difference_low",
    "ci95_difference_high",
    "cohen_dz",
    "n_paired_training_seeds",
    "paired_training_seeds",
    "paired_differences",
    "higher_wins",
    "ties",
    "higher_losses",
    "all_seed_differences_positive",
    "exact_sign_flip_p_two_sided",
    "exact_sign_flip_p_holm_within_score",
    "exact_sign_flip_p_holm_global",
    "paired_t_statistic",
    "paired_t_p_two_sided",
    "paired_t_p_holm_within_score",
    "paired_t_p_holm_global",
    "alpha",
    "ranking_significant_exact_holm_within_score",
    "ranking_significant_exact_holm_global",
]


def exact_paired_sign_flip_pvalue(differences: np.ndarray) -> float:
    """Return the exact two-sided randomization p-value for paired differences."""

    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    if differences.size == 0:
        return float("nan")
    observed = abs(float(np.mean(differences)))
    sign_patterns = np.asarray(
        list(product((-1.0, 1.0), repeat=differences.size)),
        dtype=float,
    )
    permuted = np.abs(np.mean(sign_patterns * differences, axis=1))
    tolerance = 16.0 * np.finfo(float).eps * max(1.0, observed)
    return float(np.mean(permuted >= observed - tolerance))


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    """Holm step-down adjustment, preserving NaNs and input order."""

    values = np.asarray(list(p_values), dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    valid_positions = np.flatnonzero(np.isfinite(values))
    if valid_positions.size == 0:
        return adjusted
    ordered_positions = valid_positions[
        np.argsort(values[valid_positions], kind="stable")
    ]
    running_max = 0.0
    n_tests = int(ordered_positions.size)
    for order_index, position in enumerate(ordered_positions):
        candidate = (n_tests - order_index) * float(values[position])
        running_max = max(running_max, candidate)
        adjusted[position] = min(running_max, 1.0)
    return adjusted


def pairwise_model_score_tests(
    per_seed: pd.DataFrame,
    model_names: list[str],
    *,
    alpha: float,
) -> pd.DataFrame:
    """Compare every model pair using cached split-averaged seed scores.

    Training seeds are paired by identifier and are the replication unit.
    Model ordering is descriptive (descending mean score); significance uses
    two-sided tests and therefore does not select a one-sided alternative
    after observing the ranking.
    """

    tested_score_keys = [*FAMILY_KEYS, FINAL_COMPOSITE_KEY]
    rows: list[dict[str, object]] = []
    model_order = {model: index for index, model in enumerate(model_names)}

    for score_key in tested_score_keys:
        score_rows = per_seed[
            (per_seed["score_key"] == score_key)
            & (per_seed["model"].isin(model_names))
        ].copy()
        model_means = (
            score_rows.groupby("model", sort=False)[
                "seed_mean_over_splits"
            ]
            .mean()
            .to_dict()
        )
        ranked_models = sorted(
            model_means,
            key=lambda model: (
                -float(model_means[model]),
                model_order.get(model, len(model_order)),
            ),
        )
        ranks = {
            model: rank
            for rank, model in enumerate(ranked_models, start=1)
        }

        for higher_model, lower_model in combinations(ranked_models, 2):
            higher = score_rows[
                score_rows["model"] == higher_model
            ][["training_seed", "seed_mean_over_splits"]].rename(
                columns={"seed_mean_over_splits": "higher_score"}
            )
            lower = score_rows[
                score_rows["model"] == lower_model
            ][["training_seed", "seed_mean_over_splits"]].rename(
                columns={"seed_mean_over_splits": "lower_score"}
            )
            paired = higher.merge(
                lower,
                on="training_seed",
                how="inner",
                validate="one_to_one",
            ).sort_values("training_seed")
            finite = np.isfinite(paired["higher_score"]) & np.isfinite(
                paired["lower_score"]
            )
            paired = paired.loc[finite]
            differences = (
                paired["higher_score"] - paired["lower_score"]
            ).to_numpy(dtype=float)
            n_paired = int(differences.size)

            mean_difference = (
                float(np.mean(differences))
                if n_paired
                else float("nan")
            )
            difference_std = (
                float(np.std(differences, ddof=1))
                if n_paired >= 2
                else float("nan")
            )
            if n_paired >= 2 and np.isfinite(difference_std):
                difference_sem = difference_std / np.sqrt(n_paired)
                if difference_std <= np.finfo(float).eps:
                    if abs(mean_difference) <= np.finfo(float).eps:
                        t_statistic = 0.0
                        t_pvalue = 1.0
                    else:
                        t_statistic = float(
                            np.copysign(np.inf, mean_difference)
                        )
                        t_pvalue = 0.0
                    cohen_dz = float("nan")
                else:
                    t_statistic = mean_difference / difference_sem
                    t_pvalue = float(
                        2.0
                        * student_t.sf(
                            abs(t_statistic),
                            df=n_paired - 1,
                        )
                    )
                    cohen_dz = mean_difference / difference_std
                critical_t = float(
                    student_t.ppf(0.975, df=n_paired - 1)
                )
                ci_low = mean_difference - critical_t * difference_sem
                ci_high = mean_difference + critical_t * difference_sem
            else:
                t_statistic = float("nan")
                t_pvalue = float("nan")
                cohen_dz = float("nan")
                ci_low = float("nan")
                ci_high = float("nan")

            tie_tolerance = 1e-12
            higher_wins = int(np.sum(differences > tie_tolerance))
            ties = int(np.sum(np.abs(differences) <= tie_tolerance))
            higher_losses = int(np.sum(differences < -tie_tolerance))
            rows.append(
                {
                    "score_key": score_key,
                    "score_label": DISPLAY_NAMES[score_key].replace(
                        "\n", " "
                    ),
                    "higher_ranked_model": higher_model,
                    "lower_ranked_model": lower_model,
                    "higher_rank": ranks[higher_model],
                    "lower_rank": ranks[lower_model],
                    "higher_model_mean_score": float(
                        model_means[higher_model]
                    ),
                    "lower_model_mean_score": float(
                        model_means[lower_model]
                    ),
                    "mean_paired_difference": mean_difference,
                    "std_paired_difference": difference_std,
                    "ci95_difference_low": ci_low,
                    "ci95_difference_high": ci_high,
                    "cohen_dz": cohen_dz,
                    "n_paired_training_seeds": n_paired,
                    "paired_training_seeds": (
                        paired["training_seed"].astype(int).tolist()
                    ),
                    "paired_differences": differences.tolist(),
                    "higher_wins": higher_wins,
                    "ties": ties,
                    "higher_losses": higher_losses,
                    "all_seed_differences_positive": bool(
                        n_paired > 0 and higher_wins == n_paired
                    ),
                    "exact_sign_flip_p_two_sided": (
                        exact_paired_sign_flip_pvalue(differences)
                    ),
                    "paired_t_statistic": t_statistic,
                    "paired_t_p_two_sided": t_pvalue,
                    "alpha": float(alpha),
                }
            )

    tests = pd.DataFrame(rows)
    if tests.empty:
        return pd.DataFrame(columns=PAIRWISE_TEST_COLUMNS)

    for raw_column, adjusted_column in (
        (
            "exact_sign_flip_p_two_sided",
            "exact_sign_flip_p_holm_within_score",
        ),
        (
            "paired_t_p_two_sided",
            "paired_t_p_holm_within_score",
        ),
    ):
        tests[adjusted_column] = np.nan
        for _, indices in tests.groupby("score_key", sort=False).groups.items():
            tests.loc[indices, adjusted_column] = holm_adjust(
                tests.loc[indices, raw_column]
            )

    tests["exact_sign_flip_p_holm_global"] = holm_adjust(
        tests["exact_sign_flip_p_two_sided"]
    )
    tests["paired_t_p_holm_global"] = holm_adjust(
        tests["paired_t_p_two_sided"]
    )
    tests["ranking_significant_exact_holm_within_score"] = (
        tests["exact_sign_flip_p_holm_within_score"] < alpha
    )
    tests["ranking_significant_exact_holm_global"] = (
        tests["exact_sign_flip_p_holm_global"] < alpha
    )
    return tests.reindex(columns=PAIRWISE_TEST_COLUMNS)


def _json_compatible(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _json_compatible(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_pairwise_test_reports(
    tests: pd.DataFrame,
    *,
    csv_path: Path,
    json_path: Path,
    cache_path: Path,
    alpha: float,
) -> None:
    tests.to_csv(csv_path, index=False, float_format="%.10g")
    payload = {
        "source_cache": str(cache_path),
        "scores_recomputed_for_tests": False,
        "score_keys": [*FAMILY_KEYS, FINAL_COMPOSITE_KEY],
        "ranking_rule": (
            "descending mean of the four-split mean within each training "
            "seed, then mean across training seeds"
        ),
        "replication_unit": (
            "training seed; four validation splits are averaged within seed"
        ),
        "primary_test": (
            "exact two-sided paired sign-flip randomization test on "
            "training-seed mean differences"
        ),
        "secondary_test": (
            "two-sided paired Student t-test on the same seed differences"
        ),
        "multiple_testing": {
            "primary_decision": (
                "Holm family-wise correction across all model pairs "
                "separately within each score"
            ),
            "additional_strict_result": (
                "Holm correction across every pair and every reported score"
            ),
        },
        "alpha": float(alpha),
        "small_sample_warning": (
            "With three paired training seeds, the smallest attainable "
            "two-sided exact sign-flip p-value is 0.25. At the configured "
            f"alpha={alpha:g}, collect more independent training seeds for "
            "confirmatory inference."
        ),
        "n_pairwise_tests": int(len(tests)),
        "tests": tests.to_dict(orient="records"),
    }
    json_path.write_text(
        json.dumps(
            _json_compatible(payload),
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def _friedman_statistic_from_ranks(
    ranks: np.ndarray,
) -> tuple[float, float]:
    """Return tie-corrected Friedman Q and Kendall's W."""

    ranks = np.asarray(ranks, dtype=float)
    if ranks.ndim != 2:
        raise ValueError(f"Expected [block, model] ranks, got {ranks.shape}")
    n_blocks, n_models = ranks.shape
    if n_blocks < 2 or n_models < 2:
        raise ValueError(
            "Friedman ranking test needs at least two blocks and two models"
        )
    rank_sums = ranks.sum(axis=0)
    statistic = (
        12.0
        * float(np.sum(rank_sums**2))
        / (n_blocks * n_models * (n_models + 1))
        - 3.0 * n_blocks * (n_models + 1)
    )
    tie_term = 0.0
    for row in ranks:
        _, counts = np.unique(row, return_counts=True)
        tie_term += float(np.sum(counts**3 - counts))
    tie_correction = 1.0 - tie_term / (
        n_blocks * (n_models**3 - n_models)
    )
    if tie_correction <= 0.0:
        raise ValueError("All model ranks are tied within every training seed")
    statistic /= tie_correction
    kendalls_w = statistic / (n_blocks * (n_models - 1))
    return float(statistic), float(kendalls_w)


def omnibus_model_ranking_tests(
    per_seed: pd.DataFrame,
    model_names: list[str],
    *,
    permutation_count: int,
    permutation_seed: int,
    alpha: float,
) -> pd.DataFrame:
    """Blocked permutation tests of the complete model ranking per score.

    Each training seed is a block. Model labels are permuted within every
    block, while a shared permutation is used across scores so the max-T
    correction retains their dependence.
    """

    score_keys = [*FAMILY_KEYS, FINAL_COMPOSITE_KEY]
    training_seeds = sorted(
        int(seed) for seed in per_seed["training_seed"].unique()
    )
    n_seeds = len(training_seeds)
    n_models = len(model_names)
    if n_seeds < 2 or n_models < 2:
        raise ValueError(
            "Omnibus ranking tests require at least two training seeds and "
            "two models"
        )

    observed_statistics: list[float] = []
    kendalls_w_values: list[float] = []
    rank_orders: list[list[str]] = []
    model_mean_scores: list[dict[str, float]] = []
    rank_arrays: list[np.ndarray] = []
    tie_corrections: list[float] = []

    for score_key in score_keys:
        matrix = (
            per_seed[per_seed["score_key"] == score_key]
            .pivot(
                index="training_seed",
                columns="model",
                values="seed_mean_over_splits",
            )
            .reindex(index=training_seeds, columns=model_names)
        )
        if matrix.isna().any().any():
            missing = np.argwhere(matrix.isna().to_numpy())
            raise ValueError(
                f"Missing cached seed/model observations for {score_key}: "
                f"{missing.tolist()}"
            )
        values = matrix.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite cached scores for {score_key}")
        ranks = rankdata(-values, method="average", axis=1)
        statistic, kendalls_w = _friedman_statistic_from_ranks(ranks)

        tie_term = 0.0
        for row in ranks:
            _, counts = np.unique(row, return_counts=True)
            tie_term += float(np.sum(counts**3 - counts))
        tie_correction = 1.0 - tie_term / (
            n_seeds * (n_models**3 - n_models)
        )

        means = values.mean(axis=0)
        order = np.argsort(-means, kind="stable")
        rank_arrays.append(ranks)
        observed_statistics.append(statistic)
        kendalls_w_values.append(kendalls_w)
        tie_corrections.append(tie_correction)
        rank_orders.append([model_names[index] for index in order])
        model_mean_scores.append(
            {
                model_names[index]: float(means[index])
                for index in order
            }
        )

    observed = np.asarray(observed_statistics, dtype=float)
    exceedance_counts = np.zeros(len(score_keys), dtype=np.int64)
    max_t_counts = np.zeros(len(score_keys), dtype=np.int64)
    rng = np.random.default_rng(permutation_seed)
    batch_size = min(5_000, permutation_count)
    completed = 0
    tolerance = 64.0 * np.finfo(float).eps

    while completed < permutation_count:
        current = min(batch_size, permutation_count - completed)
        permutations = np.argsort(
            rng.random((current, n_seeds, n_models)),
            axis=2,
            kind="stable",
        )
        batch_statistics = np.empty(
            (current, len(score_keys)), dtype=float
        )
        for score_index, ranks in enumerate(rank_arrays):
            permuted_ranks = np.take_along_axis(
                np.broadcast_to(
                    ranks,
                    (current, n_seeds, n_models),
                ),
                permutations,
                axis=2,
            )
            rank_sums = permuted_ranks.sum(axis=1)
            statistics = (
                12.0
                * np.sum(rank_sums**2, axis=1)
                / (n_seeds * n_models * (n_models + 1))
                - 3.0 * n_seeds * (n_models + 1)
            )
            statistics /= tie_corrections[score_index]
            batch_statistics[:, score_index] = statistics
            exceedance_counts[score_index] += int(
                np.sum(statistics >= observed[score_index] - tolerance)
            )
        maximum_statistics = batch_statistics.max(axis=1)
        for score_index, observed_statistic in enumerate(observed):
            max_t_counts[score_index] += int(
                np.sum(
                    maximum_statistics
                    >= observed_statistic - tolerance
                )
            )
        completed += current

    raw_pvalues = (exceedance_counts + 1.0) / (
        permutation_count + 1.0
    )
    max_t_pvalues = (max_t_counts + 1.0) / (
        permutation_count + 1.0
    )
    holm_pvalues = holm_adjust(raw_pvalues)

    rows: list[dict[str, object]] = []
    for score_index, score_key in enumerate(score_keys):
        raw_pvalue = float(raw_pvalues[score_index])
        rows.append(
            {
                "score_key": score_key,
                "score_label": DISPLAY_NAMES[score_key].replace("\n", " "),
                "n_training_seeds": n_seeds,
                "n_models": n_models,
                "training_seeds": training_seeds,
                "rank_order": rank_orders[score_index],
                "model_mean_scores": model_mean_scores[score_index],
                "friedman_q": observed_statistics[score_index],
                "kendalls_w": kendalls_w_values[score_index],
                "permutation_count": permutation_count,
                "permutation_seed": permutation_seed,
                "permutation_p_raw": raw_pvalue,
                "permutation_p_monte_carlo_se": float(
                    np.sqrt(
                        raw_pvalue
                        * (1.0 - raw_pvalue)
                        / (permutation_count + 1.0)
                    )
                ),
                "permutation_p_holm_six_scores": float(
                    holm_pvalues[score_index]
                ),
                "permutation_p_max_t_six_scores": float(
                    max_t_pvalues[score_index]
                ),
                "alpha": float(alpha),
                "significant_max_t": bool(
                    max_t_pvalues[score_index] < alpha
                ),
                "interpretation": (
                    "Rejecting the omnibus null means at least one model "
                    "differs for this score; it does not identify a "
                    "significant individual model pair."
                ),
            }
        )
    return pd.DataFrame(rows)


def write_omnibus_test_reports(
    tests: pd.DataFrame,
    *,
    csv_path: Path,
    json_path: Path,
    cache_path: Path,
) -> None:
    csv_frame = tests.copy()
    for column in ("training_seeds", "rank_order", "model_mean_scores"):
        csv_frame[column] = csv_frame[column].map(json.dumps)
    csv_frame.to_csv(csv_path, index=False, float_format="%.10g")
    payload = {
        "source_cache": str(cache_path),
        "scores_recomputed_for_tests": False,
        "test": (
            "blocked Friedman-style Monte Carlo permutation test of all model "
            "labels within each training seed"
        ),
        "null_hypothesis": (
            "all model labels are exchangeable within training seed for the "
            "reported score"
        ),
        "multiplicity": (
            "max-T family-wise correction across the five family scores and "
            "the composite; Holm-adjusted p-values are also reported"
        ),
        "scope_warning": (
            "This is an omnibus ranking test. A significant result establishes "
            "that not all models are equivalent for that score, but cannot "
            "make an underpowered individual pairwise comparison significant."
        ),
        "tests": tests.to_dict(orient="records"),
    }
    json_path.write_text(
        json.dumps(
            _json_compatible(payload),
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def score_matrix(
    summary: pd.DataFrame,
    model_names: list[str],
    score_keys: list[str],
    value_column: str,
) -> pd.DataFrame:
    matrix = summary.pivot(
        index="model",
        columns="score_key",
        values=value_column,
    )
    return matrix.reindex(index=model_names, columns=score_keys)


def setup_plot_style() -> None:
    setup_cnsplots_style(
        {
            "svg.fonttype": "none",
            "axes.linewidth": 0.8,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )


def save_figure_svg_png(fig: plt.Figure, svg_path: Path) -> None:
    """Save SVG and 300-dpi PNG, avoiding Pillow path-open failures."""
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    png_path = svg_path.with_suffix(".png")
    for attempt in range(5):
        try:
            with png_path.open("wb") as png_file:
                fig.savefig(
                    png_file,
                    format="png",
                    dpi=300,
                    bbox_inches="tight",
                )
            break
        except OSError as error:
            if error.errno != 22 or attempt == 4:
                raise
            # Windows can briefly reject an overwrite while another process
            # (for example, an indexer or image previewer) releases the file.
            time.sleep(0.15 * (attempt + 1))


def plot_family_bar(
    summary: pd.DataFrame,
    model_names: list[str],
    output_path: Path,
) -> None:
    setup_plot_style()
    plot_models = ordered_plot_models(model_names)
    means = score_matrix(
        summary,
        plot_models,
        COMPARISON_KEYS,
        "mean_across_training_seeds",
    )
    seed_std = score_matrix(
        summary,
        plot_models,
        COMPARISON_KEYS,
        "std_across_training_seeds",
    ).fillna(0.0)

    x = np.arange(len(COMPARISON_KEYS))
    group_width = 0.86
    bar_width = group_width / len(plot_models) * 0.76
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for model_index, model_name in enumerate(plot_models):
        offsets = (
            x
            + (model_index - (len(plot_models) - 1) / 2) * bar_width
        )
        model_means = means.loc[model_name].to_numpy(dtype=float)
        model_std = seed_std.loc[model_name].to_numpy(dtype=float)
        ax.bar(
            offsets,
            model_means,
            width=bar_width,
            color=MODEL_SPECS[model_name].color,
            edgecolor="none",
            linewidth=0.0,
            label=MODEL_DISPLAY_NAMES[model_name],
            zorder=2,
        )
        ax.errorbar(
            offsets,
            model_means,
            yerr=model_std,
            color="#222222",
            linestyle="none",
            elinewidth=0.7,
            capsize=0,
            alpha=0.4,
            zorder=3,
        )
    reference_start = COMPARISON_KEYS.index(FINAL_COMPOSITE_KEY)
    ax.axvline(
        reference_start - 0.5,
        color="#777777",
        linestyle="--",
        linewidth=1.0,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [DISPLAY_NAMES[key].replace("\n", " ") for key in COMPARISON_KEYS],
        rotation=12,
        ha="right",
    )
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.28)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(
        frameon=False,
        fontsize=8,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
    )
    ax.set_title(
        "Mean ± SD across training seeds "
        "(each seed averaged over 4 evaluation splits)"
    )
    fig.tight_layout()
    save_figure_svg_png(fig, output_path)
    plt.close(fig)


def plot_family_radar(
    summary: pd.DataFrame,
    model_names: list[str],
    output_path: Path,
) -> None:
    setup_plot_style()
    means = score_matrix(
        summary,
        model_names,
        COMPARISON_KEYS,
        "mean_across_training_seeds",
    )
    seed_std = score_matrix(
        summary,
        model_names,
        COMPARISON_KEYS,
        "std_across_training_seeds",
    ).fillna(0.0)
    angles = np.linspace(
        0.0,
        2.0 * np.pi,
        len(COMPARISON_KEYS),
        endpoint=False,
    )
    angles_closed = np.append(angles, angles[0])

    fig = plt.figure(figsize=(8.8, 7.2))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(angles)
    ax.set_xticklabels([DISPLAY_NAMES[key] for key in COMPARISON_KEYS])
    ax.set_yticks(np.linspace(0.2, 1.0, 5))
    ax.grid(color="#D5D5D5", linewidth=0.8)
    for model_name in model_names:
        mean_values = means.loc[model_name].to_numpy(dtype=float)
        std_values = seed_std.loc[model_name].to_numpy(dtype=float)
        color = MODEL_SPECS[model_name].color
        ax.plot(
            angles_closed,
            np.append(mean_values, mean_values[0]),
            color=color,
            linewidth=1.9,
            label=model_name,
        )
        ax.fill_between(
            angles_closed,
            np.append(
                np.clip(mean_values - std_values, 0.0, 1.0),
                np.clip(mean_values - std_values, 0.0, 1.0)[0],
            ),
            np.append(
                np.clip(mean_values + std_values, 0.0, 1.0),
                np.clip(mean_values + std_values, 0.0, 1.0)[0],
            ),
            color=color,
            alpha=0.13,
            linewidth=0,
        )
        ax.scatter(angles, mean_values, s=25, color=color, zorder=3)
    ax.set_title(
        "Nethobench structure and Pointwise Fidelity\n"
        "mean ± SD across training seeds",
        pad=28,
    )
    ax.legend(
        frameon=False,
        fontsize=8,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
    )
    fig.tight_layout()
    save_figure_svg_png(fig, output_path)
    plt.close(fig)


def plot_model_panels(
    summary: pd.DataFrame,
    model_names: list[str],
    output_path: Path,
) -> None:
    """Plot one fidelity-versus-structure panel per model with seed SD bars."""
    setup_plot_style()
    plot_models = ordered_plot_models(model_names)
    panel_keys = [
        POINTWISE_FIDELITY_KEY,
        FINAL_COMPOSITE_KEY,
        *FAMILY_KEYS,
    ]
    panel_labels = [DISPLAY_NAMES[key] for key in panel_keys]
    means = score_matrix(
        summary,
        plot_models,
        panel_keys,
        "mean_across_training_seeds",
    )
    seed_std = score_matrix(
        summary,
        plot_models,
        panel_keys,
        "std_across_training_seeds",
    ).fillna(0.0)

    fig, axes = plt.subplots(
        4,
        2,
        figsize=(12.0, 12.5),
        sharey=True,
    )
    flat_axes = axes.ravel()
    for ax, model_name in zip(flat_axes, plot_models):
        values = means.loc[model_name, panel_keys].to_numpy(dtype=float)
        errors = seed_std.loc[model_name, panel_keys].to_numpy(dtype=float)
        model_color = MODEL_SPECS[model_name].color
        colors = ["#222222", *([model_color] * (len(panel_keys) - 1))]
        bars = ax.bar(
            np.arange(len(panel_keys)),
            values,
            yerr=errors,
            capsize=3,
            color=colors,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.9,
            error_kw={"elinewidth": 0.9, "capthick": 0.9},
        )
        bars[0].set_hatch("//")
        ax.set_title(
            MODEL_DISPLAY_NAMES[model_name],
            color=model_color,
            fontweight="bold",
        )
        ax.set_xticks(np.arange(len(panel_keys)))
        ax.set_xticklabels(
            panel_labels,
            rotation=25,
            ha="right",
            fontsize=8,
        )
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        for bar, value, error in zip(bars, values, errors):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(value + error + 0.025, 0.98),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    for ax in flat_axes[len(plot_models) :]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        if ax.get_visible():
            ax.set_ylabel("Score (higher is better)")
    fig.suptitle(
        "Pointwise predictive fidelity versus Nethobench structural realism\n"
        "mean ± SD across training seeds",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    save_figure_svg_png(fig, output_path)
    plt.close(fig)


def plot_metric_dotplot(
    summary: pd.DataFrame,
    model_names: list[str],
    output_path: Path,
) -> None:
    """Plot one across-training-seed mean dot per model and metric."""
    setup_plot_style()
    plot_models = ordered_plot_models(model_names)
    family_metric_map = OrderedDict(
        (family, list(metrics))
        for family, metrics in NEURO_FAMILY_METRICS.items()
    )
    metric_keys = [
        metric
        for metrics in family_metric_map.values()
        for metric in metrics
    ]
    metric_names = [
        metric.replace("_score01", "").replace("_", " ")
        for metric in metric_keys
    ]
    mean_matrix = score_matrix(
        summary,
        plot_models,
        metric_keys,
        "mean_across_training_seeds",
    )
    y = np.arange(len(metric_keys))
    model_offsets = np.linspace(-0.28, 0.28, len(plot_models))
    fig, ax = plt.subplots(figsize=(9.4, 7.6))
    fig.subplots_adjust(left=0.36, right=0.985, top=0.86, bottom=0.10)

    family_blocks: list[tuple[str, int, int]] = []
    start = 0
    for family_index, (family, metrics) in enumerate(
        family_metric_map.items()
    ):
        end = start + len(metrics) - 1
        family_blocks.append((family, start, end))
        if family_index % 2 == 0:
            ax.axhspan(
                start - 0.5,
                end + 0.5,
                color="#F6F6F6",
                zorder=0,
            )
        if end < len(metric_keys) - 1:
            ax.axhline(
                end + 0.5,
                color="#B0B0B0",
                lw=1.0,
                ls=(0, (3, 3)),
            )
        start = end + 1

    handles = []
    for model_offset, model_name in zip(model_offsets, plot_models):
        color = MODEL_SPECS[model_name].color
        handle = ax.scatter(
            mean_matrix.loc[model_name].to_numpy(dtype=float),
            y + model_offset,
            marker="o",
            s=58,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            label=MODEL_DISPLAY_NAMES[model_name],
            zorder=3,
        )
        handles.append(handle)

    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(np.arange(0.0, 1.01, 0.2))
    ax.set_xlabel(
        "Mean Nethobench subscore across training seeds",
        fontsize=11,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(metric_names, fontsize=10)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#D0D0D0", linewidth=0.8)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.spines[["top", "right"]].set_visible(False)
    transform = mtransforms.blended_transform_factory(
        ax.transAxes,
        ax.transData,
    )
    for family, start, end in family_blocks:
        ax.annotate(
            family.replace("_", " ").title(),
            xy=(0, 0.5 * (start + end)),
            xycoords=transform,
            xytext=(-112, 0),
            textcoords="offset points",
            rotation=60,
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#555555",
        )
    fig.legend(
        handles,
        [MODEL_DISPLAY_NAMES[model_name] for model_name in plot_models],
        frameon=False,
        fontsize=9,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.66, 0.975),
    )
    save_figure_svg_png(fig, output_path)
    plt.close(fig)


def horizon_split_scores(
    *,
    seed_folders: list[SeedFolder],
    model_names: list[str],
    selected_indices: np.ndarray,
    horizons: list[int],
    cache_path: Path,
    force_recompute: bool,
) -> dict[str, dict[str, dict[str, list[dict[str, float]]]]]:
    signature = json_signature(
        kind="nethobench_horizon_4split_across_training_seeds",
        seed_folders=seed_folders,
        model_names=model_names,
        selected_indices=selected_indices,
        horizons=horizons,
    )
    cached = load_score_cache(cache_path, signature, force_recompute)
    for seed_folder in seed_folders:
        seed_key = str(seed_folder.training_seed)
        cached.setdefault(seed_key, {})
        for model_name in model_names:
            cached[seed_key].setdefault(model_name, {})
            if all(
                isinstance(cached[seed_key][model_name].get(str(horizon)), list)
                and len(cached[seed_key][model_name][str(horizon)]) == N_SPLITS
                for horizon in horizons
            ):
                print(f"Skipping all horizons for train seed {seed_key}, {model_name} (cached)")
                continue
            gt, pred, _ = load_aligned_arrays(
                seed_folder,
                model_name,
                selected_indices,
            )
            for horizon in horizons:
                horizon_key = str(horizon)
                existing = cached[seed_key][model_name].get(horizon_key)
                if isinstance(existing, list) and len(existing) == N_SPLITS:
                    print(
                        f"Skipping horizon train seed {seed_key}, "
                        f"{model_name}, H={horizon} (cached)"
                    )
                    continue
                # load_aligned_arrays returns forecast-only tensors, so every
                # cumulative horizon begins at forecast-relative timestep 0.
                gt_h = gt[:, :horizon]
                pred_h = pred[:, :horizon]
                if gt_h.shape[1] != horizon:
                    raise ValueError(
                        f"{seed_folder.path.name}/{model_name}: horizon "
                        f"{horizon} produced shape {gt_h.shape}"
                    )
                print(
                    f"Scoring horizon train seed {seed_key}, {model_name}, "
                    f"H={horizon}"
                )
                cached[seed_key][model_name][horizon_key] = [
                    all_scores(gt_h[indices], pred_h[indices])
                    for indices in sequence_split_indices(gt_h.shape[0])
                ]
                save_score_cache(cache_path, signature, cached)
    return cached


def horizon_scores_to_long(
    scores: Mapping[
        str,
        Mapping[str, Mapping[str, list[Mapping[str, float]]]],
    ],
    seed_folders: list[SeedFolder],
) -> pd.DataFrame:
    folder_by_seed = {
        str(folder.training_seed): folder for folder in seed_folders
    }
    rows: list[dict[str, object]] = []
    for training_seed, model_scores in scores.items():
        seed_folder = folder_by_seed[str(training_seed)]
        for model_name, horizon_scores in model_scores.items():
            for horizon, split_scores in horizon_scores.items():
                for split_index, score_dict in enumerate(
                    split_scores,
                    start=1,
                ):
                    for score_key, value in score_dict.items():
                        rows.append(
                            {
                                "validation_seed": (
                                    seed_folder.validation_seed
                                ),
                                "training_seed": int(training_seed),
                                "seed_folder": seed_folder.path.name,
                                "model": model_name,
                                "horizon": int(horizon),
                                "split": split_index,
                                "score_key": score_key,
                                "score": float(value),
                            }
                        )
    return pd.DataFrame(rows)


def plot_horizon_scores(
    summary: pd.DataFrame,
    model_names: list[str],
    horizons: list[int],
    output_dir: Path,
    result_tag: str,
) -> None:
    setup_plot_style()
    plot_models = ordered_plot_models(model_names)
    horizon_seconds = np.asarray(horizons, dtype=float) / 30.0
    # Symmetric, model-specific offsets reduce overlap without changing the
    # evaluated rollout durations represented by the shared tick locations.
    offset_step_seconds = 0.12
    centered_offsets = (
        np.arange(len(plot_models)) - (len(plot_models) - 1) / 2.0
    ) * offset_step_seconds
    model_offsets = dict(zip(plot_models, centered_offsets))

    def plot_score_axis(ax: plt.Axes, score_key: str) -> list[object]:
        plotted_lows: list[float] = []
        plotted_highs: list[float] = []
        handles: list[object] = []
        for model_name in plot_models:
            model_rows = summary[
                (summary["model"] == model_name)
                & (summary["score_key"] == score_key)
            ].sort_values("horizon")
            if model_rows.empty:
                continue
            means = model_rows[
                "mean_across_training_seeds"
            ].to_numpy(dtype=float)
            errors = model_rows[
                "std_across_training_seeds"
            ].fillna(0.0).to_numpy(dtype=float)
            seconds = (
                model_rows["horizon"].to_numpy(dtype=float) / 30.0
                + model_offsets[model_name]
            )
            finite = np.isfinite(means) & np.isfinite(errors)
            plotted_lows.extend((means[finite] - errors[finite]).tolist())
            plotted_highs.extend((means[finite] + errors[finite]).tolist())
            ax.errorbar(
                seconds,
                means,
                yerr=errors,
                color=MODEL_SPECS[model_name].color,
                linestyle="none",
                elinewidth=0.7,
                capsize=0,
                alpha=0.4,
                zorder=1,
            )
            handles.append(
                ax.plot(
                    seconds,
                    means,
                    color=MODEL_SPECS[model_name].color,
                    marker="o",
                    markersize=3.5,
                    linewidth=1.5,
                    label=MODEL_DISPLAY_NAMES[model_name],
                    zorder=3,
                )[0]
            )
        ax.set_xticks(horizon_seconds)
        ax.set_xlabel("Evaluated rollout duration (s)")
        ax.set_ylabel("Score")
        if plotted_lows and plotted_highs:
            plotted_min = float(np.min(plotted_lows))
            plotted_max = float(np.max(plotted_highs))
            padding = max(0.08 * (plotted_max - plotted_min), 0.01)
            ax.set_ylim(plotted_min - padding, plotted_max + padding)
        ax.grid(axis="y", color="#D7D7D7", linewidth=0.55, alpha=0.55)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        return handles

    for score_key in COMPARISON_KEYS:
        fig, ax = plt.subplots(figsize=(6.0, 4.5))
        plot_score_axis(ax, score_key)
        ax.set_ylabel(DISPLAY_NAMES[score_key].replace("\n", " ") + " score")
        ax.legend(frameon=False, fontsize=8, ncol=2)
        fig.text(
            0.5,
            0.01,
            "Model positions are offset horizontally for readability; "
            "ticks show the evaluated durations.",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#555555",
        )
        fig.tight_layout(rect=(0, 0.045, 1, 1))
        safe_name = score_key.lower().replace("final_", "")
        output_path = output_dir / (
            f"horizon_{safe_name}_{result_tag}_across_training_seeds_std.svg"
        )
        save_figure_svg_png(fig, output_path)
        plt.close(fig)

    panel_keys = [*FAMILY_KEYS, FINAL_COMPOSITE_KEY]
    fig, axes = plt.subplots(3, 3, figsize=(10.2, 8.3))
    shared_handles: list[object] = []
    for ax, score_key in zip(axes.flat, panel_keys):
        handles = plot_score_axis(ax, score_key)
        if handles and not shared_handles:
            shared_handles = handles
        ax.set_title(DISPLAY_NAMES[score_key].replace("\n", " "), fontsize=10)
    legend_ax = axes.flat[len(panel_keys)]
    legend_ax.axis("off")
    legend_ax.legend(
        shared_handles,
        [MODEL_DISPLAY_NAMES[name] for name in plot_models],
        loc="center",
        frameon=False,
        fontsize=8.5,
        ncol=2,
    )
    for ax in axes.flat[len(panel_keys) + 1:]:
        ax.axis("off")
    fig.text(
        0.5,
        0.008,
        "Model positions are offset horizontally for readability; "
        "ticks show the evaluated durations.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1), w_pad=1.2, h_pad=1.5)
    panel_path = output_dir / (
        f"horizon_family_composite_panels_{result_tag}"
        "_across_training_seeds_std.svg"
    )
    save_figure_svg_png(fig, panel_path)
    plt.close(fig)

    if not {90, 720}.issubset(set(horizons)):
        print("Skipping horizon heatmaps: horizons 90 and 720 are required.")
        return

    family_summary = summary[summary["score_key"].isin(FAMILY_KEYS)]

    def family_matrix(horizon: int) -> pd.DataFrame:
        return (
            family_summary[family_summary["horizon"] == horizon]
            .pivot(
                index="score_key",
                columns="model",
                values="mean_across_training_seeds",
            )
            .reindex(index=FAMILY_KEYS, columns=plot_models)
        )

    mean_matrix = family_matrix(720)
    change_matrix = mean_matrix - family_matrix(90)
    max_abs_change = max(
        float(np.nanmax(np.abs(change_matrix.to_numpy(dtype=float)))),
        0.01,
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(12.0, 4.6), constrained_layout=True
    )
    heatmap_specs = (
        (mean_matrix, "Mean family score at 24 s", "viridis", 0.0, 1.0),
        (
            change_matrix,
            "Change from 3 to 24 s",
            "RdBu_r",
            -max_abs_change,
            max_abs_change,
        ),
    )
    for ax, (matrix, title, cmap, vmin, vmax) in zip(axes, heatmap_specs):
        values = matrix.to_numpy(dtype=float)
        image = ax.imshow(
            values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto"
        )
        ax.set_title(title, fontsize=11)
        ax.set_xticks(np.arange(len(plot_models)))
        ax.set_xticklabels(
            [MODEL_DISPLAY_NAMES[name] for name in plot_models],
            rotation=40,
            ha="right",
            rotation_mode="anchor",
        )
        ax.set_yticks(np.arange(len(FAMILY_KEYS)))
        ax.set_yticklabels(
            [DISPLAY_NAMES[key].replace("\n", " ") for key in FAMILY_KEYS]
        )
        ax.tick_params(length=0, labelsize=8.5)
        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                value = values[row_index, column_index]
                if np.isfinite(value):
                    normalized = (value - vmin) / (vmax - vmin)
                    color = (
                        "white"
                        if normalized < 0.25 or normalized > 0.78
                        else "black"
                    )
                    ax.text(
                        column_index,
                        row_index,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7.5,
                        color=color,
                    )
        fig.colorbar(image, ax=ax, fraction=0.045, pad=0.025)
    heatmap_path = output_dir / (
        f"horizon_family_heatmaps_{result_tag}_mean24s_change3to24s.svg"
    )
    save_figure_svg_png(fig, heatmap_path)
    plt.close(fig)


def write_tables(
    *,
    prefix: str,
    raw_long: pd.DataFrame,
    per_seed: pd.DataFrame,
    summary: pd.DataFrame,
    observation_report: pd.DataFrame,
    output_dir: Path,
) -> None:
    raw_long.to_csv(
        output_dir / f"{prefix}_scores_long.csv",
        index=False,
        float_format="%.10g",
    )
    raw_wide = raw_long.pivot_table(
        index=[
            column
            for column in (
                "validation_seed",
                "training_seed",
                "seed_folder",
                "model",
                "horizon",
                "split",
            )
            if column in raw_long.columns
        ],
        columns="score_key",
        values="score",
    ).reset_index()
    raw_wide.to_csv(
        output_dir / f"{prefix}_scores_wide.csv",
        index=False,
        float_format="%.10g",
    )
    per_seed.to_csv(
        output_dir / f"{prefix}_per_training_seed_summary.csv",
        index=False,
        float_format="%.10g",
    )
    summary.to_csv(
        output_dir / f"{prefix}_across_training_seed_summary.csv",
        index=False,
        float_format="%.10g",
    )
    observation_report.to_csv(
        output_dir / f"{prefix}_scores_with_nested_std.csv",
        index=False,
        float_format="%.10g",
    )


def write_methodology(
    path: Path,
    seed_folders: list[SeedFolder],
    selected_indices: np.ndarray,
    statistical_alpha: float,
) -> None:
    lines = [
        "# Three-training-seed Nethobench aggregation",
        "",
        "## Headline statistic",
        "",
        "For every model and score, the four validation-sequence split values "
        "are first averaged within each training seed. Figures show the mean "
        "of those seed means, with sample SD (ddof=1) across training-seed "
        "means. Training seeds are the independent replication unit.",
        "",
        "The split values are repeated measurements nested within a trained "
        "model, not additional independent training runs. Therefore the 12 "
        "seed×split observations are not pooled for headline error bars.",
        "",
        "The tables additionally report within-seed split SD, naive pooled "
        "seed×split SD for transparency, and method-of-moments nested variance "
        "components. With only three training seeds, all SD estimates should "
        "be interpreted descriptively.",
        "",
        "## Pairwise model tests",
        "",
        "For every Nethobench family score and the final composite, every "
        "model pair is compared using paired training-seed means. The primary "
        "test is an exact two-sided sign-flip randomization test; a paired "
        "two-sided Student t-test, paired mean-difference confidence interval, "
        "and Cohen's dz are also reported. Holm correction is applied across "
        "all model pairs within each score, with an additional global Holm "
        "correction across every score and pair.",
        "",
        "The ranking is descriptive and is established from the observed "
        "mean scores. Tests remain two-sided rather than choosing a one-sided "
        "alternative after seeing that ranking. With only three independent "
        "training seeds, the smallest possible two-sided exact sign-flip "
        "p-value is 0.25, so exact confirmatory significance at "
        f"alpha={statistical_alpha:g} requires more training seeds when the "
        "configured threshold is below 0.25.",
        "",
        "The pairwise reports are generated from the loaded score cache and "
        "the per-training-seed split averages; no metric is recomputed for "
        "these tests.",
        "",
        "## Omnibus model-ranking tests",
        "",
        "A blocked Friedman-style permutation test asks, separately for each "
        "family and the composite, whether all model labels are exchangeable "
        "within training seed. Model labels are independently permuted inside "
        "each seed block. The same permutations are used across scores, "
        "allowing a max-T family-wise correction over the six score-level "
        "tests; Holm-adjusted p-values are also reported.",
        "",
        "This omnibus test has substantially more label permutations than an "
        "individual three-pair sign-flip test and can establish that at least "
        "one model differs. It cannot identify a significant individual pair "
        "and must not be presented as evidence that every position in the "
        "observed ranking is significant.",
        "",
        "## Inputs",
        "",
        f"- training seeds: {[folder.training_seed for folder in seed_folders]}",
        f"- validation seed: {seed_folders[0].validation_seed}",
        f"- seed folders: {[folder.path.name for folder in seed_folders]}",
        f"- selected sequences: {selected_indices.tolist()}",
        f"- sequence splits: {N_SPLITS}",
        "",
        "## Pointwise Fidelity",
        "",
        "`FIDELITY_SCORE = 0.65 * Error_score + 0.35 * MI_score`.",
        "",
        "## Statistical references",
        "",
        "- Bouthillier et al., *Accounting for Variance in Machine Learning "
        "Benchmarks*: https://arxiv.org/abs/2103.03098",
        "- Dodge et al., *Fine-Tuning Pretrained Language Models: Weight "
        "Initializations, Data Orders, and Early Stopping*: "
        "https://arxiv.org/abs/2002.06305",
        "- Bengio and Grandvalet, *No Unbiased Estimator of the Variance of "
        "K-Fold Cross-Validation*: "
        "https://www.jmlr.org/papers/v5/grandvalet04a.html",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> dict[str, Path]:
    args = parse_args(argv)
    evaluation_root = args.evaluation_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_folders = discover_seed_folders(evaluation_root)
    model_names = list(MODEL_SPECS) if args.models is None else args.models
    validate_inputs(seed_folders, model_names)
    selected_indices = common_sequence_indices(
        seed_folders,
        model_names,
        args.num_sequences,
        args.sequence_selection_seed,
    )
    if (evaluation_root / "manifest.json").is_file() and not np.array_equal(selected_indices, np.arange(222)):
        raise ValueError("Corrected protocol requires all 222 manifest positions and fixed groups")
    print(
        "Discovered training seeds:",
        [folder.training_seed for folder in seed_folders],
    )
    print(
        f"Using {selected_indices.size} common validation sequences:",
        selected_indices.tolist()
        if selected_indices.size <= 30
        else f"{selected_indices[:10].tolist()} ...",
    )

    selection_tag = (
        ""
        if args.num_sequences is None
        else f"_n{args.num_sequences}_seed{args.sequence_selection_seed}"
    )
    result_tag = f"90_810_4split_{len(seed_folders)}seeds{selection_tag}"
    cache_path = output_dir / f"scores_cache_{result_tag}.json"
    cached_scores = main_split_scores(
        seed_folders=seed_folders,
        model_names=model_names,
        selected_indices=selected_indices,
        cache_path=cache_path,
        force_recompute=args.force_recompute,
    )
    raw_long = scores_to_long(cached_scores, seed_folders)
    per_seed, summary, observation_report = summarize_nested_scores(raw_long)
    write_tables(
        prefix=result_tag,
        raw_long=raw_long,
        per_seed=per_seed,
        summary=summary,
        observation_report=observation_report,
        output_dir=output_dir,
    )
    tested_models = model_names
    pairwise_tests = pairwise_model_score_tests(
        per_seed,
        tested_models,
        alpha=args.statistical_alpha,
    )
    pairwise_tests_csv = output_dir / (
        f"pairwise_model_score_tests_{result_tag}.csv"
    )
    pairwise_tests_json = output_dir / (
        f"pairwise_model_score_tests_{result_tag}.json"
    )
    write_pairwise_test_reports(
        pairwise_tests,
        csv_path=pairwise_tests_csv,
        json_path=pairwise_tests_json,
        cache_path=cache_path,
        alpha=args.statistical_alpha,
    )
    print(f"Pairwise model tests: {pairwise_tests_csv}")
    print(f"Pairwise model test metadata: {pairwise_tests_json}")
    omnibus_tests = omnibus_model_ranking_tests(
        per_seed,
        tested_models,
        permutation_count=args.omnibus_permutation_count,
        permutation_seed=args.omnibus_permutation_seed,
        alpha=args.statistical_alpha,
    )
    omnibus_tests_csv = output_dir / (
        f"omnibus_model_ranking_tests_{result_tag}.csv"
    )
    omnibus_tests_json = output_dir / (
        f"omnibus_model_ranking_tests_{result_tag}.json"
    )
    write_omnibus_test_reports(
        omnibus_tests,
        csv_path=omnibus_tests_csv,
        json_path=omnibus_tests_json,
        cache_path=cache_path,
    )
    print(f"Omnibus model-ranking tests: {omnibus_tests_csv}")
    print(f"Omnibus model-ranking metadata: {omnibus_tests_json}")

    bar_path = output_dir / (
        f"bar_family_scores_{result_tag}_across_training_seeds_std.svg"
    )
    radar_path = output_dir / (
        f"radar_family_scores_{result_tag}_across_training_seeds_std.svg"
    )
    dotplot_path = output_dir / (
        f"metric_dotplot_{result_tag}_training_seed_values.svg"
    )
    model_panels_path = output_dir / (
        f"model_panels_{result_tag}_across_training_seeds_std.svg"
    )
    plot_family_bar(summary, model_names, bar_path)
    plot_family_radar(summary, model_names, radar_path)
    plot_metric_dotplot(summary, model_names, dotplot_path)
    plot_model_panels(summary, model_names, model_panels_path)

    if not args.skip_horizon:
        horizons = sorted(set(args.horizons))
        horizon_tag = "_".join(str(value) for value in horizons)
        horizon_cache_path = output_dir / (
            f"horizon_scores_cache_{result_tag}_h{horizon_tag}.json"
        )
        cached_horizon = horizon_split_scores(
            seed_folders=seed_folders,
            model_names=model_names,
            selected_indices=selected_indices,
            horizons=horizons,
            cache_path=horizon_cache_path,
            force_recompute=args.force_recompute,
        )
        horizon_raw = horizon_scores_to_long(
            cached_horizon,
            seed_folders,
        )
        (
            horizon_per_seed,
            horizon_summary,
            horizon_observation_report,
        ) = summarize_nested_scores(
            horizon_raw,
            context_columns=["horizon"],
        )
        write_tables(
            prefix=f"horizon_{result_tag}_h{horizon_tag}",
            raw_long=horizon_raw,
            per_seed=horizon_per_seed,
            summary=horizon_summary,
            observation_report=horizon_observation_report,
            output_dir=output_dir,
        )
        plot_horizon_scores(
            horizon_summary,
            model_names,
            horizons,
            output_dir,
            result_tag,
        )

    methodology_path = output_dir / "STATISTICAL_METHOD.md"
    write_methodology(
        methodology_path,
        seed_folders,
        selected_indices,
        args.statistical_alpha,
    )
    print(f"Saved outputs to: {output_dir}")
    return {
        "output_dir": output_dir,
        "cache": cache_path,
        "bar": bar_path,
        "radar": radar_path,
        "metric_dotplot": dotplot_path,
        "model_panels": model_panels_path,
        "pairwise_tests_csv": pairwise_tests_csv,
        "pairwise_tests_json": pairwise_tests_json,
        "omnibus_tests_csv": omnibus_tests_csv,
        "omnibus_tests_json": omnibus_tests_json,
        "methodology": methodology_path,
    }


if __name__ == "__main__":
    main()
