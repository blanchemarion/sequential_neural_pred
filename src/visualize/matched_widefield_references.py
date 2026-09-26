"""Matched 720-step references for the 222 targets of the widefield evaluation.

Every draw retains all 222 targets and scores four joint groups of 56/56/55/55
over the complete forecast window used in the model comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
EVALUATION = ROOT / "evaluation_results/90_810"
OUTPUT = ROOT / "output/widefield_empirical_ceiling_floor/matched_original_222"
DATA = ROOT / "data_processed/data100_ba16.npy"
IDS = ROOT / "data_processed/processed_val_seq_indices_data100_ba16_Tin90_Tout90_splitseed101.npy"
NORMALIZATION = ROOT / "data_processed/train_norm_stats_data100_ba16_Tin90_Tout90_splitseed101"
SELECTED = EVALUATION / "selected_indices_N222_splitseed101_evalseed102.npy"
GROUND_TRUTH = EVALUATION / "val_seed_102_train_seed_101/long_ground_truth_90_810.npy"
CONDITIONS = ("real_real", "within_channel_temporal_shuffle", "pooled_time_channel_shuffle")
GROUPS = tuple(np.array_split(np.arange(222), 4))
FORECAST_STEPS = 720
# The original multi-horizon implementation produced the existing cache.
MULTI_HORIZON_IMPLEMENTATION_SHA256 = "174134cf34b8e1ef29d172ea792b3ce172baec2f2302774a31ee3dc7f43c29bc"


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def save_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def load_targets_and_candidates():
    """Check target provenance and enumerate finite, nonoverlapping companions."""
    targets_full = np.load(GROUND_TRUTH)
    selected = np.load(SELECTED)
    ids = np.load(IDS)
    data = np.load(DATA, mmap_mode="r")
    mean = np.load(str(NORMALIZATION) + "_mean.npy").reshape(1, 16)
    std = np.load(str(NORMALIZATION) + "_std.npy").reshape(1, 16)
    if targets_full.shape != (222, 810, 16) or selected.shape != (222,):
        raise ValueError("Original evaluation must have exactly 222 ordered targets")
    if len(ids) != len(np.unique(ids)) * data.shape[1] or not np.array_equal(ids, np.repeat(np.unique(ids), data.shape[1])):
        raise ValueError("Validation recording blocks are not complete and ordered")
    if len(set(map(int, selected))) != 222 or np.any(selected < 0) or np.any(selected >= len(ids)):
        raise ValueError("Invalid original evaluation indices")
    if not np.isfinite(targets_full).all() or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Nonfinite targets or training normalization")
    for folder in ("val_seed_102_train_seed_102", "val_seed_102_train_seed103"):
        other = np.load(EVALUATION / folder / "long_ground_truth_90_810.npy", mmap_mode="r")
        if not np.array_equal(targets_full, other):
            raise ValueError(f"Training seeds use different evaluation targets: {folder}")

    traces, candidates, padded = {}, [], []
    for position, index in enumerate(selected):
        index = int(index)
        recording = int(ids[index])
        block = index % data.shape[1]
        if recording not in traces:
            raw = data[recording].transpose(0, 2, 1).reshape(-1, 16)
            traces[recording] = ((raw - mean) / std).astype(np.float32)
        trace = traces[recording]
        context_start = block * data.shape[3] + 50
        target_start = context_start + 90
        target_end = target_start + 720
        real_end = min(target_end, len(trace))
        original = trace[context_start:real_end]
        if not np.allclose(targets_full[position, :len(original)], original, rtol=1e-5, atol=1e-6):
            raise ValueError(f"Original target differs from source at position {position}")
        if real_end < target_end:
            padded.append(position)
        finite = np.isfinite(trace).all(axis=1)
        bad = np.concatenate(([0], np.cumsum(~finite)))
        starts = np.flatnonzero(bad[720:] - bad[:-720] == 0)
        starts = starts[(starts + 720 <= target_start) | (starts >= target_end)]
        if not len(starts):
            raise ValueError(f"No nonoverlapping companion for position {position}")
        candidates.append((recording, starts))
    return targets_full[:, 90:].astype(np.float64), traces, candidates, padded


def reference_arrays(targets, traces, candidates, repetition, seed):
    real, temporal, pooled = (np.empty_like(targets) for _ in CONDITIONS)
    starts = []
    rng = np.random.default_rng(np.random.SeedSequence([seed, repetition]))
    for position, (recording, eligible) in enumerate(candidates):
        start = int(rng.choice(eligible))
        starts.append(start)
        real[position] = traces[recording][start:start + 720]
        for channel in range(16):
            temporal[position, :, channel] = rng.permutation(targets[position, :, channel])
        pooled[position] = rng.permutation(targets[position].reshape(-1)).reshape(720, 16)
    return dict(zip(CONDITIONS, (real, temporal, pooled))), starts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args(argv)
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets, traces, candidates, padded = load_targets_and_candidates()
    signature = {
        "protocol": "matched_original_222_v1",
        "selected_indices_sha256": digest(SELECTED),
        "targets_sha256": digest(GROUND_TRUTH),
        "data_sha256": digest(DATA),
        "ids_sha256": digest(IDS),
        "normalization_sha256": [digest(Path(str(NORMALIZATION) + suffix)) for suffix in ("_mean.npy", "_std.npy")],
        "implementation_sha256": digest(Path(__file__)),
        "repetitions": args.repetitions, "seed": args.seed,
        "group_sizes": [len(group) for group in GROUPS],
        "horizons": [FORECAST_STEPS], "conditions": CONDITIONS,
        "original_padded_positions": padded,
    }
    save_json(args.output_dir / "validation.json", {
        "signature": signature, "targets_per_draw": len(targets),
        "reference_pairs_per_group": [len(group) for group in GROUPS],
        "original_padded_target_positions": padded,
    })
    if args.validate_only:
        print(f"Validated 222 original targets in four groups; {len(padded)} original targets contain padding")
        return

    from neuro_subscores_from_npy_with_sequifier_4split_3seeds import all_scores, evaluator_signature
    signature["evaluator"] = evaluator_signature()
    cache_path = args.output_dir / "matched_reference_scores_720.json"
    cache = {"signature": signature, "scores": {}, "sampled_starts": {}}
    if cache_path.exists() and not args.force_recompute:
        previous = json.loads(cache_path.read_text(encoding="utf-8"))
        if previous.get("signature") == signature:
            cache = previous
    elif not args.force_recompute:
        legacy_path = args.output_dir / "matched_reference_scores.json"
        if legacy_path.exists():
            previous = json.loads(legacy_path.read_text(encoding="utf-8"))
            old_signature = previous.get("signature", {})
            comparable = {
                key: value for key, value in signature.items()
                if key not in ("horizons", "implementation_sha256")
            }
            if (
                old_signature.get("implementation_sha256") == MULTI_HORIZON_IMPLEMENTATION_SHA256
                and old_signature.get("horizons") == [90, 360, FORECAST_STEPS]
                and all(old_signature.get(key) == value for key, value in comparable.items())
            ):
                cache["scores"] = {
                    key: scores for key, scores in previous.get("scores", {}).items()
                    if key.split(":")[1] == str(FORECAST_STEPS)
                }
                cache["sampled_starts"] = previous.get("sampled_starts", {})
                print(f"Reused {len(cache['scores'])} completed 720-step scores", flush=True)
    for repetition in range(args.repetitions):
        reference, starts = reference_arrays(targets, traces, candidates, repetition, args.seed)
        cache["sampled_starts"][str(repetition)] = starts
        for condition, values in reference.items():
            for group_index, group in enumerate(GROUPS, start=1):
                key = f"{repetition}:{FORECAST_STEPS}:{condition}:{group_index}"
                if key not in cache["scores"]:
                    cache["scores"][key] = all_scores(targets[group], values[group])
                    save_json(cache_path, cache)
                    print(f"Scored reference {key}", flush=True)
    records = []
    for key, scores in cache["scores"].items():
        repetition, horizon, condition, group = key.split(":")
        for metric, value in scores.items():
            records.append({"repetition": int(repetition), "horizon": int(horizon),
                            "condition": condition, "group": int(group), "metric": metric, "value": value})
    raw = pd.DataFrame(records)
    raw.to_csv(args.output_dir / "group_scores.csv", index=False)
    per_draw = raw.groupby(["repetition", "horizon", "condition", "metric"], as_index=False).value.mean()
    per_draw.to_csv(args.output_dir / "repetition_group_means.csv", index=False)
    summary = per_draw.groupby(["horizon", "condition", "metric"]).value.agg(
        mean="mean", monte_carlo_sample_sd="std", repetitions="count"
    ).reset_index()
    summary.to_csv(args.output_dir / "monte_carlo_summary.csv", index=False)
    print(f"Saved 222-target matched references to {args.output_dir}")


if __name__ == "__main__":
    main()
