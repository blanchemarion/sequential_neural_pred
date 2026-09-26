#!/usr/bin/env python3
"""Rebuild the 90/810 family-score table and bar figure from released scores.

The bundled cache contains the four split-level NethoBench scores for each
training seed. This script reproduces aggregation and plotting; it does not
recompute predictions or the underlying NethoBench metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "reproducibility" / "scores_cache_90_810_4split_3seeds.json"
REFERENCE = ROOT / "reproducibility" / "90_810_4split_3seeds_across_training_seed_summary.csv"
SOURCE_SHA256 = "f1ed0cba4d577144ab653ceee572ae00497f3d4e5e6017646012daf7295705d4"
TAG = "90_810_4split_3seeds"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "reproduced_90_810_summary",
    )
    args = parser.parse_args()

    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f"Released score cache SHA-256 mismatch: {digest}")
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    signature = payload["_cache_signature"]
    if signature["kind"] != "nethobench_4split_across_training_seeds":
        raise ValueError("Unexpected score cache format")

    sys.path.insert(0, str(ROOT / "src" / "visualize"))
    import neuro_subscores_from_npy_with_sequifier_4split_3seeds as analysis

    folders = [
        analysis.SeedFolder(
            path=Path(folder_name),
            validation_seed=int(signature["validation_seed"]),
            training_seed=int(seed),
        )
        for seed, folder_name in zip(
            signature["training_seeds"], signature["seed_folders"], strict=True
        )
    ]
    raw_long = analysis.scores_to_long(payload["scores"], folders)
    per_seed, summary, observation_report = analysis.summarize_nested_scores(raw_long)

    expected = pd.read_csv(REFERENCE)
    pd.testing.assert_frame_equal(
        summary.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        rtol=1e-8,
        atol=1e-10,
    )

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    analysis.write_tables(
        prefix=TAG,
        raw_long=raw_long,
        per_seed=per_seed,
        summary=summary,
        observation_report=observation_report,
        output_dir=out,
    )
    figure = out / f"bar_family_scores_{TAG}_across_training_seeds_std.svg"
    analysis.plot_family_bar(summary, list(signature["models"]), figure)
    print(f"Verified {len(summary)} summary rows against {REFERENCE.name}")
    print(f"Table: {out / (TAG + '_across_training_seed_summary.csv')}")
    print(f"Figure: {figure}")


if __name__ == "__main__":
    main()
