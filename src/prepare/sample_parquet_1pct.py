"""Write a parquet keeping all rows for ~1% of sequences (ceil) from data-clean-all.parquet."""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

SEQUENCE_COL = "sequenceId"


def n_sequences_to_keep(n_total: int, sequence_frac: float) -> int:
    if n_total <= 0:
        return 0
    if sequence_frac >= 1.0:
        return n_total
    return min(n_total, max(1, math.ceil(n_total * sequence_frac)))


def sample_sequence_ids(all_ids: np.ndarray, sequence_frac: float, rng: np.random.Generator) -> np.ndarray:
    n_total = len(all_ids)
    n_keep = n_sequences_to_keep(n_total, sequence_frac)
    if n_keep == 0:
        return np.array([], dtype=all_ids.dtype)
    if n_keep >= n_total:
        return all_ids.copy()
    idx = rng.choice(n_total, size=n_keep, replace=False)
    idx.sort()
    return all_ids[idx]


def main():
    root = Path(__file__).resolve().parents[2]  # repo root (this file is under src/prepare/)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "data" / "data-clean-all.parquet",
        help="Source parquet path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "data-clean-all-2pct.parquet",
        help="Output parquet path",
    )
    parser.add_argument(
        "--frac",
        type=float,
        default=0.02,
        help="Fraction of distinct sequences to keep (default: 0.01; count uses ceil)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sequence sampling",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    df = pd.read_parquet(args.input)
    if SEQUENCE_COL not in df.columns:
        raise KeyError(f"Expected column {SEQUENCE_COL!r}")

    all_ids = np.sort(df[SEQUENCE_COL].unique())
    n_total = len(all_ids)
    rng = np.random.default_rng(args.seed)
    chosen = sample_sequence_ids(all_ids, args.frac, rng)
    n_keep = len(chosen)
    subset = df.loc[df[SEQUENCE_COL].isin(chosen)].copy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(args.output, index=False)

    print(f"Sequences: {n_total:,} -> {n_keep:,} (frac={args.frac:g}, ceil rule)")
    print(f"Rows: {len(df):,} -> {len(subset):,}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
