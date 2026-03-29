"""
Generate JSON run configs under configs/ for train_scaling_law.py.

Combines every data_processed/*.npy with T_in in {30, 90, 300} and random_seed in
{101..105}. num_epochs follows data share in the filename (data25→400, data50→200,
data100→100). n_vars and region_names come from the stem (…_baK) and *_metadata.json.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

STEM_RE = re.compile(r"^data(?P<share>\d+)_ba(?P<nvars>\d+)$")

T_IN_CHOICES = (10,) # (30, 90, 300)
SEEDS = (101,) # (101, 102, 103, 104, 105)

# data share in filename → num_epochs
"""SHARE_TO_NUM_EPOCHS = {
    25: 400,
    50: 200,
    100: 100,
}"""
SHARE_TO_NUM_EPOCHS = {
    25: 5,
}


def project_root() -> Path:
    """Repo root (parent of ``src/``); this file lives under ``src/prepare/``."""
    return Path(__file__).resolve().parents[2]


def parse_npy_stem(stem: str) -> tuple[int, int] | None:
    """Return (data_share, n_vars) or None if name does not match data{share}_ba{n}."""
    m = STEM_RE.match(stem)
    if not m:
        return None
    share = int(m.group("share"))
    nvars = int(m.group("nvars"))
    return share, nvars


def num_epochs_for_share(share: int) -> int:
    if share not in SHARE_TO_NUM_EPOCHS:
        raise ValueError(
            f"Unsupported data share {share} in filename (expected one of {sorted(SHARE_TO_NUM_EPOCHS)})"
        )
    return SHARE_TO_NUM_EPOCHS[share]


def load_region_names(metadata_path: Path) -> list[str]:
    with open(metadata_path, encoding="utf-8") as f:
        meta = json.load(f)
    names = meta.get("region_names")
    if not isinstance(names, list) or not names:
        raise ValueError(f"{metadata_path}: missing or empty top-level 'region_names' array")
    if not all(isinstance(x, str) for x in names):
        raise ValueError(f"{metadata_path}: region_names must be a list of strings")
    return names


def relative_data_path(npy_path: Path, root: Path) -> str:
    rel = npy_path.relative_to(root)
    return rel.as_posix()


def build_config_dict(
    *,
    data_path: str,
    t_in: int,
    random_seed: int,
    num_epochs: int,
    n_vars: int,
    region_names: list[str],
) -> dict:
    return {
        "T_in": t_in,
        "data_path": data_path,
        "random_seed": random_seed,
        "num_epochs": num_epochs,
        "n_vars": n_vars,
        "region_names": region_names,
    }


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=root / "data_processed",
        help="Directory containing .npy and matching *_metadata.json files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "configs",
        help="Where to write config_*.json files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned files only; do not write",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    out_dir = args.output_dir.resolve()
    if not data_dir.is_dir():
        print(f"[ERROR] data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    npy_files = sorted(data_dir.glob("*.npy"))
    if not npy_files:
        print(f"[ERROR] No .npy files under {data_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0

    for npy_path in npy_files:
        stem = npy_path.stem
        parsed = parse_npy_stem(stem)
        if parsed is None:
            print(f"[SKIP] {npy_path.name}: stem does not match data{{share}}_ba{{n}}")
            skipped += 1
            continue

        share, n_vars_from_name = parsed
        try:
            nepochs = num_epochs_for_share(share)
        except ValueError as e:
            print(f"[SKIP] {npy_path.name}: {e}")
            skipped += 1
            continue

        meta_path = data_dir / f"{stem}_metadata.json"
        if not meta_path.is_file():
            print(f"[SKIP] {npy_path.name}: missing metadata {meta_path.name}")
            skipped += 1
            continue

        try:
            region_names = load_region_names(meta_path)
        except ValueError as e:
            print(f"[SKIP] {npy_path.name}: {e}")
            skipped += 1
            continue

        if len(region_names) != n_vars_from_name:
            print(
                f"[WARN] {npy_path.name}: len(region_names)={len(region_names)} "
                f"!= n_vars from filename ({n_vars_from_name}); using filename n_vars for 'n_vars' field"
            )

        data_path = relative_data_path(npy_path, root)

        for t_in in T_IN_CHOICES:
            for seed in SEEDS:
                cfg_name = f"config_{stem}_Tin{t_in}_seed{seed}.json"
                out_path = out_dir / cfg_name
                payload = build_config_dict(
                    data_path=data_path,
                    t_in=t_in,
                    random_seed=seed,
                    num_epochs=nepochs,
                    n_vars=n_vars_from_name,
                    region_names=region_names,
                )
                if args.dry_run:
                    print(f"would write {out_path}")
                else:
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(payload, f, indent=2)
                        f.write("\n")
                written += 1

    print(
        f"Done: {written} config file(s) {'planned' if args.dry_run else 'written'} "
        f"under {out_dir} ({skipped} .npy skipped)"
    )


if __name__ == "__main__":
    main()
