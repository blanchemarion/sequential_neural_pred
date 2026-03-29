"""
Generate JSON run configs under configs/ for train_scaling_law.py.

Choices for ``T_in``, seeds, and ``share_to_num_epochs`` come from ``scaling_law_globals.json``
(override with ``--scaling-law-globals PATH``).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from helpers.scaling_law_globals import (
    load_scaling_law_globals,
    merge_generate_scaling_configs_section,
    merge_paths_section,
    resolve_repo_relative,
    scaling_law_repo_root,
)

STEM_RE = re.compile(r"^data(?P<share>\d+)_ba(?P<nvars>\d+)$")


def parse_npy_stem(stem: str) -> tuple[int, int] | None:
    """Return (data_share, n_vars) or None if name does not match data{share}_ba{n}."""
    m = STEM_RE.match(stem)
    if not m:
        return None
    share = int(m.group("share"))
    nvars = int(m.group("nvars"))
    return share, nvars


def num_epochs_for_share(share: int, share_to_num_epochs: dict[int, int]) -> int:
    if share not in share_to_num_epochs:
        raise ValueError(
            f"Unsupported data share {share} in filename (expected one of {sorted(share_to_num_epochs)})"
        )
    return share_to_num_epochs[share]


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
    repo = scaling_law_repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scaling-law-globals",
        type=Path,
        default=None,
        help="Path to scaling_law_globals.json (default: <repo>/scaling_law_globals.json)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing .npy and matching *_metadata.json files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write config_*.json files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned files only; do not write",
    )
    args = parser.parse_args()

    full = load_scaling_law_globals(args.scaling_law_globals)
    paths = merge_paths_section(full)
    gsc = merge_generate_scaling_configs_section(full)
    t_in_choices = tuple(gsc["t_in_choices"])
    seeds = tuple(gsc["seeds"])
    share_to_num_epochs: dict[int, int] = gsc["share_to_num_epochs"]

    data_dir = (
        resolve_repo_relative(repo, paths["data_processed"])
        if args.data_dir is None
        else Path(args.data_dir).resolve()
    )
    out_dir = (
        resolve_repo_relative(repo, paths["configs"])
        if args.output_dir is None
        else Path(args.output_dir).resolve()
    )
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
            nepochs = num_epochs_for_share(share, share_to_num_epochs)
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

        data_path = relative_data_path(npy_path, repo)

        for t_in in t_in_choices:
            for seed in seeds:
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
