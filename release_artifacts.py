#!/usr/bin/env python3
"""Download and verify the pinned widefield release and checkpoint archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path


DATASET_REPO = "Nethobench/nethobench-widefield-v1"
DATASET_REVISION = "6e45145353ec1b99b22984a4075187cd9bba5408"
CHECKPOINT_MANIFEST_SHA256 = (
    "96e2d74ca4562c826c1101eaa802556d8ae825b91438e54cfbf34795457f689e"
)
DATA_FILES = {
    "data/data100_ba16.npy": (
        697616768,
        "d77a70c594ae851c969cd3242a774607238dbf37ac3e526a66f98d6685668af5",
    ),
    "data/data100_ba16_metadata.json": (
        2240,
        "6b2002c42e39c6e6a8875e7a45a64eb507f81b822379fca76701fd8901839e7f",
    ),
    "data/data-clean-all.parquet": (
        1511370251,
        "7b8ecd89b8f94fac6e377883965227a24613fcbaa498e4779eda0f7145798d98",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, size: int, expected_hash: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != size:
        raise ValueError(f"Wrong size for {path}: {path.stat().st_size} != {size}")
    actual_hash = sha256(path)
    if actual_hash != expected_hash:
        raise ValueError(f"Wrong SHA-256 for {path}: {actual_hash} != {expected_hash}")
    print(f"Verified {path}")


def download_data(root: Path, include_parquet: bool) -> None:
    paths = list(DATA_FILES) if include_parquet else list(DATA_FILES)[:2]
    for relative in paths:
        size, expected_hash = DATA_FILES[relative]
        destination = root / relative
        if destination.exists():
            verify_file(destination, size, expected_hash)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = (
            f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/"
            f"{DATASET_REVISION}/{relative}"
        )
        temporary = destination.with_name(destination.name + ".partial")
        request = urllib.request.Request(url, headers={"User-Agent": "research-reproduction/1.0"})
        with urllib.request.urlopen(request, timeout=60) as source, temporary.open("wb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
        verify_file(temporary, size, expected_hash)
        temporary.replace(destination)


def verify_data(root: Path, include_parquet: bool) -> None:
    paths = list(DATA_FILES) if include_parquet else list(DATA_FILES)[:2]
    for relative in paths:
        size, expected_hash = DATA_FILES[relative]
        verify_file(root / relative, size, expected_hash)


def verify_checkpoints(root: Path) -> None:
    manifest_path = root / "MANIFEST.json"
    if sha256(manifest_path) != CHECKPOINT_MANIFEST_SHA256:
        raise ValueError("Checkpoint manifest differs from the verified anonymous release")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest["files"]
    if manifest["algorithm"] != "sha256" or len(entries) != manifest["file_count"]:
        raise ValueError("Invalid checkpoint manifest")
    for item in entries:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe manifest path: {relative}")
        verify_file(root / relative, item["size_bytes"], item["sha256"])
    print(f"Verified {len(entries)} checkpoint release files")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("download-data", "verify-data"):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, default=Path("data_release/nethobench-widefield-v1"))
        command.add_argument("--include-source-parquet", action="store_true")
    checkpoint_command = commands.add_parser("verify-checkpoints")
    checkpoint_command.add_argument("--root", type=Path, default=Path("checkpoint_release"))
    args = parser.parse_args()
    if args.command == "download-data":
        download_data(args.root, args.include_source_parquet)
    elif args.command == "verify-data":
        verify_data(args.root, args.include_source_parquet)
    else:
        verify_checkpoints(args.root)


if __name__ == "__main__":
    main()
