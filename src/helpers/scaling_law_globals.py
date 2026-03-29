"""
Load ``scaling_law_globals.json`` at the repository root (single source for pipeline defaults).

Scripts accept ``--scaling-law-globals PATH`` to override the default file location.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


def scaling_law_repo_root() -> Path:
    """``sequential_neural_pred/`` (parent of ``src/``)."""
    return Path(__file__).resolve().parents[2]


def default_scaling_law_globals_path() -> Path:
    return scaling_law_repo_root() / "scaling_law_globals.json"


def load_scaling_law_globals(path: Path | None = None) -> dict[str, Any]:
    p = Path(path) if path is not None else default_scaling_law_globals_path()
    if not p.is_file():
        return {}
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def resolve_repo_relative(repo_root: Path, rel: str | Path) -> Path:
    p = Path(rel)
    return p.resolve() if p.is_absolute() else (repo_root / p).resolve()


def merge_paths_section(full: dict[str, Any]) -> dict[str, str]:
    defaults = {
        "data_raw": "data_raw",
        "data_processed": "data_processed",
        "configs": "configs",
        "checkpoints": "checkpoints",
        "predictions": "predictions",
        "evaluation": "evaluation",
    }
    p = full.get("paths")
    if not isinstance(p, dict):
        return defaults.copy()
    out = {**defaults}
    for k, v in p.items():
        if isinstance(v, str):
            out[k] = v
    return out


def merge_prepare_data_section(full: dict[str, Any]) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "parquet_filename": "data-clean-all-2pct.parquet",
        "metadata_json_filename": "data-clean-all.json",
        "npy_partitions": [
            {"name": "100", "sequence_frac": 1.0},
            {"name": "50", "sequence_frac": 0.5},
            {"name": "25", "sequence_frac": 0.25},
        ],
        "brain_region_partitions": [
            {"name": "2", "brain_areas": 2},
            {"name": "4", "brain_areas": 4},
            {"name": "8", "brain_areas": 8},
            {"name": "16", "brain_areas": 16},
        ],
        "sequence_id_col": "sequenceId",
        "sequence_sample_base_seed": 42,
        "brain_region_sample_base_seed": 4242,
        "subsequence_length": 360,
        "only_full_subsequences": True,
        "id_cols": ["sequenceId", "itemPosition"],
    }
    sec = full.get("prepare_data")
    if not isinstance(sec, dict):
        return {**defaults}
    out = {**defaults, **sec}
    if "id_cols" in sec and isinstance(sec["id_cols"], list):
        out["id_cols"] = [str(x) for x in sec["id_cols"]]
    return out


def merge_generate_scaling_configs_section(full: dict[str, Any]) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "t_in_choices": [10],
        "seeds": [101],
        "share_to_num_epochs": {"25": 5, "50": 200, "100": 100},
    }
    sec = full.get("generate_scaling_configs")
    if not isinstance(sec, dict):
        return {**defaults}
    out = {**defaults, **sec}
    if "t_in_choices" in sec and isinstance(sec["t_in_choices"], list):
        out["t_in_choices"] = [int(x) for x in sec["t_in_choices"]]
    if "seeds" in sec and isinstance(sec["seeds"], list):
        out["seeds"] = [int(x) for x in sec["seeds"]]
    if "share_to_num_epochs" in sec and isinstance(sec["share_to_num_epochs"], dict):
        raw = sec["share_to_num_epochs"]
        out["share_to_num_epochs"] = {int(k): int(v) for k, v in raw.items()}
    return out


def train_base_config_from_globals(full: dict[str, Any]) -> dict[str, Any] | None:
    sec = full.get("train_scaling_law")
    if not isinstance(sec, dict):
        return None
    base = sec.get("base_config")
    return copy.deepcopy(base) if isinstance(base, dict) and base else None


def merge_inference_scaling_law_section(full: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "num_sequences": 10,
        "long_pred_length": 810,
        "n_plot_examples": 0,
    }
    sec = full.get("inference_scaling_law")
    if not isinstance(sec, dict):
        return {**defaults}
    out = {**defaults, **sec}
    for k in ("num_sequences", "long_pred_length", "n_plot_examples"):
        if k in out:
            out[k] = int(out[k])
    return out
