"""
Prepare 2p trace CSV exports into the same dataset format as prepare.py outputs.

Expected input CSV format:
- No header row
- Rows = neurons
- Columns = timepoints

Outputs:
- data_processed/<file_stem>.npy
- data_processed/<file_stem>_metadata.json
- output/<file_stem>_diagnostics/*.png
"""

import argparse
import gc
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_PREP_DIR = Path(__file__).resolve().parent
if str(_PREP_DIR) not in sys.path:
    sys.path.insert(0, str(_PREP_DIR))

from prepare_data import export_organized_data

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from helpers.scaling_law_globals import (
    load_scaling_law_globals,
    merge_paths_section,
    merge_prepare_data_section,
    resolve_repo_relative,
    scaling_law_repo_root,
)


def infer_file_stem(csv_path: Path) -> str:
    """Generate a stable output stem from the source CSV filename."""
    stem = csv_path.stem
    stem = stem.replace("Data_Valence_FULL_", "")
    stem = stem.replace("_Session_", "_S")
    stem = stem.replace("__", "_")
    return f"data2p_{stem}"


def load_trace_matrix(csv_path: Path) -> np.ndarray:
    """Load matrix CSV as float32 with shape (n_neurons, n_timepoints)."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    matrix = pd.read_csv(csv_path, header=None, dtype=np.float32).to_numpy()
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix in {csv_path}, got shape {matrix.shape}")
    if matrix.shape[0] < 1 or matrix.shape[1] < 2:
        raise ValueError(f"Unexpected tiny matrix shape {matrix.shape} in {csv_path}")
    return matrix


def matrix_to_wide_df(matrix: np.ndarray) -> pd.DataFrame:
    """
    Convert (n_neurons, n_time) matrix to wide dataframe with:
    sequenceId, itemPosition, neuron_000, neuron_001, ...
    """
    n_neurons, n_time = matrix.shape
    neuron_names = [f"neuron_{i:03d}" for i in range(n_neurons)]

    traces_t = matrix.T  # (n_time, n_neurons)
    df = pd.DataFrame(traces_t, columns=neuron_names)
    df.insert(0, "itemPosition", np.arange(n_time, dtype=np.int64))
    df.insert(0, "sequenceId", 0)
    return df


def filter_low_activity_neurons(
    matrix: np.ndarray,
    min_std: float = 0.0,
    min_abs_max: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """
    Remove neurons with near-flat activity.

    Keeps neurons that satisfy BOTH:
      - std(trace) >= min_std
      - max(abs(trace)) >= min_abs_max
    """
    if min_std < 0 or min_abs_max < 0:
        raise ValueError("min_std and min_abs_max must be >= 0")

    per_neuron_std = np.std(matrix, axis=1)
    per_neuron_abs_max = np.max(np.abs(matrix), axis=1)
    keep_mask = (per_neuron_std >= min_std) & (per_neuron_abs_max >= min_abs_max)

    if not np.any(keep_mask):
        raise ValueError(
            "All neurons were filtered out. Lower --min-std / --min-abs-max thresholds."
        )

    filtered = matrix[keep_mask]
    stats = {
        "n_total": int(matrix.shape[0]),
        "n_kept": int(filtered.shape[0]),
        "n_removed": int(matrix.shape[0] - filtered.shape[0]),
        "pct_removed": float(100.0 * (matrix.shape[0] - filtered.shape[0]) / matrix.shape[0]),
        "std_median_all": float(np.median(per_neuron_std)),
        "std_median_kept": float(np.median(per_neuron_std[keep_mask])),
        "absmax_median_all": float(np.median(per_neuron_abs_max)),
        "absmax_median_kept": float(np.median(per_neuron_abs_max[keep_mask])),
    }
    return filtered, keep_mask, stats


def transform_spike_traces(
    matrix: np.ndarray,
    trace_transform: str = "auto",
    clip_upper_quantile: float = 0.999,
) -> tuple[np.ndarray, dict[str, float | str | None]]:
    """
    Stabilize sparse deconvolved traces before windowing.

    For nonnegative C_dec-like traces, sqrt/log1p compress rare large events so the
    model does not spend all capacity fitting a few extreme peaks. F_dFF-like traces
    with negatives are left untouched in auto mode.
    """
    if not (0.0 < clip_upper_quantile <= 1.0):
        raise ValueError(f"clip_upper_quantile must be in (0, 1], got {clip_upper_quantile}")

    out = matrix.astype(np.float32, copy=True)
    min_value = float(np.nanmin(out))
    chosen = trace_transform
    if trace_transform == "auto":
        chosen = "sqrt" if min_value >= 0.0 else "identity"

    clip_value = None
    if clip_upper_quantile < 1.0:
        clip_value = float(np.nanquantile(out, clip_upper_quantile))
        out = np.minimum(out, clip_value)

    if chosen == "identity":
        pass
    elif chosen == "sqrt":
        if np.nanmin(out) < 0:
            raise ValueError("sqrt transform requires nonnegative traces")
        out = np.sqrt(out)
    elif chosen == "log1p":
        if np.nanmin(out) < 0:
            raise ValueError("log1p transform requires nonnegative traces")
        out = np.log1p(out)
    else:
        raise ValueError("trace_transform must be one of: auto, identity, sqrt, log1p")

    info = {
        "trace_transform_requested": trace_transform,
        "trace_transform_applied": chosen,
        "clip_upper_quantile": float(clip_upper_quantile),
        "clip_upper_value": clip_value,
        "min_value_before_transform": min_value,
    }
    return out, info


def make_strided_data_array(
    matrix: np.ndarray,
    window_length: int,
    window_stride: int,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    Build overlapping windows as a 4D array compatible with the training pipeline.

    Output format:
      data_array: (n_sequences=1, n_subsequences, n_regions, n_time=window_length)
    """
    n_regions, n_time = matrix.shape
    if window_length <= 1:
        raise ValueError(f"window_length must be > 1, got {window_length}")
    if window_stride <= 0:
        raise ValueError(f"window_stride must be > 0, got {window_stride}")
    if n_time < window_length:
        raise ValueError(
            f"Not enough timepoints ({n_time}) for window_length={window_length}"
        )

    starts = np.arange(0, n_time - window_length + 1, window_stride, dtype=np.int64)
    n_sub = int(len(starts))

    # windows -> (n_sub, n_regions, window_length)
    windows = np.stack(
        [matrix[:, s : s + window_length] for s in starts],
        axis=0,
    ).astype(np.float32, copy=False)

    # add n_sequences axis: (1, n_sub, n_regions, window_length)
    data_array = windows[None, ...]
    region_names = [f"neuron_{i:03d}" for i in range(n_regions)]

    subsequence_info = pd.DataFrame(
        {
            "sequenceId": np.zeros(n_sub, dtype=np.int64),
            "subsequenceId": np.arange(n_sub, dtype=np.int64),
            "start_pos": starts,
            "end_pos": starts + (window_length - 1),
            "length": np.full(n_sub, window_length, dtype=np.int64),
            "is_full": np.ones(n_sub, dtype=bool),
        }
    )
    return data_array, region_names, subsequence_info


def create_diagnostic_plots(
    matrix: np.ndarray,
    output_dir: Path,
    plot_prefix: str,
    sample_neurons: int = 12,
) -> None:
    """Save helper plots to quickly validate orientation/scaling/data quality."""
    output_dir.mkdir(parents=True, exist_ok=True)
    n_neurons, n_time = matrix.shape
    t_end = min(100, n_time)
    t = np.arange(t_end)

    # 1) Sample traces over time
    n_show = min(sample_neurons, n_neurons)
    sample_idx = np.linspace(0, n_neurons - 1, num=n_show, dtype=int)
    plt.figure(figsize=(14, 7))
    for idx in sample_idx:
        plt.plot(t, matrix[idx, :t_end], linewidth=0.9, alpha=0.8, label=f"n{idx}")
    plt.title(f"{plot_prefix}: sample neuron traces")
    plt.xlabel(f"Timepoint (10 Hz, 0-{max(0, t_end - 1)})")
    plt.ylabel("Signal")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_dir / f"{plot_prefix}_sample_traces.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 2) Heatmap snapshot to verify structure/drift/noisy channels
    n_time_heat = min(4000, n_time)
    m_heat = matrix[:, :n_time_heat]
    plt.figure(figsize=(14, 6))
    sns.heatmap(
        m_heat,
        cmap="viridis",
        cbar=True,
        xticklabels=False,
        yticklabels=False,
    )
    plt.title(f"{plot_prefix}: neuron x time heatmap (first {n_time_heat} steps)")
    plt.xlabel("Timepoint")
    plt.ylabel("Neuron index")
    plt.tight_layout()
    plt.savefig(output_dir / f"{plot_prefix}_heatmap_first_window.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 3) Value distribution
    sample_flat = matrix[:, :: max(1, n_time // 5000)].reshape(-1)
    plt.figure(figsize=(10, 6))
    plt.hist(sample_flat, bins=120, alpha=0.75, edgecolor="black")
    plt.title(f"{plot_prefix}: value distribution (subsampled)")
    plt.xlabel("Signal value")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_dir / f"{plot_prefix}_value_distribution.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 4) Correlation matrix for first 40 neurons
    n_corr = min(40, n_neurons)
    corr = np.corrcoef(matrix[:n_corr])
    plt.figure(figsize=(8, 7))
    sns.heatmap(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0, square=True)
    plt.title(f"{plot_prefix}: neuron correlation (first {n_corr})")
    plt.tight_layout()
    plt.savefig(output_dir / f"{plot_prefix}_correlation.png", dpi=220, bbox_inches="tight")
    plt.close()


def prepare_single_csv(
    csv_path: Path,
    output_data_dir: Path,
    output_plot_root: Path,
    window_length: int,
    window_stride: int,
    min_std: float,
    min_abs_max: float,
    trace_transform: str,
    clip_upper_quantile: float,
    split_mode: str,
    split_gap_timesteps: int,
    file_stem: str | None = None,
) -> dict[str, Path]:
    """End-to-end conversion from raw 2p csv to project npy/metadata format."""
    matrix = load_trace_matrix(csv_path)
    n_neurons_raw, n_time = matrix.shape
    chosen_stem = file_stem or infer_file_stem(csv_path)

    print("\n" + "=" * 80)
    print(f"PREPARING 2P CSV: {csv_path.name}")
    print("=" * 80)
    print(f"   Input matrix shape (neurons x time): {matrix.shape}")

    matrix, transform_info = transform_spike_traces(
        matrix,
        trace_transform=trace_transform,
        clip_upper_quantile=clip_upper_quantile,
    )
    print(
        f"   Trace transform: {transform_info['trace_transform_applied']} "
        f"(requested={trace_transform}, clip_q={clip_upper_quantile:g})"
    )

    matrix, keep_mask, filter_stats = filter_low_activity_neurons(
        matrix, min_std=min_std, min_abs_max=min_abs_max
    )
    n_neurons_kept = matrix.shape[0]
    print(
        f"   Activity filter (std>={min_std:g}, absmax>={min_abs_max:g}): "
        f"kept {n_neurons_kept}/{n_neurons_raw} neurons "
        f"({filter_stats['pct_removed']:.1f}% removed)"
    )

    per_neuron_std = np.std(matrix, axis=1)
    abs_flat = np.abs(matrix.reshape(-1))

    data_array, region_names, subsequence_info = make_strided_data_array(
        matrix,
        window_length=window_length,
        window_stride=window_stride,
    )
    print(
        f"   Strided windows: n_subsequences={data_array.shape[1]} "
        f"(window={window_length}, stride={window_stride})"
    )

    partition_info = {
        "source_type": "2p_traces_csv",
        "source_file": str(csv_path),
        "n_neurons": int(n_neurons_kept),
        "n_neurons_raw": int(n_neurons_raw),
        "n_neurons_removed_low_activity": int(n_neurons_raw - n_neurons_kept),
        "activity_filter_min_std": float(min_std),
        "activity_filter_min_abs_max": float(min_abs_max),
        "activity_filter_keep_mask": keep_mask.astype(bool).tolist(),
        "n_timepoints": int(n_time),
        "window_length": int(window_length),
        "window_stride": int(window_stride),
        "n_subsequences": int(data_array.shape[1]),
        "split_mode": str(split_mode),
        "split_gap_timesteps": int(split_gap_timesteps),
        "sampling_hz": 10.0,
        "sequence_partition": "100",
        "sequence_frac": 1.0,
        "n_sequence_ids_total": 1,
        "n_sequence_ids_kept": 1,
        "brain_partition": str(len(region_names)),
        "brain_areas_requested": int(len(region_names)),
        "n_brain_regions_in_array": int(len(region_names)),
        "n_brain_regions_available": int(len(region_names)),
        "region_names": list(region_names),
        "median_neuron_std_raw": float(np.median(per_neuron_std)),
        "p10_neuron_std_raw": float(np.percentile(per_neuron_std, 10)),
        "p90_abs_value_raw": float(np.percentile(abs_flat, 90)),
        **transform_info,
    }

    export_paths = export_organized_data(
        data_array,
        region_names,
        subsequence_info,
        output_dir=output_data_dir,
        file_stem=chosen_stem,
        partition_info=partition_info,
    )

    plot_dir = output_plot_root / f"{chosen_stem}_diagnostics"
    create_diagnostic_plots(matrix, plot_dir, plot_prefix=chosen_stem)

    print(f"   [OK] Diagnostics saved under: {plot_dir}")
    print(f"   [OK] Exported array: {export_paths['full_array']}")
    print(f"   [OK] Exported metadata: {export_paths['metadata']}")
    return export_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare 2p-traces CSV (neurons x time) into project .npy format."
    )
    parser.add_argument(
        "--scaling-law-globals",
        type=Path,
        default=None,
        help="Path to scaling_law_globals.json (default: repo root).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        action="append",
        default=[],
        help="CSV path(s) to process; can be repeated.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data_raw/2p_traces"),
        help="Directory to auto-discover CSVs when --csv is not set.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.csv",
        help="Filename glob used with --input-dir (default: *.csv).",
    )
    parser.add_argument(
        "--file-stem",
        type=str,
        default=None,
        help="Optional custom file stem (only valid when exactly one CSV is processed).",
    )
    parser.add_argument(
        "--min-std",
        type=float,
        default=0.0,
        help="Drop neurons with std(trace) below this threshold (default: 0, disabled).",
    )
    parser.add_argument(
        "--min-abs-max",
        type=float,
        default=0.0,
        help="Drop neurons with max(abs(trace)) below this threshold (default: 0, disabled).",
    )
    parser.add_argument(
        "--trace-transform",
        choices=("identity", "sqrt", "log1p", "auto"),
        default="identity",
        help="Trace transform before export. Use identity for raw continuous 2p runs.",
    )
    parser.add_argument(
        "--clip-upper-quantile",
        type=float,
        default=0.999,
        help="Clip rare high events before transform (default: 0.999; set 1.0 to disable).",
    )
    parser.add_argument(
        "--window-length",
        type=int,
        default=180,
        help="Subsequence/window length in timepoints (default: 120).",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=10,
        help="Stride between consecutive windows (default: 10, overlapping windows).",
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        default="blocked",
        choices=("blocked", "random"),
        help="Intended train split mode metadata (blocked recommended to avoid leakage).",
    )
    parser.add_argument(
        "--split-gap-timesteps",
        type=int,
        default=120,
        help="Gap between blocked train/val/test regions in timesteps.",
    )
    args = parser.parse_args()

    full = load_scaling_law_globals(args.scaling_law_globals)
    pcfg = merge_prepare_data_section(full)
    dir_paths = merge_paths_section(full)
    repo = scaling_law_repo_root()

    output_data_dir = resolve_repo_relative(repo, dir_paths["data_processed"])
    output_plot_root = resolve_repo_relative(repo, "output")
    output_data_dir.mkdir(parents=True, exist_ok=True)
    output_plot_root.mkdir(parents=True, exist_ok=True)

    _ = bool(pcfg["only_full_subsequences"])

    csv_paths: list[Path]
    if args.csv:
        csv_paths = [p if p.is_absolute() else resolve_repo_relative(repo, str(p)) for p in args.csv]
    else:
        discover_dir = args.input_dir if args.input_dir.is_absolute() else resolve_repo_relative(repo, str(args.input_dir))
        csv_paths = sorted(discover_dir.glob(args.glob))

    if not csv_paths:
        raise FileNotFoundError("No CSV files found. Provide --csv or check --input-dir/--glob.")
    if args.file_stem and len(csv_paths) != 1:
        raise ValueError("--file-stem can only be used when processing exactly one CSV.")

    print("=" * 80)
    print("2P CSV PREPARATION")
    print("=" * 80)
    print(f"   Files to process: {len(csv_paths)}")
    for p in csv_paths:
        print(f"   - {p}")

    all_exports: list[dict[str, Path]] = []
    for i, csv_path in enumerate(csv_paths):
        file_stem = args.file_stem if i == 0 and args.file_stem else None
        exports = prepare_single_csv(
            csv_path=csv_path,
            output_data_dir=output_data_dir,
            output_plot_root=output_plot_root,
            window_length=args.window_length,
            window_stride=args.window_stride,
            min_std=args.min_std,
            min_abs_max=args.min_abs_max,
            trace_transform=args.trace_transform,
            clip_upper_quantile=args.clip_upper_quantile,
            split_mode=args.split_mode,
            split_gap_timesteps=args.split_gap_timesteps,
            file_stem=file_stem,
        )
        all_exports.append(exports)

    print("\n" + "=" * 80)
    print("2P PREP COMPLETE")
    print("=" * 80)
    for exports in all_exports:
        print(f"  - {exports['full_array']}")
        print(f"    {exports['metadata']}")


if __name__ == "__main__":
    try:
        main()
    finally:
        print("\n[INFO] Cleaning up memory...")
        plt.close("all")
        gc.collect()
