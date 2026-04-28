"""
Comprehensive inference evaluation script for both short and long window performance.

This script evaluates model performance on:
1. Short windows: Single autoregressive inference loop (T_out length)
2. Long windows: Multiple autoregressive inference loops (800+ timesteps)

For reproducibility, multiple fixed random seeds are used.
"""
from __future__ import annotations

import torch
import numpy as np
from pathlib import Path
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib as mpl
import pandas as pd
import json
from typing import Dict, List, Tuple
import seaborn as sns
import torch.nn.functional as F
import sys

from itertools import combinations
try:
    from scipy.stats import mannwhitneyu, ttest_ind
except Exception as e:
    mannwhitneyu = None
    ttest_ind = None


_SRC_ROOT = Path(__file__).resolve().parent.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from helpers.preprocess_helpers import (
    load_data,
    reshape_to_examples,
    split_by_sequences,
    create_dataloaders,
    filter_examples_by_nan,
    apply_normalization_nct
)


from models.model_KV_cached_2p import create_model_cached

NUM_SEQUENCES = 10
SHORT_PRED_LENGTH = 30  # Fixed prediction length for short window
LONG_PRED_LENGTH = 420  # Fixed prediction length for long window
SEEDS = [102] #[102, 103, 104]


#MODES = ["AR_KV"]
#MODE_COLORS = {"AR_KV": "#2E86AB"}
MODES = ["TF"]
MODE_COLORS = {"TF": "#2E86AB"}
#MODES = ["1_step"]
#MODE_COLORS = {"1_step": "#2E86AB"}
#MODES = ["TF_QTL_0.1"]
#MODE_COLORS = {"TF_QTL_0.1": "#2E86AB"}
#MODES = ["TF_QTL_0.08_KL_0.02"]
#MODE_COLORS = {"TF_QTL_0.08_KL_0.02": "#2E86AB"}
#MODES = ["TF_QTL_0.08_KL_0.02_MOM_0.03"]
#MODE_COLORS = {"TF_QTL_0.08_KL_0.02_MOM_0.03": "#2E86AB"}
#MODES = ["TF_QTL_0.08_KL_0.02_TRJ_0.03"]
#MODE_COLORS = {"TF_QTL_0.08_KL_0.02_TRJ_0.03": "#2E86AB"}
PLOT_SINGLE_REGION_EXAMPLES = True
MAX_SINGLE_REGION_PLOTS = 12

def load_existing_long_neurobench_runs(
    main_output_dir: Path,
    *,
    config_name: str,
    modes: list[str],
    seeds: list[int],
    show_bars: bool = False,
) -> dict:
    """
    Build all_results_by_seed[config_name]["long"][mode] by reading existing files.

    It will:
      - prefer loading scored CSVs if present (fast)
      - otherwise, it will generate scored CSVs from saved .npy, then score (still no model)

    Expected per seed dir:
      - long_predictions_{config_name}_{mode}.npy
      - long_ground_truth_{config_name}.npy
    Optionally already present:
      - long_predictions_scored_{mode}.csv
      - long_ground_truth_scored.csv
    """
    main_output_dir = Path(main_output_dir)

    all_results_by_seed = {
        config_name: {
            "short": {m: [] for m in modes},
            "long":  {m: [] for m in modes},
        }
    }

    for seed in seeds:
        seed_dir = main_output_dir / f"seed_{seed}"
        if not seed_dir.exists():
            print(f"⚠ Missing seed dir: {seed_dir}")
            continue

        # GT is shared across modes (you save it once)
        gt_npy = seed_dir / f"long_ground_truth_{config_name}.npy"
        if not gt_npy.exists():
            print(f"⚠ Missing GT: {gt_npy}")
            continue
        gt_full = np.load(gt_npy)  # shape (N, T_in+Tpred, n_vars)

        for mode in modes:
            pred_npy = seed_dir / f"long_predictions_{config_name}_{mode}.npy"
            if not pred_npy.exists():
                print(f"⚠ Missing pred: {pred_npy}")
                continue
            pred_full = np.load(pred_npy)

            # Use the *prediction window only* (same convention as evaluate_long_window)
            # IMPORTANT: if you changed pred_start elsewhere, mirror it here.
            # Here pred_start = T_in = 90 for your config.
            # If you want to infer it: use pred_full.shape[1] - LONG_PRED_LENGTH as pred_start
            # but in your pipeline T_in is fixed.
            #pred_start = pred_full.shape[1] - LONG_PRED_LENGTH
            T_in = int(config_name.split("_")[0])  # for "90_360"
            pred_start = T_in

            assert pred_full.shape[1] >= pred_start + LONG_PRED_LENGTH

            pred_scored = pred_full[:, pred_start:, :]
            gt_scored   = gt_full[:, pred_start:, :]

            # Align just in case
            Tpred = min(pred_scored.shape[1], gt_scored.shape[1])
            V = min(pred_scored.shape[2], gt_scored.shape[2])
            pred_scored = pred_scored[:, :Tpred, :V]
            gt_scored   = gt_scored[:, :Tpred, :V]

            # scored CSVs (reuse if they exist, else create them)
            pred_csv = seed_dir / f"long_predictions_scored_{mode}.csv"
            gt_csv   = seed_dir / "long_ground_truth_scored.csv"

            if not pred_csv.exists():
                region_names = [f"roi_{i}" for i in range(pred_scored.shape[2])]
                save_sequences_to_neurobench_csv(pred_scored, pred_csv, region_names=region_names)

            if not gt_csv.exists():
                region_names = [f"roi_{i}" for i in range(gt_scored.shape[2])]
                save_sequences_to_neurobench_csv(gt_scored, gt_csv, region_names=region_names)

            global_scores, seq_scores, mean_scores, std_scores = compute_core_scores(
                predictions_csv=pred_csv,
                ground_truth_csv=gt_csv,
                show_bars=show_bars,
            )

            all_results_by_seed[config_name]["long"][mode].append({
                "global_scores": global_scores,
                "sequence_level_scores": seq_scores,
                "mean_scores": mean_scores,
                "std_scores": std_scores,
            })

    return all_results_by_seed



def save_sequences_to_neurobench_csv(arr: np.ndarray, csv_path: str | Path, region_names=None):
    """
    Save (n_seq, T, n_vars) array to NeuroBench CSV format:
    columns: sequenceId, itemPosition, <region cols...>
    """
    arr = np.asarray(arr)
    assert arr.ndim == 3, f"Expected (n_seq, T, n_vars), got {arr.shape}"
    n_seq, T, n_vars = arr.shape

    if region_names is None:
        region_names = [f"var_{i}" for i in range(n_vars)]
    assert len(region_names) == n_vars, "region_names length must match n_vars"

    # Build dataframe in long form with explicit sequence/time indexing
    df = pd.DataFrame(
        arr.reshape(n_seq * T, n_vars),
        columns=region_names
    )
    df.insert(0, "itemPosition", np.tile(np.arange(T), n_seq))
    df.insert(0, "sequenceId", np.repeat(np.arange(n_seq), T))

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return csv_path

def load_norm_stats(stats_base_path: str):
    base = Path(stats_base_path).with_suffix("")
    mean_path   = base.parent / f"{base.name}_mean.npy"
    std_path    = base.parent / f"{base.name}_std.npy"

    mean = np.load(mean_path)
    std  = np.load(std_path)

    return mean, std


def apply_robust_std_floor_numpy_infer(std: np.ndarray, cfg: dict) -> np.ndarray:
    """Match train_all_regimes_2p.normalize_split_robust_2p std flooring."""
    eps = float(cfg.get("norm_eps", 1e-5))
    floor_abs = float(cfg.get("norm_std_floor_abs", 1e-4))
    frac = float(cfg.get("norm_std_floor_frac_median", 0.25))
    flat = std.astype(np.float64).reshape(-1)
    med = float(np.median(flat))
    floor = max(eps, floor_abs, frac * med)
    return np.maximum(std.astype(np.float64), floor).astype(np.float32)


def load_norm_stats_robust(stats_base_path: str, cfg: dict | None):
    """
    Load mean/std; if cfg carries robust-floor hyperparameters, ensure std satisfies the same floor
    as training (helps older checkpoints saved before robust normalization).
    """
    mean, std = load_norm_stats(stats_base_path)
    if cfg is None:
        return mean, std
    std_before = float(np.median(std.reshape(-1)))
    std = apply_robust_std_floor_numpy_infer(std, cfg)
    std_after = float(np.median(std.reshape(-1)))
    print(f"[infer norm] median(std): {std_before:.6g} -> {std_after:.6g} (robust floor from cfg)")
    return mean, std


def _resolve_existing_file(path_value: str | None, repo_root: Path) -> Path | None:
    """
    Resolve optional checkpoint-config path to an existing file.
    Returns None when path is missing/empty or file does not exist.
    """
    if not path_value:
        return None
    p = Path(path_value)
    if p.is_file():
        return p
    if not p.is_absolute():
        candidate = (repo_root / p).resolve()
        if candidate.is_file():
            return candidate
    return None


def _build_processed_paths_from_cfg(cfg: dict, repo_root: Path) -> tuple[Path | None, Path | None]:
    """
    Rebuild default processed val paths using train_all_regimes run_stem convention:
      {data_stem}_Tin{T_in}_Tout{T_out}_seed{random_seed}
    """
    try:
        data_stem = Path(cfg["data_path"]).stem
        run_stem = (
            f"{data_stem}_Tin{int(cfg['T_in'])}_Tout{int(cfg['T_out'])}_seed{int(cfg['random_seed'])}"
        )
    except Exception:
        return None, None

    proc_root = repo_root / "data_processed"
    val_p = proc_root / f"processed_val_{run_stem}.npy"
    seq_p = proc_root / f"processed_val_seq_indices_{run_stem}.npy"
    return val_p, seq_p




def resize_pos_embedding_1d(old_pos: torch.Tensor, new_len: int) -> torch.Tensor:
    """
    old_pos: (1, old_len, d_model)
    returns: (1, new_len, d_model)
    """
    assert old_pos.ndim == 3 and old_pos.shape[0] == 1
    old_len = old_pos.shape[1]
    if new_len == old_len:
        return old_pos

    # (1, old_len, d) -> (1, d, old_len)
    x = old_pos.permute(0, 2, 1)
    x = F.interpolate(x, size=new_len, mode="linear", align_corners=False)
    # back to (1, new_len, d)
    return x.permute(0, 2, 1)


def strip_compile_prefix(state_dict):
    # If compiled, keys are like "_orig_mod.xxx"
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        return {k[len("_orig_mod."):]: v for k, v in state_dict.items()}
    return state_dict

def load_checkpoint_with_resized_pos(model, checkpoint, device="cpu"):
    sd = checkpoint["model_state_dict"]
    sd = strip_compile_prefix(sd)

    # resize pos_embedding if needed
    if hasattr(model, "pos_embedding") and "pos_embedding" in sd:
        old_pos = sd["pos_embedding"]
        new_pos = model.pos_embedding
        if old_pos.shape != new_pos.shape:
            print(f"[pos_embedding] resizing {tuple(old_pos.shape)} -> {tuple(new_pos.shape)}")
            sd["pos_embedding"] = resize_pos_embedding_1d(
                old_pos.to(new_pos.device, new_pos.dtype),
                new_pos.shape[1]
            )

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing: print("Missing keys:", missing)
    if unexpected: print("Unexpected keys:", unexpected)

    model.eval()
    return checkpoint


def create_inference_model(checkpoint_path, T_in, T_out, device='cpu'):
    print(f"  Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    original_config = checkpoint['config']
    
    model = create_model_cached(
        n_vars=original_config['n_vars'],
        d_model=original_config['d_model'],
        n_heads=original_config['n_heads'],
        n_layers=original_config['n_layers'],
        d_ff=original_config['d_ff'],
        dropout=original_config['dropout'],
        T_in=T_in,
        T_out=T_out,
        #patch_len=original_config.get('patch_len', 1),
        device=device,
        nonnegative_output=original_config.get("nonnegative_output", True),
        output_activation=original_config.get("output_activation", "softplus"),
    )
    
    # Load weights with position embedding resizing
    load_checkpoint_with_resized_pos(model, checkpoint, device=device)
    
    model.eval()
    return model, original_config


def generate_long_sequence(
    model,
    initial_input,
    target_length,
    T_in,
    T_out,
    device="cpu",
    rollout_step_size=1,
    clamp_z=None,
):
    model.eval()
    current_sequence = initial_input.to(device)
    block_offset = 0

    with torch.no_grad():
        while current_sequence.shape[1] < target_length:
            # take last T_in, left-pad if needed
            if current_sequence.shape[1] >= T_in:
                input_chunk = current_sequence[:, -T_in:, :]
            else:
                pad_len = T_in - current_sequence.shape[1]
                pad = current_sequence[:, :1, :].repeat(1, pad_len, 1)  # or zeros
                input_chunk = torch.cat([pad, current_sequence], dim=1)

            # block-relative AR inference (positions reset each block)
            #pred = model.forward_autoregressive_old(input_chunk, block_offset=block_offset)  # explicit
            pred = model.forward_autoregressive_kvcache(input_chunk, block_offset=block_offset)
            pred = pred[:, : max(1, min(int(rollout_step_size), pred.shape[1])), :]
            if clamp_z is not None:
                lo, hi = clamp_z
                pred = torch.clamp(pred, min=float(lo), max=float(hi))
            pred = pred.clamp_min(0.0)
            # Append only the first predicted step by default. This avoids the visible
            # repeated T_out-block artifacts that sparse spike traces amplify.
            current_sequence = torch.cat([current_sequence, pred], dim=1)
            #block_offset += T_out
        
    return current_sequence[:, :target_length, :]




def prepare_input_and_gt(val_examples, val_seq_indices, example_idx, T_in, target_length, MAX_CONTEXT=None):
    """
    Prepare input sequence and ground truth for a given example.
    
    Returns:
        input_tensor: Shape (1, T_in, n_vars)
        gt_sequence: Shape (target_length, n_vars)
    """
    first_example = val_examples[example_idx].T  # (n_time, n_vars)
    seq_id = int(val_seq_indices[example_idx])
    
    # Build context from the start of this subsequence (contiguous timeline).
    if first_example.shape[0] >= T_in:
        input_sequence = first_example[:T_in, :]
    else:
        pad_len = T_in - first_example.shape[0]
        input_sequence = np.concatenate(
            [first_example, np.repeat(first_example[-1:, :], pad_len, axis=0)],
            axis=0
        )
    
    # Build ground truth
    gt_parts = [input_sequence]
    first_example_future = first_example[T_in:, :]
    if first_example_future.size > 0:
        gt_parts.append(first_example_future)
    
    # Get remaining examples from same sequence
    same_sequence_mask = (val_seq_indices == seq_id)
    same_sequence_indices = np.where(same_sequence_mask)[0]
    sorted_seq_indices = sorted(same_sequence_indices)
    start_pos = sorted_seq_indices.index(int(example_idx))
    used_indices = sorted_seq_indices[start_pos + 1:]
    
    for seq_example_idx in used_indices:
        seq_example = val_examples[seq_example_idx].T
        gt_parts.append(seq_example)
    
    gt_sequence = np.concatenate(gt_parts, axis=0)
    
    # Trim/pad to target_length
    if gt_sequence.shape[0] >= target_length:
        gt_sequence = gt_sequence[:target_length, :]
    else:
        remaining = target_length - gt_sequence.shape[0]
        gt_sequence = np.concatenate(
            [gt_sequence, np.repeat(gt_sequence[-1:, :], remaining, axis=0)],
            axis=0
        )
    
    input_tensor = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0)
    return input_tensor, gt_sequence


def evaluate_long_window(
    model, checkpoint_name, mode, T_in, T_out,
    val_examples, val_seq_indices, selected_indices,
    output_dir, region_names, device, seed, target_pred_length=800, config_name=None
):
    """
    Evaluate model performance on long windows (fixed 800 timestep prediction).

    Long-horizon metrics focus on:
      - Local accuracy & drift (chunked MAE)
      - Variance / amplitude preservation
      - Temporal dynamics (PSD)
      - Population manifold structure (latent covariance)

    NEW:
      - baseline_per_sequence: metrics computed for GT vs GT (the "optimum" reference)
        so you can add a "GT" box in plots.
    """
    import numpy as np
    import torch
    import json
    import pandas as pd
    from pathlib import Path
    from sklearn.decomposition import PCA

    print("\n" + "=" * 80)
    print(f"LONG WINDOW EVALUATION: {checkpoint_name} (SEED={seed})")
    print(f"  T_out={T_out}, generating {target_pred_length} timesteps")
    print("=" * 80)

    MAX_CONTEXT = T_in
    target_length = T_in + target_pred_length  # input + fixed prediction window
    output_dir = Path(output_dir)


    all_predictions = []
    all_ground_truth = []

    for idx, example_idx in enumerate(selected_indices):
        print(f"\n  Processing sequence {idx+1}/{len(selected_indices)}...")

        input_tensor, gt_sequence = prepare_input_and_gt(
            val_examples, val_seq_indices, example_idx,
            T_in, target_length, MAX_CONTEXT
        )

        input_tensor = input_tensor.to(device)
        gt_tensor = torch.tensor(gt_sequence, dtype=torch.float32).unsqueeze(0)  # (1, T, V)
        gt_tensor = gt_tensor.to(device)

        long_pred = generate_long_sequence(
            model,
            input_tensor,
            target_length,
            T_in,
            T_out,
            device,
            rollout_step_size=1,
            clamp_z=None,
        )

        all_predictions.append(long_pred.detach().cpu().numpy()[0])
        all_ground_truth.append(gt_sequence)

    all_predictions = np.asarray(all_predictions)
    all_ground_truth = np.asarray(all_ground_truth)

    print(f"\n  Predictions shape: {all_predictions.shape}")
    print(f"  Ground truth shape: {all_ground_truth.shape}")
    print(
        f"  Prediction stats: min={all_predictions.min():.6g}, max={all_predictions.max():.6g}, "
        f"mean={all_predictions.mean():.6g}, negative_frac={(all_predictions < 0).mean():.6f}"
    )
    if np.any(all_predictions < 0):
        print("  [WARNING] Negative values found in saved predictions.")

    # Align shapes if needed
    min_n = min(all_predictions.shape[0], all_ground_truth.shape[0])
    min_t = min(all_predictions.shape[1], all_ground_truth.shape[1])
    min_v = min(all_predictions.shape[2], all_ground_truth.shape[2])
    all_predictions = all_predictions[:min_n, :min_t, :min_v]
    all_ground_truth = all_ground_truth[:min_n, :min_t, :min_v]

    # Save raw arrays
    output_dir.mkdir(parents=True, exist_ok=True)
    if config_name:
        np.save(output_dir / f"long_predictions_{config_name}_{mode}.npy", all_predictions)
        np.save(output_dir / f"long_ground_truth_{config_name}.npy", all_ground_truth)
    else:
        np.save(output_dir / f"long_predictions_{mode}.npy", all_predictions)
        np.save(output_dir / "long_ground_truth.npy", all_ground_truth)

    # Score only prediction window
    pred_start = T_in
    predictions_scored = all_predictions[:, pred_start:, :]
    gt_scored = all_ground_truth[:, pred_start:, :]

    Tpred = min(predictions_scored.shape[1], gt_scored.shape[1])
    predictions_scored = predictions_scored[:, :Tpred, :]
    gt_scored = gt_scored[:, :Tpred, :]

    print(f"  Scored window length: {Tpred}")

    # choose names (ideally the same as training / data columns)
    region_names = [f"roi_{i}" for i in range(predictions_scored.shape[2])]
    # or: region_names = list(original_dataframe.columns)[:min_v]

    pred_csv = output_dir / f"long_predictions_scored_{mode}.csv"
    gt_csv   = output_dir / f"long_ground_truth_scored.csv"

    save_sequences_to_neurobench_csv(predictions_scored, pred_csv, region_names=region_names)
    save_sequences_to_neurobench_csv(gt_scored,          gt_csv,   region_names=region_names)

    print("pred shape:", predictions_scored.shape)
    print("gt shape:  ", gt_scored.shape)
    assert predictions_scored.shape == gt_scored.shape
    assert np.isfinite(gt_scored).any() and np.isfinite(predictions_scored).any()


def plot_prediction_examples(
    predictions, ground_truth, output_dir, mode, window_type,
    n_examples=3, pred_start=None, seed=1
):
    """
    Overlay plots of predictions vs ground truth for ALL regions.
    Prediction is hidden before the dashed vertical line.
    Saves paper-ready SVG figures.
    """

    # Keep text editable in SVG (useful for Inkscape)
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["axes.linewidth"] = 0.8

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_sequences = min(n_examples, predictions.shape[0])

    for seq_idx in range(n_sequences):
        fig, ax = plt.subplots(
            figsize=(7, 2.2)
        )

        pred_seq = predictions[seq_idx]   # (T, n_vars)
        gt_seq   = ground_truth[seq_idx]  # (T, n_vars)
        T, n_vars = pred_seq.shape
        time_steps = np.arange(T)

        # pred_start = index where forecast begins
        this_pred_start = T // 2 if pred_start is None else int(pred_start)
        this_pred_start = max(0, min(this_pred_start, T))

        # Light grid behind traces
        ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)

        # Forecast start marker
        ax.axvline(
            x=this_pred_start,
            color="#8C8C8C",
            linestyle=(0, (4, 2)),
            linewidth=1.0,
            alpha=0.9,
            label="Prediction start",
            zorder=2,
        )

        for r in range(n_vars):
            # Ground truth over full timeline
            ax.plot(
                time_steps,
                gt_seq[:, r],
                color="#9A9A9A",
                linewidth=1.0,
                alpha=0.35,
                label="Ground truth" if r == 0 else None,
                zorder=1,
            )

            # Prediction only after pred_start
            y_pred = pred_seq[:, r].copy()
            y_pred[:this_pred_start] = np.nan

            ax.plot(
                time_steps,
                y_pred,
                color="#2E86AB",
                linewidth=1.2,
                alpha=0.45,
                #label=f"{mode} prediction" if r == 0 else None,
                label="1_step prediction" if r == 0 else None,
                zorder=3,
            )

        # Labels
        #ax.set_xlabel("Time step", fontsize=11)
        ax.set_ylabel("All regions", fontsize=11)

        # Tick styling
        ax.tick_params(axis="both", labelsize=10)

        # Cleaner legend
        ax.legend(
            loc="upper right",
            frameon=True,
            framealpha=0.95,
            edgecolor="#CCCCCC",
            fontsize=8,
            handlelength=2.5,
        )

        # Slightly cleaner limits
        ax.margins(x=0.01)

        plt.tight_layout()

        output_path = output_dir / f"{window_type}_example_{mode}_seq{seq_idx+1}_seed{seed}.svg"
        fig.savefig(output_path, format="svg")
        plt.close(fig)

    print(f"  ✓ Saved {n_sequences} example plots for {mode} ({window_type})")


def plot_prediction_examples_single_region(
    predictions,
    ground_truth,
    output_dir,
    mode,
    window_type,
    n_examples=3,
    pred_start=None,
    seed=1,
    max_regions=None,
):
    """
    Plot one region per figure (prediction vs ground truth).
    Useful for inspecting behavior hidden in all-region overlays.
    """
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["axes.linewidth"] = 0.8

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_sequences = min(n_examples, predictions.shape[0])
    n_vars_total = int(predictions.shape[2])
    n_regions = n_vars_total if max_regions is None else min(int(max_regions), n_vars_total)

    for seq_idx in range(n_sequences):
        pred_seq = predictions[seq_idx]   # (T, n_vars)
        gt_seq = ground_truth[seq_idx]    # (T, n_vars)
        T, n_vars = pred_seq.shape
        time_steps = np.arange(T)

        this_pred_start = T // 2 if pred_start is None else int(pred_start)
        this_pred_start = max(0, min(this_pred_start, T))

        for r in range(n_regions):
            fig, ax = plt.subplots(figsize=(7, 2.2))
            ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
            ax.set_axisbelow(True)

            ax.axvline(
                x=this_pred_start,
                color="#8C8C8C",
                linestyle=(0, (4, 2)),
                linewidth=1.0,
                alpha=0.9,
                label="Prediction start",
                zorder=2,
            )

            ax.plot(
                time_steps,
                gt_seq[:, r],
                color="#9A9A9A",
                linewidth=1.1,
                alpha=0.9,
                label="Ground truth",
                zorder=1,
            )

            y_pred = pred_seq[:, r].copy()
            y_pred[:this_pred_start] = np.nan
            ax.plot(
                time_steps,
                y_pred,
                color="#2E86AB",
                linewidth=1.4,
                alpha=0.95,
                label="1_step prediction",
                zorder=3,
            )

            ax.set_ylabel(f"Region {r}", fontsize=11)
            ax.tick_params(axis="both", labelsize=10)
            ax.legend(
                loc="upper right",
                frameon=True,
                framealpha=0.95,
                edgecolor="#CCCCCC",
                fontsize=8,
                handlelength=2.5,
            )
            ax.margins(x=0.01)
            plt.tight_layout()

            output_path = output_dir / (
                f"{window_type}_example_single_region_{mode}_seq{seq_idx+1}_r{r:03d}_seed{seed}.svg"
            )
            fig.savefig(output_path, format="svg")
            plt.close(fig)

    print(
        f"  ✓ Saved {n_sequences * n_regions} single-region plots for {mode} "
        f"({window_type}, regions={n_regions})"
    )



def discover_checkpoint_pairs():
    """
    Discover checkpoint pairs (AR and TL with same T_in/T_out).
    
    Returns:
        List of dicts with 'AR' and 'TL' checkpoint paths and config info
    """

    checkpoint_pairs = {}
    
    for ckpt_dir in sorted(Path(".").glob("checkpoints_*")):
        if not ckpt_dir.is_dir():
            continue
        
        ckpt_file = ckpt_dir / "checkpoint_fixed_epoch_25.pt" #"final_model.pt"
        if not ckpt_file.exists():
            continue
        
        # Parse checkpoint folder name
        # Expected format: checkpoints_AR_T_in_T_out or checkpoints_TL_T_in_T_out
        parts = ckpt_dir.name.split('_')
        if len(parts) < 4:
            continue
        
        mode = parts[1]
        if mode not in MODES:
            continue
                
        # Extract T_in and T_out
        try:
            T_in = int(parts[2])
            T_out = int(parts[3])
        except (ValueError, IndexError):
            continue
        
        config_key = f"{T_in}_{T_out}"
        
        if config_key not in checkpoint_pairs:
            checkpoint_pairs[config_key] = {}
        
        checkpoint_pairs[config_key][mode] = {
            'path': ckpt_file,
            'folder_name': ckpt_dir.name,
            'T_in': T_in,
            'T_out': T_out
        }
    
    # Convert to list and filter for pairs that have both AR and TL
    pairs = []

    for config_key, pair in sorted(checkpoint_pairs.items()):
        if all(m in pair for m in MODES):
            pairs.append({
                "config_name": config_key,
                "modes": {m: pair[m] for m in MODES}
            })
        else:
            missing = [m for m in MODES if m not in pair]
            print(f"⚠ Warning: Configuration {config_key} missing {missing} checkpoint(s)")
    
    return pairs



def main():
    print("="*80)
    print("COMPREHENSIVE MODEL INFERENCE EVALUATION")
    print("="*80)


    """SELECTED_CHECKPOINTS = [
        {
            "config_name": "90_1170",
            "1_step": "checkpoints_AR_90_1_dyna",
            "OS": "checkpoints_OS_90_90_dyna",
            "TF": "checkpoints_TF_90_90_dyna",
            "AR": "checkpoints_AR_90_90_dyna", 
        }
    ]"""
    SELECTED_CHECKPOINTS = [
        {
            "config_name": "60_1",
            #"AR_KV": "checkpoints_AR_KV",
            "TF": "checkpoints_TF",
            #"TF_QTL_0.1": "checkpoints_TF_QTL_0.1",
            #"TF_QTL_0.08_KL_0.02": "checkpoints_TF_QTL_0.08_KL_0.02",
            #"TF_QTL_0.08_KL_0.02_MOM_0.03": "checkpoints_TF_QTL_0.08_KL_0.02_MOM_0.03",
            #"TF_QTL_0.08_KL_0.02_TRJ_0.03": "checkpoints_TF_QTL_0.08_KL_0.02_TRJ_0.03",
            #"MIX_TF_AR_KV": "checkpoints_mix_tf_ar",
            #"1_step": "checkpoints_1_step",
        }
    ]



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Number of sequences per evaluation: {NUM_SEQUENCES}")
    print(f"Seeds: {SEEDS}")
    print(f"Short window prediction length: {SHORT_PRED_LENGTH}")
    print(f"Long window prediction length: {LONG_PRED_LENGTH}")

    # ------------------------------------------------------------
    # Build checkpoint_pairs (unified structure: {"config_name", "modes": {...}})
    # ------------------------------------------------------------
    print("\n" + "="*80)
    print("LOADING SELECTED CHECKPOINTS")
    print("="*80)

    checkpoint_pairs = []
    for entry in SELECTED_CHECKPOINTS:
        cfg_name = entry.get("config_name", None)

        modes_info = {}
        for mode in MODES:
            folder = entry.get(mode, None)
            if folder is None:
                print(f"⚠ Missing {mode} in {entry}")
                continue

            ckpt_path = Path(folder) / "best_model.pt"
            if not ckpt_path.exists():
                print(f"⚠ Missing checkpoint file: {ckpt_path}")
                continue

            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            T_in = ckpt["config"]["T_in"]
            T_out = ckpt["config"]["T_out"]

            modes_info[mode] = {
                "path": ckpt_path,
                "folder_name": folder,
                "T_in": T_in,
                "T_out": T_out,
            }

        # require all modes
        missing = [m for m in MODES if m not in modes_info]
        if missing:
            print(f"⚠ Skipping entry {cfg_name} (missing {missing})")
            continue

        tins = {modes_info[m]["T_in"] for m in MODES}
        if len(tins) != 1:
            print(f"⚠ Skipping entry {cfg_name} (T_in mismatch across modes)")
            for m in MODES:
                print(f"  {m}: T_in={modes_info[m]['T_in']}, T_out={modes_info[m]['T_out']}")
            continue

        # keep cfg_name stable even if T_out differs
        if cfg_name is None:
            cfg_name = f"{list(tins)[0]}_mixedTout"


        checkpoint_pairs.append({"config_name": cfg_name, "modes": modes_info})

    if not checkpoint_pairs:
        print("❌ No valid checkpoint groups loaded. Check paths / best_model.pt existence.")
        return

    print(f"\nLoaded {len(checkpoint_pairs)} checkpoint group(s):")
    for p in checkpoint_pairs:
        print(f"  - {p['config_name']}: modes={list(p['modes'].keys())}")

    # ------------------------------------------------------------
    # Load data config from first available mode (prefer AR, else first mode)
    # ------------------------------------------------------------
    first_pair = checkpoint_pairs[0]
    pick_mode = "AR" if "AR" in first_pair["modes"] else list(first_pair["modes"].keys())[0]
    first_checkpoint_path = first_pair["modes"][pick_mode]["path"]

    print("\n" + "="*80)
    print(f"LOADING DATA (from {pick_mode}: {first_checkpoint_path})")
    print("="*80)

    """checkpoint = torch.load(first_checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    data_array = load_data(config["data_path"])
    examples, sequence_indices = reshape_to_examples(data_array)
    T_in_common = first_pair["modes"][pick_mode]["T_in"]
    T_out_filter = max(first_pair["modes"][m]["T_out"] for m in MODES)
    examples, sequence_indices = filter_examples_by_nan(
        examples, sequence_indices,
        T_in=T_in_common,
        T_out=T_out_filter
    )
    train_examples, val_examples, _, val_seq_indices = split_by_sequences(
        examples,
        sequence_indices,
        train_ratio=config["train_ratio"],
        random_seed=config["random_seed"],
    )

    # --- Normalize after split 
    mean, std = load_norm_stats("data/train_norm_stats.npy")
    train_examples = apply_normalization_nct(train_examples, mean, std)
    val_examples   = apply_normalization_nct(val_examples,   mean, std)"""

    ckpt = torch.load(first_checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    print(
        "[checkpoint norm/spike]",
        f"norm_eps={cfg.get('norm_eps')}, norm_std_floor_abs={cfg.get('norm_std_floor_abs')}, "
        f"norm_std_floor_frac_median={cfg.get('norm_std_floor_frac_median')}; "
        f"loss_spike_weight_beta={cfg.get('loss_spike_weight_beta')}, "
        f"loss_spike_z_threshold={cfg.get('loss_spike_z_threshold')}",
    )

    repo_root = Path(__file__).resolve().parents[2]
    val_p = _resolve_existing_file(cfg.get("processed_val_examples_path"), repo_root)
    seq_p = _resolve_existing_file(cfg.get("processed_val_seq_indices_path"), repo_root)
    if val_p is None or seq_p is None:
        fallback_val_p, fallback_seq_p = _build_processed_paths_from_cfg(cfg, repo_root)
        val_p = val_p or fallback_val_p
        seq_p = seq_p or fallback_seq_p

    if val_p is not None and seq_p is not None and val_p.is_file() and seq_p.is_file():
        val_examples = np.load(val_p)
        val_seq_indices = np.load(seq_p).astype(np.int64, copy=False)
    else:
        print(f"❌ Missing processed val array or sequence indices: {val_p}, {seq_p}")
        print("   Expected checkpoint config fields or data_processed/processed_val_<run_stem>.npy fallback.")
        return

    metadata_path = Path("data/data_organized_metadata.json")
    region_names = None
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        region_names = metadata.get("region_names", None)
        if region_names:
            print(f"  Loaded {len(region_names)} region names")

    # ------------------------------------------------------------
    # Evaluation loops - All results saved to 90_90 folder
    # ------------------------------------------------------------
    base_output_dir = Path("evaluation_results")
    main_output_dir = base_output_dir / checkpoint_pairs[0]["config_name"]

    all_results = {}
    all_results_by_seed = {}

    for pair_idx, pair in enumerate(checkpoint_pairs):
        config_name = pair["config_name"]
        print("\n" + "="*80)
        print(f"PROCESSING CONFIG {pair_idx+1}/{len(checkpoint_pairs)}: {config_name}")
        print("="*80)

        all_results[config_name] = {"short": {}, "long": {}}
        all_results_by_seed[config_name] = {
            "short": {m: [] for m in MODES},
            "long":  {m: [] for m in MODES},
        }

        for seed in SEEDS:
            print(f"\n{'='*60}")
            print(f"SEED {seed}")
            print(f"{'='*60}")

            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            n_val_examples = len(val_examples)
            # Use main_output_dir for all results
            main_output_dir.mkdir(parents=True, exist_ok=True)

            indices_path = main_output_dir / f"selected_indices_N{NUM_SEQUENCES}_seed{seed}.npy"
            if indices_path.exists():
                selected_indices = np.load(indices_path)
                print(f"  Loaded indices: {indices_path}")
            else:
                selected_indices = np.random.choice(
                    n_val_examples,
                    size=min(NUM_SEQUENCES, n_val_examples),
                    replace=False
                )
                np.save(indices_path, selected_indices)
                print(f"  Saved indices: {indices_path}")

            for mode in MODES:
                print(f"\n{'='*60}")
                print(f"{config_name} {mode} MODEL")
                print(f"{'='*60}")

                ckpt_info = pair["modes"][mode]
                T_in = ckpt_info["T_in"]
                T_out = ckpt_info["T_out"]

                # Save to main_output_dir with config prefix in filenames
                output_dir = main_output_dir / f"seed_{seed}"
                output_dir.mkdir(parents=True, exist_ok=True)

                print(f"\n  Loading {config_name} {mode} model...")
                model, model_config = create_inference_model(
                    ckpt_info["path"], T_in, T_out, device
                )

                long_scores = evaluate_long_window(
                    model, ckpt_info["folder_name"], mode, T_in, T_out,
                    val_examples, val_seq_indices, selected_indices,
                    output_dir, region_names, device, seed, LONG_PRED_LENGTH, config_name
                )
                all_results_by_seed[config_name]["long"][mode].append(long_scores)

                if seed == SEEDS[0]:
                    all_results[config_name]["long"][mode] = long_scores

                if seed == SEEDS[0]:
                    long_pred = np.load(output_dir / f"long_predictions_{config_name}_{mode}.npy")
                    long_gt = np.load(output_dir / f"long_ground_truth_{config_name}.npy")
                    plot_prediction_examples(
                        long_pred, long_gt, output_dir, f"{config_name}_{mode}", "long",
                        n_examples=20, pred_start=T_in, seed=seed
                    )
                    if PLOT_SINGLE_REGION_EXAMPLES:
                        plot_prediction_examples_single_region(
                            long_pred,
                            long_gt,
                            output_dir,
                            f"{config_name}_{mode}",
                            "long",
                            n_examples=20,
                            pred_start=T_in,
                            seed=seed,
                            max_regions=MAX_SINGLE_REGION_PLOTS,
                        )

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Results saved to: {base_output_dir}")



if __name__ == "__main__":
    try:
        main()
        print("\n[SUCCESS] Evaluation completed successfully!")
    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        raise

