"""
Long-horizon autoregressive inference evaluation.

Loads validation tensors saved during training, runs KV-cache rollouts to a fixed
prediction length per sequence, writes stacked ``.npy`` archives and NeuroBench-style
CSVs for the forecast window, and saves example overlay SVGs.

Configure ``MODES``, ``SELECTED_CHECKPOINTS``, and ``EVALUATION_SEED`` below. Run from the
repository root so checkpoint and ``data_processed`` paths resolve.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

_SRC_ROOT = Path(__file__).resolve().parent.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from models.model_KV_cached import create_model_cached

NUM_SEQUENCES = 222
LONG_PRED_LENGTH = 720
EVALUATION_SEED = 102

MODES = ["TF_QTL_0.08_KL_0.02"]


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
    Rebuild default processed val paths using train_all_regimes run_stem convention.
    New checkpoints use split_seed; random_seed remains supported for old ones.
    """
    try:
        data_stem = Path(cfg["data_path"]).stem
        split_seed_value = cfg["split_seed"] if "split_seed" in cfg else cfg["random_seed"]
        split_seed = int(split_seed_value)
        run_stem = (
            f"{data_stem}_Tin{int(cfg['T_in'])}_Tout{int(cfg['T_out'])}_splitseed{split_seed}"
        )
    except Exception:
        return None, None

    proc_root = repo_root / "data_processed"
    val_p = proc_root / f"processed_val_{run_stem}.npy"
    seq_p = proc_root / f"processed_val_seq_indices_{run_stem}.npy"
    if not val_p.is_file() or not seq_p.is_file():
        # Backward-compatible path for checkpoints made before split_seed existed.
        try:
            old_stem = (
                f"{data_stem}_Tin{int(cfg['T_in'])}_Tout{int(cfg['T_out'])}"
                f"_seed{int(cfg['random_seed'])}"
            )
            val_p = proc_root / f"processed_val_{old_stem}.npy"
            seq_p = proc_root / f"processed_val_seq_indices_{old_stem}.npy"
        except (KeyError, TypeError, ValueError):
            pass
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
    )
    
    # Load weights with position embedding resizing
    load_checkpoint_with_resized_pos(model, checkpoint, device=device)
    
    model.eval()
    return model, original_config


def generate_long_sequence(model, initial_input, target_length, T_in, T_out, device="cpu"):
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
            #pred = model.forward_autoregressive_old(input_chunk)
            current_sequence = torch.cat([current_sequence, pred], dim=1)
            #block_offset += T_out
        
    return current_sequence[:, :target_length, :]




def prepare_input_and_gt(val_examples, val_seq_indices, example_idx, T_in, target_length, MAX_CONTEXT):
    """
    Prepare input sequence and ground truth for a given example.
    
    Returns:
        input_tensor: Shape (1, T_in, n_vars)
        gt_sequence: Shape (target_length, n_vars)
    """
    first_example = val_examples[example_idx].T  # (n_time, n_vars)
    seq_id = int(val_seq_indices[example_idx])
    
    # Build base context
    if first_example.shape[0] >= MAX_CONTEXT:
        base_input = first_example[:MAX_CONTEXT, :]
    else:
        pad_len = MAX_CONTEXT - first_example.shape[0]
        base_input = np.concatenate(
            [first_example, np.repeat(first_example[-1:, :], pad_len, axis=0)],
            axis=0
        )
    
    # Model input is last T_in points
    input_sequence = base_input[-T_in:, :]
    
    # Build ground truth
    gt_parts = [input_sequence]
    first_example_future = first_example[MAX_CONTEXT:, :]
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
    model,
    checkpoint_name: str,
    mode: str,
    T_in: int,
    T_out: int,
    val_examples,
    val_seq_indices,
    selected_indices,
    output_dir: Path,
    device: torch.device,
    seed: int,
    target_pred_length: int = 800,
    config_name: str | None = None,
) -> None:
    """
    Autoregressive rollout to ``T_in + target_pred_length`` timesteps per sequence.

    Saves full-length predictions/GT as ``.npy``, then writes scored CSVs for the
    forecast segment only (indices ``T_in:``).
    """
    print("\n" + "=" * 80)
    print(f"LONG WINDOW EVALUATION: {checkpoint_name} (SEED={seed})")
    print(f"  T_out={T_out}, generating {target_pred_length} timesteps")
    print("=" * 80)

    MAX_CONTEXT = T_in + 50
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

        long_pred = generate_long_sequence(model, input_tensor, target_length, T_in, T_out, device)

        all_predictions.append(long_pred.detach().cpu().numpy()[0])
        all_ground_truth.append(gt_sequence)

    all_predictions = np.asarray(all_predictions)
    all_ground_truth = np.asarray(all_ground_truth)

    print(f"\n  Predictions shape: {all_predictions.shape}")
    print(f"  Ground truth shape: {all_ground_truth.shape}")

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
                color="#7A7A7A",
                linewidth=1.2,
                alpha=0.45,
                #label=f"{mode} prediction" if r == 0 else None,
                label="Prediction" if r == 0 else None,
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


def main():
    print("="*80)
    print("COMPREHENSIVE MODEL INFERENCE EVALUATION")
    print("="*80)


    SELECTED_CHECKPOINTS = [
        {
            "config_name": "90_810",
            "TF_QTL_0.08_KL_0.02": "checkpoints_TF_QTL_0.08_KL_0.02_seed103",
        }
    ]



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Number of sequences per evaluation: {NUM_SEQUENCES}")
    print(f"Evaluation seed: {EVALUATION_SEED}")
    print(f"Long rollout length (prediction timesteps): {LONG_PRED_LENGTH}")

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

            ckpt_path = Path(folder) / "final_model.pt"
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
        print("❌ No valid checkpoint groups loaded. Check paths / final_model.pt existence.")
        return

    print(f"\nLoaded {len(checkpoint_pairs)} checkpoint group(s):")
    for p in checkpoint_pairs:
        print(f"  - {p['config_name']}: modes={list(p['modes'].keys())}")

    # ------------------------------------------------------------
    # Load validation tensors (paths from checkpoint config or data_processed fallback)
    # ------------------------------------------------------------
    first_pair = checkpoint_pairs[0]
    pick_mode = next(iter(first_pair["modes"]))
    first_checkpoint_path = first_pair["modes"][pick_mode]["path"]

    print("\n" + "="*80)
    print(f"LOADING DATA (from {pick_mode}: {first_checkpoint_path})")
    print("="*80)

    ckpt = torch.load(first_checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    split_seed = int(cfg["split_seed"] if "split_seed" in cfg else cfg["random_seed"])

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

    # ------------------------------------------------------------
    # Long-window evaluation → evaluation_results/<config layout>/seed_*
    # ------------------------------------------------------------
    base_output_dir = Path("evaluation_results")
    main_output_dir = base_output_dir / "90_810"

    for pair_idx, pair in enumerate(checkpoint_pairs):
        config_name = pair["config_name"]
        print("\n" + "="*80)
        print(f"PROCESSING CONFIG {pair_idx+1}/{len(checkpoint_pairs)}: {config_name}")
        print("="*80)

        for seed in [EVALUATION_SEED]:
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

            indices_path = main_output_dir / (
                f"selected_indices_N{NUM_SEQUENCES}"
                f"_splitseed{split_seed}_evalseed{seed}.npy"
            )
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
                model, _ = create_inference_model(
                    ckpt_info["path"], T_in, T_out, device
                )

                evaluate_long_window(
                    model,
                    ckpt_info["folder_name"],
                    mode,
                    T_in,
                    T_out,
                    val_examples,
                    val_seq_indices,
                    selected_indices,
                    output_dir,
                    device,
                    seed,
                    LONG_PRED_LENGTH,
                    config_name,
                )

                if seed == EVALUATION_SEED:
                    long_pred = np.load(output_dir / f"long_predictions_{config_name}_{mode}.npy")
                    long_gt = np.load(output_dir / f"long_ground_truth_{config_name}.npy")
                    plot_prediction_examples(
                        long_pred, long_gt, output_dir, f"{config_name}_{mode}", "long",
                        n_examples=20, pred_start=T_in, seed=seed
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

