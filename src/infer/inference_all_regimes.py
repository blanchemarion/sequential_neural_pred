"""
Long-horizon autoregressive inference evaluation.

Loads validation tensors saved during training, runs KV-cache rollouts to a fixed
prediction length per sequence, writes stacked ``.npy`` archives and NeuroBench-style
CSVs for the forecast window, and saves example overlay SVGs.

Discovers every available training seed for the four Transformer regimes and
writes each seed to its corresponding 90_810 evaluation directory. Run from the
repository root so checkpoint and validation-data paths resolve.
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

MODE_CHECKPOINT_PREFIXES = {
    "AR_KV": "AR_KV",
    "TF": "TF",
    "TF_QTL_0.08_KL_0.02": "TF_QTL_0.08_KL_0.02",
    "1_step": "1_step",
}
MODES = tuple(MODE_CHECKPOINT_PREFIXES)


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
    """Prepare a model context and the saved evaluation continuation."""
    first_example = val_examples[example_idx].T
    seq_id = int(val_seq_indices[example_idx])
    if first_example.shape[0] >= MAX_CONTEXT:
        base_input = first_example[:MAX_CONTEXT]
    else:
        pad_len = MAX_CONTEXT - first_example.shape[0]
        base_input = np.concatenate(
            [first_example, np.repeat(first_example[-1:], pad_len, axis=0)], axis=0
        )
    input_sequence = base_input[-T_in:]
    gt_parts = [input_sequence]
    first_example_future = first_example[MAX_CONTEXT:]
    if first_example_future.size:
        gt_parts.append(first_example_future)
    same_sequence_indices = np.flatnonzero(val_seq_indices == seq_id)
    start_pos = int(np.flatnonzero(same_sequence_indices == example_idx)[0])
    for seq_example_idx in same_sequence_indices[start_pos + 1:]:
        gt_parts.append(val_examples[seq_example_idx].T)
    gt_sequence = np.concatenate(gt_parts, axis=0)
    if gt_sequence.shape[0] >= target_length:
        gt_sequence = gt_sequence[:target_length]
    else:
        remaining = target_length - gt_sequence.shape[0]
        gt_sequence = np.concatenate(
            [gt_sequence, np.repeat(gt_sequence[-1:], remaining, axis=0)], axis=0
        )
    return torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0), gt_sequence


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

    print(f"  Saved {n_sequences} example plots for {mode} ({window_type})")


def discover_checkpoint_groups(repo_root: Path) -> dict[int, dict[str, Path]]:
    """Find complete sets of best checkpoints, keyed by training seed."""
    groups: dict[int, dict[str, Path]] = {}
    for mode, prefix in MODE_CHECKPOINT_PREFIXES.items():
        stem = f"checkpoints_{prefix}_seed"
        for folder in repo_root.glob(f"{stem}*"):
            if not folder.is_dir():
                continue
            suffix = folder.name[len(stem):]
            if not suffix.isdecimal():
                continue
            checkpoint_path = folder / "best_model.pt"
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Missing best checkpoint: {checkpoint_path}")
            groups.setdefault(int(suffix), {})[mode] = checkpoint_path

    if not groups:
        raise FileNotFoundError("No Transformer best_model.pt checkpoints found")
    for training_seed, paths in sorted(groups.items()):
        missing = set(MODES) - paths.keys()
        if missing:
            raise FileNotFoundError(
                f"Training seed {training_seed} is missing checkpoints for {sorted(missing)}"
            )
    return dict(sorted(groups.items()))


def evaluation_directory(root: Path, training_seed: int) -> Path:
    """Reuse an existing seed directory or use the standard spelling."""
    standard = root / f"val_seed_{EVALUATION_SEED}_train_seed_{training_seed}"
    legacy = root / f"val_seed_{EVALUATION_SEED}_train_seed{training_seed}"
    if standard.is_dir() and legacy.is_dir():
        raise RuntimeError(f"Two evaluation directories for seed {training_seed}: {standard}, {legacy}")
    return legacy if legacy.is_dir() else standard


def checkpoint_config(checkpoint_path: Path, training_seed: int, mode: str) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    actual_seed = int(config.get("training_seed", config.get("random_seed", training_seed)))
    if actual_seed != training_seed:
        raise ValueError(f"{checkpoint_path}: training seed {actual_seed} != {training_seed}")
    expected_t_out = 1 if mode == "1_step" else 90
    if int(config["T_in"]) != 90 or int(config["T_out"]) != expected_t_out:
        raise ValueError(f"{checkpoint_path}: unexpected T_in/T_out for 90_810 evaluation")
    return config


def selected_validation_indices(
    output_root: Path, split_seed: int, n_examples: int
) -> np.ndarray:
    if n_examples < NUM_SEQUENCES:
        raise ValueError(f"Need {NUM_SEQUENCES} validation examples; found {n_examples}")
    path = output_root / (
        f"selected_indices_N{NUM_SEQUENCES}"
        f"_splitseed{split_seed}_evalseed{EVALUATION_SEED}.npy"
    )
    if path.is_file():
        indices = np.load(path, allow_pickle=False)
        print(f"Using saved validation indices: {path}")
    else:
        indices = np.random.RandomState(EVALUATION_SEED).choice(
            n_examples, size=NUM_SEQUENCES, replace=False
        )
        output_root.mkdir(parents=True, exist_ok=True)
        np.save(path, indices)
        print(f"Saved validation indices: {path}")
    if (
        indices.ndim != 1
        or len(indices) != NUM_SEQUENCES
        or not np.issubdtype(indices.dtype, np.integer)
        or len(np.unique(indices)) != NUM_SEQUENCES
        or np.any(indices < 0)
        or np.any(indices >= n_examples)
    ):
        raise ValueError(f"Invalid 222-sequence selection in {path}")
    return indices


def main():
    repo_root = Path(__file__).resolve().parents[2]
    output_root = repo_root / "evaluation_results" / "90_810"
    groups = discover_checkpoint_groups(repo_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Training seeds: {list(groups)}")
    print(f"Models: {list(MODES)}")
    print(f"Validation seed: {EVALUATION_SEED}")
    print(f"Sequences per model and seed: {NUM_SEQUENCES}")

    # The one-step checkpoint has a different training target length. All four
    # models nevertheless use the same 90-step validation rows for comparison.
    configs = {}
    for training_seed, paths in groups.items():
        for mode, checkpoint_path in paths.items():
            configs[(training_seed, mode)] = checkpoint_config(
                checkpoint_path, training_seed, mode
            )

    reference = configs[(next(iter(groups)), "AR_KV")]
    split_seed = int(reference.get("split_seed", reference.get("random_seed")))
    n_vars = int(reference["n_vars"])
    data_stem = Path(reference["data_path"]).stem
    for (training_seed, mode), cfg in configs.items():
        cfg_split_seed = int(cfg.get("split_seed", cfg.get("random_seed")))
        if (
            cfg_split_seed != split_seed
            or int(cfg["n_vars"]) != n_vars
            or Path(cfg["data_path"]).stem != data_stem
        ):
            raise ValueError(
                f"Checkpoint for {mode}, training seed {training_seed}, "
                "uses a different validation split, data file, or region count"
            )

    val_p = _resolve_existing_file(reference.get("processed_val_examples_path"), repo_root)
    seq_p = _resolve_existing_file(
        reference.get("processed_val_seq_indices_path"), repo_root
    )
    if val_p is None or seq_p is None:
        fallback_val_p, fallback_seq_p = _build_processed_paths_from_cfg(
            reference, repo_root
        )
        val_p = val_p or fallback_val_p
        seq_p = seq_p or fallback_seq_p
    if val_p is None or seq_p is None or not val_p.is_file() or not seq_p.is_file():
        raise FileNotFoundError(
            f"Missing processed validation arrays: {val_p}, {seq_p}"
        )

    val_examples = np.load(val_p, mmap_mode="r")
    val_seq_indices = np.load(seq_p, allow_pickle=False).astype(np.int64, copy=False)
    if len(val_examples) != len(val_seq_indices):
        raise ValueError("Validation examples and sequence IDs have different lengths")
    selected_indices = selected_validation_indices(
        output_root, split_seed, len(val_examples)
    )

    for training_seed, paths in groups.items():
        output_dir = evaluation_directory(output_root, training_seed)
        print(f"Training seed {training_seed}: {output_dir}")
        np.random.seed(EVALUATION_SEED)
        torch.manual_seed(EVALUATION_SEED)
        random.seed(EVALUATION_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(EVALUATION_SEED)

        for mode in MODES:
            checkpoint_path = paths[mode]
            cfg = configs[(training_seed, mode)]
            t_in, t_out = int(cfg["T_in"]), int(cfg["T_out"])
            print(f"Evaluating {mode} from {checkpoint_path}")
            model, _ = create_inference_model(
                checkpoint_path, t_in, t_out, device
            )
            evaluate_long_window(
                model, checkpoint_path.parent.name, mode, t_in, t_out,
                val_examples, val_seq_indices, selected_indices,
                output_dir, device, EVALUATION_SEED, LONG_PRED_LENGTH, "90_810",
            )
            long_pred = np.load(
                output_dir / f"long_predictions_90_810_{mode}.npy", mmap_mode="r"
            )
            long_gt = np.load(
                output_dir / "long_ground_truth_90_810.npy", mmap_mode="r"
            )
            plot_prediction_examples(
                long_pred, long_gt, output_dir, f"90_810_{mode}", "long",
                n_examples=20, pred_start=t_in, seed=EVALUATION_SEED,
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"Evaluation complete: {output_root}")


if __name__ == "__main__":
    try:
        main()
        print("\n[SUCCESS] Evaluation completed successfully!")
    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        raise

