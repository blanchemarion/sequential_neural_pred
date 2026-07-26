"""
Inference/evaluation script for the linear VAR baseline model.

Inspired by ``inference_all_regimes.py``:
- loads baseline checkpoint from ``train_var_baseline.py``
- performs long autoregressive rollouts
- saves full arrays (.npy), scored CSVs, and overlay SVG example plots
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

_SRC_ROOT = Path(__file__).resolve().parent.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from models.model_var_baseline import create_var_baseline


def save_sequences_to_neurobench_csv(arr: np.ndarray, csv_path: Path, region_names=None) -> Path:
    """Save array (n_seq, T, n_vars) to NeuroBench CSV format."""
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected (n_seq, T, n_vars), got {arr.shape}")
    n_seq, t_steps, n_vars = arr.shape

    if region_names is None:
        region_names = [f"roi_{i}" for i in range(n_vars)]
    if len(region_names) != n_vars:
        raise ValueError("region_names length must equal n_vars")

    df = pd.DataFrame(arr.reshape(n_seq * t_steps, n_vars), columns=region_names)
    df.insert(0, "itemPosition", np.tile(np.arange(t_steps), n_seq))
    df.insert(0, "sequenceId", np.repeat(np.arange(n_seq), t_steps))

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return csv_path


def create_inference_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]

    model = create_var_baseline(
        n_vars=int(cfg["n_vars"]),
        T_in=int(cfg["T_in"]),
        T_out=int(cfg["T_out"]),
        bias=True,
        device=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, cfg


def generate_long_sequence(
    model,
    initial_input: torch.Tensor,
    target_length: int,
    t_in: int,
    t_out: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate long sequence by repeatedly calling model autoregressive rollout."""
    model.eval()
    current_sequence = initial_input.to(device)
    with torch.no_grad():
        while current_sequence.shape[1] < target_length:
            if current_sequence.shape[1] >= t_in:
                input_chunk = current_sequence[:, -t_in:, :]
            else:
                pad_len = t_in - current_sequence.shape[1]
                pad = current_sequence[:, :1, :].repeat(1, pad_len, 1)
                input_chunk = torch.cat([pad, current_sequence], dim=1)

            pred = model.forward_autoregressive(input_chunk)
            current_sequence = torch.cat([current_sequence, pred], dim=1)

    return current_sequence[:, :target_length, :]


def prepare_input_and_gt(
    val_examples: np.ndarray,
    val_seq_indices: np.ndarray,
    example_idx: int,
    t_in: int,
    target_length: int,
    max_context: int,
):
    """Build model input + long ground-truth continuation from same logical sequence."""
    first_example = val_examples[example_idx].T  # (time, vars)
    seq_id = int(val_seq_indices[example_idx])

    if first_example.shape[0] >= max_context:
        base_input = first_example[:max_context, :]
    else:
        pad_len = max_context - first_example.shape[0]
        base_input = np.concatenate(
            [first_example, np.repeat(first_example[-1:, :], pad_len, axis=0)],
            axis=0,
        )

    input_sequence = base_input[-t_in:, :]

    gt_parts = [input_sequence]
    first_example_future = first_example[max_context:, :]
    if first_example_future.size > 0:
        gt_parts.append(first_example_future)

    same_sequence_indices = np.where(val_seq_indices == seq_id)[0]
    sorted_seq_indices = sorted(same_sequence_indices.tolist())
    start_pos = sorted_seq_indices.index(int(example_idx))

    for seq_example_idx in sorted_seq_indices[start_pos + 1 :]:
        gt_parts.append(val_examples[seq_example_idx].T)

    gt_sequence = np.concatenate(gt_parts, axis=0)

    if gt_sequence.shape[0] >= target_length:
        gt_sequence = gt_sequence[:target_length, :]
    else:
        remaining = target_length - gt_sequence.shape[0]
        gt_sequence = np.concatenate(
            [gt_sequence, np.repeat(gt_sequence[-1:, :], remaining, axis=0)],
            axis=0,
        )

    input_tensor = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0)
    return input_tensor, gt_sequence


def plot_prediction_examples(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    output_dir: Path,
    mode: str,
    config_name: str,
    pred_start: int,
    seed: int,
    n_examples: int,
) -> None:
    """
    Overlay all-region traces for a subset of sequences and save SVG figures.
    """
    mpl.rcParams["svg.fonttype"] = "none"
    output_dir.mkdir(parents=True, exist_ok=True)
    n_sequences = min(n_examples, predictions.shape[0])

    for seq_idx in range(n_sequences):
        fig, ax = plt.subplots(figsize=(7, 2.2))
        pred_seq = predictions[seq_idx]
        gt_seq = ground_truth[seq_idx]
        t_steps, n_vars = pred_seq.shape
        x = np.arange(t_steps)

        ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
        ax.axvline(
            x=pred_start,
            color="#8C8C8C",
            linestyle=(0, (4, 2)),
            linewidth=1.0,
            alpha=0.9,
            label="Prediction start",
            zorder=2,
        )

        for r in range(n_vars):
            ax.plot(
                x,
                gt_seq[:, r],
                color="#9A9A9A",
                linewidth=1.0,
                alpha=0.35,
                label="Ground truth" if r == 0 else None,
                zorder=1,
            )

            y_pred = pred_seq[:, r].copy()
            y_pred[:pred_start] = np.nan
            ax.plot(
                x,
                y_pred,
                color="#7A7A7A",
                linewidth=1.2,
                alpha=0.45,
                label=f"{mode} prediction" if r == 0 else None,
                zorder=3,
            )

        ax.set_ylabel("All regions", fontsize=11)
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
        out = output_dir / f"long_example_{config_name}_{mode}_seq{seq_idx + 1}_seed{seed}.svg"
        fig.savefig(out, format="svg")
        plt.close(fig)

    print(f"Saved {n_sequences} long-window example plot(s).")


def evaluate_long_window(
    model,
    val_examples: np.ndarray,
    val_seq_indices: np.ndarray,
    selected_indices: np.ndarray,
    output_dir: Path,
    t_in: int,
    t_out: int,
    target_pred_length: int,
    device: torch.device,
    seed: int,
    config_name: str,
    mode: str,
    n_plot_examples: int,
) -> None:
    max_context = t_in + 50
    target_length = t_in + target_pred_length

    all_predictions = []
    all_ground_truth = []

    for idx, example_idx in enumerate(selected_indices):
        print(f"  Sequence {idx + 1}/{len(selected_indices)}")
        input_tensor, gt_sequence = prepare_input_and_gt(
            val_examples, val_seq_indices, int(example_idx), t_in, target_length, max_context
        )
        input_tensor = input_tensor.to(device)
        long_pred = generate_long_sequence(model, input_tensor, target_length, t_in, t_out, device)
        all_predictions.append(long_pred.detach().cpu().numpy()[0])
        all_ground_truth.append(gt_sequence)

    all_predictions = np.asarray(all_predictions, dtype=np.float32)
    all_ground_truth = np.asarray(all_ground_truth, dtype=np.float32)

    min_n = min(all_predictions.shape[0], all_ground_truth.shape[0])
    min_t = min(all_predictions.shape[1], all_ground_truth.shape[1])
    min_v = min(all_predictions.shape[2], all_ground_truth.shape[2])
    all_predictions = all_predictions[:min_n, :min_t, :min_v]
    all_ground_truth = all_ground_truth[:min_n, :min_t, :min_v]

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"long_predictions_{config_name}_{mode}.npy", all_predictions)
    np.save(output_dir / f"long_ground_truth_{config_name}.npy", all_ground_truth)

    pred_scored = all_predictions[:, t_in:, :]
    gt_scored = all_ground_truth[:, t_in:, :]
    t_pred = min(pred_scored.shape[1], gt_scored.shape[1])
    v = min(pred_scored.shape[2], gt_scored.shape[2])
    pred_scored = pred_scored[:, :t_pred, :v]
    gt_scored = gt_scored[:, :t_pred, :v]

    region_names = [f"roi_{i}" for i in range(v)]
    pred_csv = output_dir / f"long_predictions_scored_{mode}.csv"
    gt_csv = output_dir / "long_ground_truth_scored.csv"
    save_sequences_to_neurobench_csv(pred_scored, pred_csv, region_names=region_names)
    save_sequences_to_neurobench_csv(gt_scored, gt_csv, region_names=region_names)

    plot_prediction_examples(
        all_predictions,
        all_ground_truth,
        output_dir,
        mode=mode,
        config_name=config_name,
        pred_start=t_in,
        seed=seed,
        n_examples=n_plot_examples,
    )


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints_VAR_BASELINE_90_90_seed103") / "final_model.pt",
        help="Path to baseline checkpoint from train_var_baseline.py",
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=222,
        help="Number of validation rows to evaluate",
    )
    parser.add_argument(
        "--evaluation-seed",
        "--seed",
        dest="evaluation_seed",
        type=int,
        default=102,
        help="Fixed seed for sampled evaluation indices",
    )
    parser.add_argument(
        "--long-pred-length",
        type=int,
        default=720,
        help="Forecast horizon length (after T_in context)",
    )
    parser.add_argument(
        "--n-plot-examples",
        type=int,
        default=20,
        help="How many long-window examples to plot",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("evaluation_results"),
        help="Root output directory",
    )
    return parser.parse_args(argv if argv is not None else sys.argv[1:])


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = "VAR_BASELINE"

    print("=" * 80)
    print("VAR BASELINE INFERENCE")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    model, cfg = create_inference_model(args.checkpoint, device=device)
    t_in = int(cfg["T_in"])
    t_out = int(cfg["T_out"])
    config_name = f"{t_in}_{args.long_pred_length}"

    val_path = cfg.get("processed_val_examples_path")
    val_seq_path = cfg.get("processed_val_seq_indices_path")
    if not val_path or not val_seq_path:
        raise ValueError(
            "Checkpoint config is missing processed val paths. "
            "Retrain with current train_var_baseline.py to populate them."
        )

    val_examples = np.load(Path(val_path))
    val_seq_indices = np.load(Path(val_seq_path)).astype(np.int64, copy=False)
    if val_examples.ndim != 3:
        raise ValueError(f"Expected val examples shape (N,C,T), got {val_examples.shape}")
    if val_seq_indices.ndim != 1 or val_seq_indices.shape[0] != val_examples.shape[0]:
        raise ValueError("val_seq_indices is incompatible with val_examples")

    evaluation_seed = int(args.evaluation_seed)
    split_seed = int(cfg.get("split_seed", cfg.get("random_seed", 101)))
    print(f"Split seed: {split_seed}")
    print(f"Evaluation seed: {evaluation_seed}")
    np.random.seed(evaluation_seed)
    torch.manual_seed(evaluation_seed)
    random.seed(evaluation_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(evaluation_seed)

    n_val = len(val_examples)
    n_pick = min(args.num_sequences, n_val)
    output_dir = args.output_root / config_name / f"seed_{evaluation_seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    idx_path = output_dir.parent / (
        f"selected_indices_N{n_pick}_splitseed{split_seed}"
        f"_evalseed{evaluation_seed}.npy"
    )

    if idx_path.exists():
        selected_indices = np.load(idx_path).astype(np.int64, copy=False)
        print(f"Loaded selected indices: {idx_path}")
    else:
        selected_indices = np.random.choice(n_val, size=n_pick, replace=False).astype(np.int64, copy=False)
        np.save(idx_path, selected_indices)
        print(f"Saved selected indices: {idx_path}")

    evaluate_long_window(
        model=model,
        val_examples=val_examples,
        val_seq_indices=val_seq_indices,
        selected_indices=selected_indices,
        output_dir=output_dir,
        t_in=t_in,
        t_out=t_out,
        target_pred_length=args.long_pred_length,
        device=device,
        seed=evaluation_seed,
        config_name=config_name,
        mode=mode,
        n_plot_examples=args.n_plot_examples,
    )

    print("\nDone.")
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[ERROR] Inference failed: {exc}")
        raise
