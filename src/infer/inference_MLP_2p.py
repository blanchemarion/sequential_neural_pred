from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import sys

_SRC_ROOT = Path(__file__).resolve().parent.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from models.mlp_2p import MLP2P


def load_raw_csv_time_major(csv_path: Path) -> np.ndarray:
    arr = pd.read_csv(csv_path, header=None).values.astype(np.float32)  # (N, T)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D CSV (N, T), got shape {arr.shape}")
    return arr.T  # (T, N)


def compute_blocked_regions(total_T: int, T_in: int, gap: int):
    train_end = int(total_T * 0.70)
    val_end = int(total_T * 0.85)
    gap_used = int(gap)

    def starts_count(start: int, end: int) -> int:
        return max(0, end - (start + T_in))

    while gap_used >= 0:
        val_start = train_end + gap_used
        test_start = val_end + gap_used
        if starts_count(val_start, val_end) > 0 and starts_count(test_start, total_T) > 0:
            return {
                "train": [0, train_end],
                "val": [val_start, val_end],
                "test": [test_start, total_T],
                "gap_used": gap_used,
            }
        gap_used -= 1
    raise ValueError("Unable to compute non-empty split regions.")


def sample_test_starts(test_start: int, test_end: int, T_in: int, pred_len: int, num_sequences: int) -> np.ndarray:
    max_start = test_end - (T_in + pred_len)
    if max_start < test_start:
        raise ValueError("Test split too short for requested T_in + pred_len.")
    all_starts = np.arange(test_start, max_start + 1, dtype=np.int64)
    if len(all_starts) <= num_sequences:
        return all_starts
    idx = np.linspace(0, len(all_starts) - 1, num_sequences, dtype=np.int64)
    return all_starts[idx]


@torch.no_grad()
def autoregressive_rollout(
    model: torch.nn.Module,
    initial_context_btn: torch.Tensor,
    T_in: int,
    pred_len: int,
) -> torch.Tensor:
    # initial_context_btn: (B, T_in, N)
    current = initial_context_btn
    generated = [initial_context_btn]
    for _ in range(pred_len):
        pred_next = model(current[:, -T_in:, :]).clamp_min(0.0)  # (B, 1, N)
        generated.append(pred_next)
        current = torch.cat([current, pred_next], dim=1)
    return torch.cat(generated, dim=1)  # (B, T_in + pred_len, N)

@torch.no_grad()
def block_rollout(
    model: torch.nn.Module,
    initial_context_btn: torch.Tensor,
    T_in: int,
    pred_len: int,
) -> torch.Tensor:
    current = initial_context_btn
    generated = [initial_context_btn]

    remaining = pred_len

    while remaining > 0:
        pred_block = model(current[:, -T_in:, :])  # (B, T_out, N)

        if pred_block.shape[1] > remaining:
            pred_block = pred_block[:, :remaining, :]

        generated.append(pred_block)
        current = torch.cat([current, pred_block], dim=1)

        remaining -= pred_block.shape[1]

    return torch.cat(generated, dim=1)


@torch.no_grad()
def block_gt_reset_rollout(
    model: torch.nn.Module,
    initial_context_btn: torch.Tensor,
    ground_truth_full_btn: torch.Tensor,
    T_in: int,
    pred_len: int,
) -> torch.Tensor:
    """
    Multi-step block rollout with periodic GT re-anchoring.

    Each iteration predicts one model block (e.g. T_out=10), appends it,
    then resets context to ground truth up to the current horizon.
    """
    expected_len = T_in + pred_len
    if ground_truth_full_btn.shape[1] < expected_len:
        raise ValueError("ground_truth_full_btn is shorter than T_in + pred_len")

    current = initial_context_btn
    generated = [initial_context_btn]
    produced = 0

    while produced < pred_len:
        pred_block = model(current[:, -T_in:, :]).clamp_min(0.0)  # (B, T_out, N)
        if pred_block.ndim != 3:
            raise ValueError(f"Expected model output (B, T_out, N), got {tuple(pred_block.shape)}")

        # Safety: if a one-step model is used accidentally, still make progress.
        if pred_block.shape[1] == 0:
            raise ValueError("Model returned empty block along time dimension.")

        if pred_block.shape[1] > (pred_len - produced):
            pred_block = pred_block[:, : (pred_len - produced), :]

        generated.append(pred_block)
        produced += pred_block.shape[1]

        if produced < pred_len:
            gt_context_end = T_in + produced
            current = ground_truth_full_btn[:, :gt_context_end, :]

    return torch.cat(generated, dim=1)


@torch.no_grad()
def chunked_gt_reset_rollout(
    model: torch.nn.Module,
    initial_context_btn: torch.Tensor,
    ground_truth_full_btn: torch.Tensor,
    T_in: int,
    pred_len: int,
    chunk_len: int = 10,
) -> torch.Tensor:
    """
    Hybrid rollout:
      1) predict chunk_len steps autoregressively
      2) reset context back to ground truth up to current horizon
      3) repeat until pred_len steps are produced

    Returns full sequence including context: (B, T_in + pred_len, N)
    """
    if chunk_len <= 0:
        raise ValueError("chunk_len must be > 0")
    expected_len = T_in + pred_len
    if ground_truth_full_btn.shape[1] < expected_len:
        raise ValueError("ground_truth_full_btn is shorter than T_in + pred_len")

    current = initial_context_btn
    generated = [initial_context_btn]
    produced = 0

    while produced < pred_len:
        this_chunk = min(chunk_len, pred_len - produced)
        chunk_preds = []
        for _ in range(this_chunk):
            pred_step = model(current[:, -T_in:, :]).clamp_min(0.0)
            # Keep one-step progression even if model emits multi-step blocks.
            if pred_step.shape[1] > 1:
                pred_step = pred_step[:, :1, :]
            chunk_preds.append(pred_step)
            current = torch.cat([current, pred_step], dim=1)

        pred_block = torch.cat(chunk_preds, dim=1)  # (B, this_chunk, N)
        generated.append(pred_block)
        produced += this_chunk

        # Re-anchor context to ground truth so drift does not compound as quickly.
        if produced < pred_len:
            gt_context_end = T_in + produced
            current = ground_truth_full_btn[:, :gt_context_end, :]

    return torch.cat(generated, dim=1)


def save_scored_csv(arr_s_t_n: np.ndarray, csv_path: Path) -> None:
    n_seq, t_pred, n_neurons = arr_s_t_n.shape
    cols = [f"neuron_{i:03d}" for i in range(n_neurons)]
    df = pd.DataFrame(arr_s_t_n.reshape(n_seq * t_pred, n_neurons), columns=cols)
    df.insert(0, "itemPosition", np.tile(np.arange(t_pred), n_seq))
    df.insert(0, "sequenceId", np.repeat(np.arange(n_seq), t_pred))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)


def save_diagnostic_plots(
    pred_s_t_n: np.ndarray,
    gt_s_t_n: np.ndarray,
    T_in: int,
    out_dir: Path,
    max_sequences: int = 8,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_seq, total_t, n_neurons = pred_s_t_n.shape
    k_seq = min(max_sequences, n_seq)
    chosen_neurons = np.linspace(0, n_neurons - 1, min(6, n_neurons), dtype=int)

    for s in range(k_seq):
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        for n in chosen_neurons:
            ax.plot(gt_s_t_n[s, :, n], color="gray", alpha=0.45, linewidth=1.0)
            ax.plot(pred_s_t_n[s, :, n], color="#1f77b4", alpha=0.75, linewidth=1.0)
        ax.axvline(T_in, color="black", linestyle="--", linewidth=1.0)
        ax.set_title(f"Sequence {s} - selected neurons")
        ax.set_xlabel("Time")
        ax.set_ylabel("Activity")
        fig.tight_layout()
        fig.savefig(out_dir / f"sequence_{s:02d}_overlay.png", dpi=140)
        plt.close(fig)

    pred_flat = pred_s_t_n[:, T_in:, :].reshape(-1)
    gt_flat = gt_s_t_n[:, T_in:, :].reshape(-1)
    pmax = float(np.quantile(np.concatenate([pred_flat, gt_flat]), 0.995))
    bins = np.linspace(0.0, max(pmax, 1e-6), 80)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.hist(gt_flat, bins=bins, alpha=0.45, color="gray", label="Ground truth", density=True)
    ax.hist(pred_flat, bins=bins, alpha=0.45, color="#1f77b4", label="Prediction", density=True)
    ax.set_title("Prediction vs ground-truth distribution")
    ax.set_xlabel("Activity")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "distribution_compare.png", dpi=140)
    plt.close(fig)

    pred_mean_t = pred_s_t_n.mean(axis=(0, 2))
    gt_mean_t = gt_s_t_n.mean(axis=(0, 2))
    fig, ax = plt.subplots(1, 1, figsize=(9, 4))
    ax.plot(gt_mean_t, color="gray", label="Ground truth")
    ax.plot(pred_mean_t, color="#1f77b4", label="Prediction")
    ax.axvline(T_in, color="black", linestyle="--", linewidth=1.0)
    ax.set_title("Mean activity over time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Mean activity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "mean_activity_over_time.png", dpi=140)
    plt.close(fig)

    pred_active_t = (pred_s_t_n > 0).mean(axis=(0, 2))
    gt_active_t = (gt_s_t_n > 0).mean(axis=(0, 2))
    fig, ax = plt.subplots(1, 1, figsize=(9, 4))
    ax.plot(gt_active_t, color="gray", label="Ground truth")
    ax.plot(pred_active_t, color="#1f77b4", label="Prediction")
    ax.axvline(T_in, color="black", linestyle="--", linewidth=1.0)
    ax.set_title("Fraction of active neurons (>0)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Active fraction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fraction_active_over_time.png", dpi=140)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference for MLP 2p baseline.")
    p.add_argument("--csv_path", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="evaluation_results/mlp_2p")
    p.add_argument("--T_in", type=int, default=90)
    p.add_argument("--pred_len", type=int, default=720)
    p.add_argument(
        "--rollout_mode",
        type=str,
        default="block_gt_reset",
        choices=["autoregressive", "block", "block_gt_reset", "chunked_gt_reset"],
    )
    p.add_argument("--chunk_len", type=int, default=10, help="Used only with chunked_gt_reset mode.")
    p.add_argument("--num_sequences", type=int, default=10)
    p.add_argument("--split_gap", type=int, default=None, help="Default: T_in + 720")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    data_tn = load_raw_csv_time_major(Path(args.csv_path))
    T_total, n_neurons = data_tn.shape

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_n = int(ckpt["n_neurons"])
    if ckpt_n != n_neurons:
        raise ValueError(f"Neuron count mismatch: checkpoint={ckpt_n}, csv={n_neurons}")

    ckpt_T_in = int(ckpt["T_in"])
    ckpt_T_out = int(ckpt["T_out"])

    if int(args.T_in) != ckpt_T_in:
        raise ValueError(
            f"T_in mismatch: checkpoint={ckpt_T_in}, requested={args.T_in}"
        )

    model = MLP2P(
        n_neurons=ckpt_n,
        T_in=ckpt_T_in,
        T_out=ckpt_T_out,
        d_local=int(ckpt.get("d_local", 64)),
        d_pop=int(ckpt.get("d_pop", 128)),
        dropout=float(ckpt.get("dropout", 0.05)),
        decay_init=float(ckpt.get("decay_init", 0.85)),
    )
    
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    split_info = ckpt.get("split_info", None)
    if split_info is None:
        gap = args.split_gap if args.split_gap is not None else (args.T_in + 720)
        split_info = compute_blocked_regions(T_total, args.T_in, gap)
    test_start, test_end = split_info["test"]

    starts = sample_test_starts(
        test_start=int(test_start),
        test_end=int(test_end),
        T_in=args.T_in,
        pred_len=args.pred_len,
        num_sequences=args.num_sequences,
    )
    print(f"[inference] selected {len(starts)} test starts")
    print(f"[inference] rollout_mode={args.rollout_mode} pred_len={args.pred_len}")
    if args.rollout_mode == "chunked_gt_reset":
        print(f"[inference] chunk_len={args.chunk_len}")

    pred_full_list = []
    gt_full_list = []
    for s in starts:
        s = int(s)
        gt = data_tn[s : s + args.T_in + args.pred_len, :]  # (T_in+pred_len, N)
        ctx = torch.from_numpy(gt[: args.T_in, :]).unsqueeze(0).to(device=device, dtype=torch.float32)
        gt_full = torch.from_numpy(gt).unsqueeze(0).to(device=device, dtype=torch.float32)

        if args.rollout_mode == "autoregressive":
            pred_full = autoregressive_rollout(model, ctx, T_in=args.T_in, pred_len=args.pred_len)
        elif args.rollout_mode == "block":
            pred_full = block_rollout(model, ctx, T_in=args.T_in, pred_len=args.pred_len)
        elif args.rollout_mode == "block_gt_reset":
            pred_full = block_gt_reset_rollout(
                model=model,
                initial_context_btn=ctx,
                ground_truth_full_btn=gt_full,
                T_in=args.T_in,
                pred_len=args.pred_len,
            )
        else:
            pred_full = chunked_gt_reset_rollout(
                model=model,
                initial_context_btn=ctx,
                ground_truth_full_btn=gt_full,
                T_in=args.T_in,
                pred_len=args.pred_len,
                chunk_len=args.chunk_len,
            )
        pred_full_np = pred_full.squeeze(0).detach().cpu().numpy().astype(np.float32)
        pred_full_list.append(pred_full_np)
        gt_full_list.append(gt.astype(np.float32))

    pred_full_s_t_n = np.stack(pred_full_list, axis=0)
    gt_full_s_t_n = np.stack(gt_full_list, axis=0)
    pred_scored = pred_full_s_t_n[:, args.T_in :, :]
    gt_scored = gt_full_s_t_n[:, args.T_in :, :]

    pred_npy = output_dir / "long_predictions_MLP_2p.npy"
    gt_npy = output_dir / "long_ground_truth_MLP_2p.npy"
    np.save(pred_npy, pred_full_s_t_n)
    np.save(gt_npy, gt_full_s_t_n)

    pred_scored_npy = output_dir / "long_predictions_scored_MLP_2p.npy"
    gt_scored_npy = output_dir / "long_ground_truth_scored_MLP_2p.npy"
    np.save(pred_scored_npy, pred_scored)
    np.save(gt_scored_npy, gt_scored)

    pred_csv = output_dir / "long_predictions_scored_MLP_2p.csv"
    gt_csv = output_dir / "long_ground_truth_scored_MLP_2p.csv"
    save_scored_csv(pred_scored, pred_csv)
    save_scored_csv(gt_scored, gt_csv)

    save_diagnostic_plots(
        pred_s_t_n=pred_full_s_t_n,
        gt_s_t_n=gt_full_s_t_n,
        T_in=args.T_in,
        out_dir=plots_dir,
        max_sequences=10,
    )

    print(f"[done] saved {pred_npy}")
    print(f"[done] saved {gt_npy}")
    print(f"[done] saved {pred_csv}")
    print(f"[done] saved {gt_csv}")
    print(f"[done] plots in {plots_dir}")


if __name__ == "__main__":
    main()
