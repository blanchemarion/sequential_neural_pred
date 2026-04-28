from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

_SRC_ROOT = Path(__file__).resolve().parent.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from models.simple_2p_baselines import (
    ExpDecayBaseline,
    RectifiedVARBaseline,
    TinyGRUForecaster,
    rollout_model,
)


def save_sequences_to_neurobench_csv(arr: np.ndarray, csv_path: str | Path, region_names=None):
    arr = np.asarray(arr)
    assert arr.ndim == 3, f"Expected (n_seq, T, n_vars), got {arr.shape}"
    n_seq, t_len, n_vars = arr.shape
    if region_names is None:
        region_names = [f"var_{i}" for i in range(n_vars)]
    df = pd.DataFrame(arr.reshape(n_seq * t_len, n_vars), columns=region_names)
    df.insert(0, "itemPosition", np.tile(np.arange(t_len), n_seq))
    df.insert(0, "sequenceId", np.repeat(np.arange(n_seq), t_len))
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return csv_path


def plot_prediction_examples(predictions, ground_truth, output_dir, mode, n_examples=3, pred_start=None, seed=1):
    mpl.rcParams["svg.fonttype"] = "none"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_sequences = min(n_examples, predictions.shape[0])
    for seq_idx in range(n_sequences):
        fig, ax = plt.subplots(figsize=(7, 2.2))
        pred_seq = predictions[seq_idx]
        gt_seq = ground_truth[seq_idx]
        t_len, n_vars = pred_seq.shape
        t = np.arange(t_len)
        p0 = t_len // 2 if pred_start is None else int(pred_start)
        ax.axvline(x=p0, color="#8C8C8C", linestyle=(0, (4, 2)), linewidth=1.0, alpha=0.9, label="Prediction start")
        for r in range(n_vars):
            ax.plot(t, gt_seq[:, r], color="#9A9A9A", linewidth=1.0, alpha=0.35, label="Ground truth" if r == 0 else None)
            yp = pred_seq[:, r].copy()
            yp[:p0] = np.nan
            ax.plot(t, yp, color="#2E86AB", linewidth=1.2, alpha=0.45, label="Prediction" if r == 0 else None)
        ax.set_ylabel("All regions")
        ax.legend(loc="upper right", fontsize=8, frameon=True)
        plt.tight_layout()
        fig.savefig(output_dir / f"long_example_{mode}_seq{seq_idx+1}_seed{seed}.svg", format="svg")
        plt.close(fig)


def prepare_input_and_gt(val_examples, val_seq_indices, example_idx, T_in, target_length):
    first_example = val_examples[example_idx].T  # (T, C)
    seq_id = int(val_seq_indices[example_idx])
    input_sequence = first_example[:T_in, :]
    gt_parts = [input_sequence]
    first_future = first_example[T_in:, :]
    if first_future.size > 0:
        gt_parts.append(first_future)
    same_seq_idx = np.where(val_seq_indices == seq_id)[0]
    sorted_idx = sorted(same_seq_idx)
    start_pos = sorted_idx.index(int(example_idx))
    for j in sorted_idx[start_pos + 1 :]:
        gt_parts.append(val_examples[j].T)
    gt = np.concatenate(gt_parts, axis=0)
    if gt.shape[0] >= target_length:
        gt = gt[:target_length, :]
    else:
        pad = target_length - gt.shape[0]
        gt = np.concatenate([gt, np.repeat(gt[-1:, :], pad, axis=0)], axis=0)
    x = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0)
    return x, gt


def build_model_from_checkpoint(ckpt: dict, device: torch.device):
    model_type = ckpt["model_type"]
    cfg = ckpt.get("config", {})
    n_vars = int(ckpt["n_vars"])
    if model_type == "DECAY":
        model = ExpDecayBaseline(n_vars=n_vars, output_activation=cfg.get("output_activation", "softplus"))
    elif model_type == "RVAR":
        model = RectifiedVARBaseline(
            n_vars=n_vars,
            T_in=int(ckpt["T_in"]),
            K_lags=int(ckpt.get("K_lags", cfg.get("K_lags", ckpt["T_in"]))),
            dropout=float(cfg.get("dropout", 0.0)),
            output_activation=cfg.get("output_activation", "softplus"),
        )
    elif model_type == "GRU":
        model = TinyGRUForecaster(
            n_vars=n_vars,
            hidden_size=int(ckpt.get("hidden_size", cfg.get("hidden_size", 128))),
            num_layers=int(cfg.get("num_layers", 1)),
            dropout=float(cfg.get("dropout", 0.0)),
            output_activation=cfg.get("output_activation", "softplus"),
        )
    else:
        raise ValueError(f"Unknown model_type in checkpoint: {model_type}")
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    return model, model_type, cfg


def main():
    p = argparse.ArgumentParser("Inference for simple 2p baselines")
    p.add_argument("--data_path", type=str, default="data_processed/data2p_A_3_1189451_S10_C_dec_10Hz.npy")
    p.add_argument("--val_examples_path", type=str, default="")
    p.add_argument("--val_seq_indices_path", type=str, default="")
    p.add_argument("--checkpoints", nargs="*", default=[
        "checkpoints_DECAY/best_model.pt",
        "checkpoints_RVAR/best_model.pt",
        "checkpoints_GRU/best_model.pt",
    ])
    p.add_argument("--long_pred_len", type=int, default=420)
    p.add_argument("--num_sequences", type=int, default=10)
    p.add_argument("--seed", type=int, default=102)
    p.add_argument("--output_dir", type=str, default="evaluation_results/simple_2p")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.val_examples_path and args.val_seq_indices_path:
        val_examples = np.load(args.val_examples_path)
        val_seq_indices = np.load(args.val_seq_indices_path).astype(np.int64, copy=False)
    else:
        data = np.load(args.data_path)
        n_seq, n_sub, n_vars, n_time = data.shape
        examples = data.reshape(n_seq * n_sub, n_vars, n_time)
        seq_idx = np.repeat(np.arange(n_seq), n_sub)
        n = len(examples)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        val_examples = examples[n_train : n_train + n_val]
        val_seq_indices = seq_idx[n_train : n_train + n_val]

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    n_val = len(val_examples)
    chosen = np.random.choice(n_val, size=min(args.num_sequences, n_val), replace=False)

    for ckpt_path in args.checkpoints:
        cp = Path(ckpt_path)
        if not cp.exists():
            print(f"Skipping missing checkpoint: {cp}")
            continue
        ckpt = torch.load(cp, map_location=device, weights_only=False)
        model, model_type, cfg = build_model_from_checkpoint(ckpt, device)
        T_in = int(ckpt["T_in"])
        config_name = f"{T_in}_{args.long_pred_len}"
        seed_dir = out_root / f"seed_{args.seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        preds_all = []
        gts_all = []
        for idx in chosen:
            x0, gt = prepare_input_and_gt(
                val_examples=val_examples,
                val_seq_indices=val_seq_indices,
                example_idx=int(idx),
                T_in=T_in,
                target_length=T_in + args.long_pred_len,
            )
            x0 = x0.to(device)
            rolled = rollout_model(model, x0, pred_len=args.long_pred_len, T_in=T_in)
            pred = rolled[0].detach().cpu().numpy()
            pred = np.clip(pred, a_min=0.0, a_max=None)
            preds_all.append(pred)
            gts_all.append(gt)

        preds_all = np.asarray(preds_all)
        gts_all = np.asarray(gts_all)
        np.save(seed_dir / f"long_predictions_{config_name}_{model_type}.npy", preds_all)
        np.save(seed_dir / f"long_ground_truth_{config_name}.npy", gts_all)

        pred_scored = preds_all[:, T_in:, :]
        gt_scored = gts_all[:, T_in:, :]
        save_sequences_to_neurobench_csv(pred_scored, seed_dir / f"long_predictions_scored_{model_type}.csv")
        save_sequences_to_neurobench_csv(gt_scored, seed_dir / "long_ground_truth_scored.csv")
        plot_prediction_examples(
            predictions=preds_all,
            ground_truth=gts_all,
            output_dir=seed_dir,
            mode=model_type,
            n_examples=min(10, len(preds_all)),
            pred_start=T_in,
            seed=args.seed,
        )
        print(
            f"[{model_type}] saved predictions with stats: "
            f"min={preds_all.min():.6g} max={preds_all.max():.6g} mean={preds_all.mean():.6g}"
        )


if __name__ == "__main__":
    main()

