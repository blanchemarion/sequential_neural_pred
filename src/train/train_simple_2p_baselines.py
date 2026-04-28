from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

_SRC_ROOT = Path(__file__).resolve().parent.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from helpers.preprocess_helpers import load_data, reshape_to_examples, filter_examples_by_nan
from models.simple_2p_baselines import (
    ExpDecayBaseline,
    RectifiedVARBaseline,
    TinyGRUForecaster,
)


def split_blocked(examples: np.ndarray, train_ratio: float, val_ratio: float):
    n = len(examples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = examples[:n_train]
    val = examples[n_train : n_train + n_val]
    test = examples[n_train + n_val :]
    return train, val, test


class OneStepDataset(torch.utils.data.Dataset):
    def __init__(self, examples_nct: np.ndarray, T_in: int):
        self.x = examples_nct[:, :, :T_in].transpose(0, 2, 1).astype(np.float32)  # (N, T_in, C)
        self.y = examples_nct[:, :, T_in : T_in + 1].transpose(0, 2, 1).astype(np.float32)  # (N, 1, C)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]), torch.from_numpy(self.y[idx])


def compute_target_scale(train_examples_nct: np.ndarray, T_in: int, q: float) -> float:
    y = train_examples_nct[:, :, T_in : T_in + 1]
    pos = y[y > 0]
    if pos.size == 0:
        return 1.0
    return float(max(np.quantile(pos, q), 1e-8))


def weighted_smooth_l1_loss(pred, target, beta, target_scale, max_weight):
    per_elem = F.smooth_l1_loss(pred, target, reduction="none")
    weight = 1.0 + float(beta) * torch.clamp(target / float(target_scale), min=0.0, max=float(max_weight))
    return (weight * per_elem).mean(), weight.mean()


def build_model(args, n_vars: int):
    if args.model_type == "decay":
        return ExpDecayBaseline(n_vars=n_vars, output_activation=args.output_activation), "DECAY", {}
    if args.model_type == "rectified_var":
        extra = {"K_lags": int(args.K_lags)}
        return (
            RectifiedVARBaseline(
                n_vars=n_vars,
                T_in=args.T_in,
                K_lags=args.K_lags,
                dropout=args.dropout,
                output_activation=args.output_activation,
            ),
            "RVAR",
            extra,
        )
    if args.model_type == "gru":
        extra = {"hidden_size": int(args.hidden_size), "num_layers": int(args.num_layers)}
        return (
            TinyGRUForecaster(
                n_vars=n_vars,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                dropout=args.dropout,
                output_activation=args.output_activation,
            ),
            "GRU",
            extra,
        )
    raise ValueError(f"Unknown model_type: {args.model_type}")


def run_epoch(model, loader, optimizer, device, beta, target_scale, max_weight, train: bool):
    model.train(train)
    total = 0.0
    total_w = 0.0
    pred_stats = []
    targ_stats = []
    neg_fracs = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x).clamp_min(0.0)
        loss, avg_w = weighted_smooth_l1_loss(pred, y, beta, target_scale, max_weight)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total += float(loss.item())
        total_w += float(avg_w.item())
        pred_stats.append((float(pred.min().item()), float(pred.max().item()), float(pred.mean().item())))
        targ_stats.append((float(y.min().item()), float(y.max().item()), float(y.mean().item())))
        neg_fracs.append(float((pred < 0).float().mean().item()))

    n = max(1, len(loader))
    pmin = min(s[0] for s in pred_stats)
    pmax = max(s[1] for s in pred_stats)
    pmean = float(np.mean([s[2] for s in pred_stats]))
    tmin = min(s[0] for s in targ_stats)
    tmax = max(s[1] for s in targ_stats)
    tmean = float(np.mean([s[2] for s in targ_stats]))
    return {
        "loss": total / n,
        "weighted_loss": total / n,
        "avg_weight": total_w / n,
        "pred_min": pmin,
        "pred_max": pmax,
        "pred_mean": pmean,
        "target_min": tmin,
        "target_max": tmax,
        "target_mean": tmean,
        "pred_negative_fraction": float(np.mean(neg_fracs)),
    }


def main():
    p = argparse.ArgumentParser("Train simple 2p baselines")
    p.add_argument("--data_path", type=str, default="data_processed/data2p_A_3_1189451_S10_C_dec_10Hz.npy")
    p.add_argument("--model_type", type=str, default="rectified_var", choices=("decay", "rectified_var", "gru"))
    p.add_argument("--T_in", type=int, default=90)
    p.add_argument("--T_out_train", type=int, default=1)
    p.add_argument("--K_lags", type=int, default=60)
    p.add_argument("--hidden_size", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--output_activation", type=str, default="softplus", choices=("softplus", "clamp", "relu"))
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--num_epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--loss_beta", type=float, default=2.0)
    p.add_argument("--target_scale_quantile", type=float, default=0.95)
    p.add_argument("--loss_max_weight", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=101)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = load_data(args.data_path)
    examples, seq_idx = reshape_to_examples(data)
    examples, seq_idx = filter_examples_by_nan(examples, seq_idx, T_in=args.T_in, T_out=args.T_out_train)
    train_ex, val_ex, test_ex = split_blocked(examples, args.train_ratio, args.val_ratio)
    print(f"Split sizes: train={len(train_ex)}, val={len(val_ex)}, test={len(test_ex)}")

    if len(train_ex) == 0:
        raise ValueError("No training examples.")
    n_vars = int(train_ex.shape[1])
    if args.T_in + args.T_out_train > train_ex.shape[2]:
        raise ValueError("T_in + T_out_train exceeds example length.")

    target_scale = compute_target_scale(train_ex, args.T_in, args.target_scale_quantile)
    print(f"target_scale (q={args.target_scale_quantile}): {target_scale:.6g}")

    tr_ds = OneStepDataset(train_ex, args.T_in)
    va_ds = OneStepDataset(val_ex, args.T_in)
    bs = min(args.batch_size, len(tr_ds))
    tr_loader = DataLoader(tr_ds, batch_size=bs, shuffle=True, drop_last=False, num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=min(args.batch_size, max(1, len(va_ds))), shuffle=False, drop_last=False, num_workers=0)

    model, model_tag, extra_cfg = build_model(args, n_vars)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    ckpt_dir = Path(f"checkpoints_{model_tag}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.num_epochs + 1):
        train_m = run_epoch(model, tr_loader, opt, device, args.loss_beta, target_scale, args.loss_max_weight, train=True)
        val_m = run_epoch(model, va_loader, opt, device, args.loss_beta, target_scale, args.loss_max_weight, train=False)
        print(
            f"[{model_tag}] epoch {epoch:03d} | "
            f"train_loss={train_m['loss']:.6f} val_loss={val_m['loss']:.6f} | "
            f"pred(min/max/mean)=({val_m['pred_min']:.4g}/{val_m['pred_max']:.4g}/{val_m['pred_mean']:.4g}) | "
            f"target(min/max/mean)=({val_m['target_min']:.4g}/{val_m['target_max']:.4g}/{val_m['target_mean']:.4g}) | "
            f"neg_frac={val_m['pred_negative_fraction']:.6f}"
        )

        if val_m["loss"] < best_val:
            best_val = val_m["loss"]
            cfg = vars(args).copy()
            cfg.update(extra_cfg)
            cfg["n_vars"] = n_vars
            cfg["target_scale"] = target_scale
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_type": model_tag,
                    "n_vars": n_vars,
                    "T_in": args.T_in,
                    "K_lags": int(args.K_lags),
                    "hidden_size": int(args.hidden_size),
                    "config": cfg,
                },
                ckpt_dir / "best_model.pt",
            )
            print(f"  saved best -> {ckpt_dir / 'best_model.pt'}")

    print("Done.")


if __name__ == "__main__":
    main()

