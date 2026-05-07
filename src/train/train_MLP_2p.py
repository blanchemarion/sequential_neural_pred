from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import sys

_SRC_ROOT = Path(__file__).resolve().parent.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from models.mlp_2p import MLP2P


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_raw_csv_time_major(csv_path: Path) -> np.ndarray:
    arr = pd.read_csv(csv_path, header=None).values.astype(np.float32)  # (N, T)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D CSV (N, T), got shape {arr.shape}")
    return arr.T  # (T, N)


def print_data_summary(x_tn: np.ndarray, name: str = "dataset") -> None:
    n_timepoints, n_neurons = x_tn.shape
    flat = x_tn.reshape(-1)
    zero_frac = float(np.mean(flat == 0))
    neg_frac = float(np.mean(flat < 0))
    print(f"[{name}] neurons={n_neurons} timepoints={n_timepoints}")
    print(
        f"[{name}] min={float(np.min(flat)):.6g} max={float(np.max(flat)):.6g} "
        f"mean={float(np.mean(flat)):.6g} std={float(np.std(flat)):.6g}"
    )
    print(f"[{name}] frac_exact_zero={zero_frac:.6f} frac_negative={neg_frac:.6f}")
    if neg_frac > 0:
        print(f"[WARNING] {name} contains negative values ({100.0 * neg_frac:.4f}%).")


@dataclass
class SplitRegions:
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int
    gap_used: int

    def to_dict(self) -> dict:
        return {
            "train": [self.train_start, self.train_end],
            "val": [self.val_start, self.val_end],
            "test": [self.test_start, self.test_end],
            "gap_used": self.gap_used,
        }


def compute_blocked_regions(total_T: int, T_in: int, gap: int) -> SplitRegions:
    train_end = int(total_T * 0.70)
    val_end = int(total_T * 0.85)
    gap_used = int(gap)

    def starts_count(start: int, end: int) -> int:
        return max(0, end - (start + T_in))

    while gap_used >= 0:
        val_start = train_end + gap_used
        test_start = val_end + gap_used
        val_count = starts_count(val_start, val_end)
        test_count = starts_count(test_start, total_T)
        if val_count > 0 and test_count > 0:
            return SplitRegions(
                train_start=0,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=total_T,
                gap_used=gap_used,
            )
        gap_used -= 1

    raise ValueError("Unable to create non-empty val/test split; dataset too short for chosen T_in.")

class MultiStepForecastDataset(Dataset):
    def __init__(
        self,
        data_tn: np.ndarray,
        region_start: int,
        region_end: int,
        T_in: int = 90,
        T_out: int = 16,
        stride: int = 1,
        max_windows: Optional[int] = None,
    ) -> None:
        self.data_tn = data_tn
        self.T_in = int(T_in)
        self.T_out = int(T_out)
        self.region_start = int(region_start)
        self.region_end = int(region_end)
        self.stride = int(stride)

        max_start = self.region_end - self.T_in - self.T_out

        if max_start < self.region_start:
            self.starts = np.array([], dtype=np.int64)
        else:
            self.starts = np.arange(
                self.region_start,
                max_start + 1,
                self.stride,
                dtype=np.int64,
            )

        if max_windows is not None and len(self.starts) > max_windows:
            self.starts = self.starts[: int(max_windows)]

    def __len__(self) -> int:
        return int(self.starts.shape[0])

    def __getitem__(self, idx: int):
        s = int(self.starts[idx])

        context = self.data_tn[s : s + self.T_in, :]  
        target = self.data_tn[s + self.T_in : s + self.T_in + self.T_out, :]

        return torch.from_numpy(context), torch.from_numpy(target)

class OneStepForecastDataset(Dataset):
    def __init__(
        self,
        data_tn: np.ndarray,
        region_start: int,
        region_end: int,
        T_in: int = 90,
        stride: int = 1,
        max_windows: Optional[int] = None,
    ) -> None:
        self.data_tn = data_tn
        self.T_in = int(T_in)
        self.region_start = int(region_start)
        self.region_end = int(region_end)
        self.stride = int(stride)

        max_start = self.region_end - self.T_in - 1
        if max_start < self.region_start:
            self.starts = np.array([], dtype=np.int64)
        else:
            self.starts = np.arange(self.region_start, max_start + 1, self.stride, dtype=np.int64)
        if max_windows is not None and len(self.starts) > max_windows:
            self.starts = self.starts[: int(max_windows)]

    def __len__(self) -> int:
        return int(self.starts.shape[0])

    def __getitem__(self, idx: int):
        s = int(self.starts[idx])
        context = self.data_tn[s : s + self.T_in, :]  # (T_in, N)
        target_next = self.data_tn[s + self.T_in, :]  # (N,)
        return torch.from_numpy(context), torch.from_numpy(target_next)


"""def compute_target_scale(
    data_tn: np.ndarray,
    starts: np.ndarray,
    T_in: int,
    quantile: float,
) -> float:
    if len(starts) == 0:
        return 1.0
    target_idxs = starts + T_in
    targets = data_tn[target_idxs, :].reshape(-1)
    pos = targets[targets > 0]
    if pos.size == 0:
        return 1.0
    scale = float(np.quantile(pos, quantile))
    return max(scale, 1e-6)"""

def compute_target_scale(
    data_tn: np.ndarray,
    starts: np.ndarray,
    T_in: int,
    T_out: int,
    quantile: float,
) -> float:
    if len(starts) == 0:
        return 1.0

    chunks = []
    for s in starts:
        target = data_tn[s + T_in : s + T_in + T_out, :]
        chunks.append(target.reshape(-1))

    targets = np.concatenate(chunks, axis=0)
    pos = targets[targets > 0]

    if pos.size == 0:
        return 1.0

    scale = float(np.quantile(pos, quantile))
    return max(scale, 1e-6)


"""def weighted_smooth_l1_loss(
    pred_bn: torch.Tensor,
    target_bn: torch.Tensor,
    target_scale: float,
    beta: float = 2.0,
    max_weight: float = 10.0,
) -> torch.Tensor:
    base = F.smooth_l1_loss(pred_bn, target_bn, reduction="none")
    scale = max(float(target_scale), 1e-6)
    weight = 1.0 + float(beta) * torch.clamp(target_bn / scale, min=0.0, max=float(max_weight))
    return (weight * base).mean()"""

def weighted_smooth_l1_loss(
    pred_btn: torch.Tensor,
    target_btn: torch.Tensor,
    target_scale: float,
    beta: float = 2.0,
    max_weight: float = 10.0,
) -> torch.Tensor:
    base = F.smooth_l1_loss(pred_btn, target_btn, reduction="none")

    scale = max(float(target_scale), 1e-6)
    weight = 1.0 + float(beta) * torch.clamp(
        target_btn / scale,
        min=0.0,
        max=float(max_weight),
    )

    return (weight * base).mean()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_scale: float,
    loss_beta: float,
    max_weight: float,
) -> dict:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    pred_chunks = []
    target_chunks = []
    for context, target in loader:
        context = context.to(device=device, dtype=torch.float32)
        target = target.to(device=device, dtype=torch.float32)
        pred = model(context) #pred = model(context).squeeze(1)
        loss = weighted_smooth_l1_loss(pred, target, target_scale, beta=loss_beta, max_weight=max_weight)
        total_loss += float(loss.item())
        n_batches += 1
        pred_chunks.append(pred.detach().cpu())
        target_chunks.append(target.detach().cpu())

    if n_batches == 0:
        return {
            "loss": math.nan,
            "pred_min": math.nan,
            "pred_max": math.nan,
            "pred_mean": math.nan,
            "target_min": math.nan,
            "target_max": math.nan,
            "target_mean": math.nan,
            "pred_neg_frac": math.nan,
        }
    pred_all = torch.cat(pred_chunks, dim=0).reshape(-1)
    target_all = torch.cat(target_chunks, dim=0).reshape(-1)

    return {
        "loss": total_loss / n_batches,
        "pred_min": float(pred_all.min().item()),
        "pred_max": float(pred_all.max().item()),
        "pred_mean": float(pred_all.mean().item()),
        "target_min": float(target_all.min().item()),
        "target_max": float(target_all.max().item()),
        "target_mean": float(target_all.mean().item()),
        "pred_neg_frac": float((pred_all < 0).float().mean().item()),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MLP for 2p traces.")
    p.add_argument("--csv_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="checkpoints_MLP_2P")
    p.add_argument("--T_in", type=int, default=90)
    p.add_argument("--T_out", type=int, default=16)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--split_gap", type=int, default=None, help="Default: T_in + 720")
    p.add_argument("--max_windows", type=int, default=None)
    p.add_argument("--d_local", type=int, default=64)
    p.add_argument("--d_pop", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--decay_init", type=float, default=0.965)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--loss_beta", type=float, default=2.0)
    p.add_argument("--max_weight", type=float, default=10.0)
    p.add_argument("--target_scale_quantile", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_tn = load_raw_csv_time_major(Path(args.csv_path))
    print_data_summary(data_tn, name="raw_csv")

    T_total, n_neurons = data_tn.shape
    gap = args.split_gap if args.split_gap is not None else (args.T_in + 720)
    regions = compute_blocked_regions(total_T=T_total, T_in=args.T_in, gap=gap)
    if regions.gap_used != gap:
        print(f"[split] Requested gap={gap}, reduced to gap={regions.gap_used} to keep non-empty val/test.")
    print(f"[split] train={regions.train_start}:{regions.train_end}")
    print(f"[split] val={regions.val_start}:{regions.val_end}")
    print(f"[split] test={regions.test_start}:{regions.test_end}")

    """train_ds = OneStepForecastDataset(data_tn, regions.train_start, regions.train_end, T_in=args.T_in, stride=args.stride, max_windows=args.max_windows)
    val_ds = OneStepForecastDataset(data_tn, regions.val_start, regions.val_end, T_in=args.T_in, stride=1, max_windows=args.max_windows)
    test_ds = OneStepForecastDataset(data_tn, regions.test_start, regions.test_end, T_in=args.T_in, stride=1, max_windows=args.max_windows)"""

    train_ds = MultiStepForecastDataset(data_tn, regions.train_start, regions.train_end, T_in=args.T_in, T_out=args.T_out, stride=args.stride, max_windows=args.max_windows)
    val_ds = MultiStepForecastDataset(data_tn, regions.val_start, regions.val_end, T_in=args.T_in, T_out=args.T_out, stride=1, max_windows=args.max_windows)
    test_ds = MultiStepForecastDataset(data_tn, regions.test_start, regions.test_end, T_in=args.T_in, T_out=args.T_out, stride=1, max_windows=args.max_windows)
    
    print(f"[windows] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    if len(train_ds) == 0:
        raise ValueError("No training windows produced. Adjust T_in/stride/gap or use a longer sequence.")

    """target_scale = compute_target_scale(
        data_tn=data_tn,
        starts=train_ds.starts,
        T_in=args.T_in,
        quantile=args.target_scale_quantile,
    )"""
    target_scale = compute_target_scale(
        data_tn=data_tn,
        starts=train_ds.starts,
        T_in=args.T_in,
        T_out=args.T_out,
        quantile=args.target_scale_quantile,
    )
    print(f"[loss] target_scale(q={args.target_scale_quantile})={target_scale:.6g}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP2P(
        n_neurons=n_neurons,
        T_in=args.T_in,
        T_out=args.T_out,
        d_local=args.d_local,
        d_pop=args.d_pop,
        dropout=args.dropout,
        decay_init=args.decay_init,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val = float("inf")
    best_path = output_dir / "best_model.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        n_train_batches = 0
        for context, target in train_loader:
            context = context.to(device=device, dtype=torch.float32)
            target = target.to(device=device, dtype=torch.float32)
            pred = model(context) #pred = model(context).squeeze(1)
            loss = weighted_smooth_l1_loss(
                pred,
                target,
                target_scale=target_scale,
                beta=args.loss_beta,
                max_weight=args.max_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            total_train_loss += float(loss.item())
            n_train_batches += 1

        train_loss = total_train_loss / max(1, n_train_batches)
        val_stats = evaluate(model, val_loader, device, target_scale, args.loss_beta, args.max_weight)
        test_stats = evaluate(model, test_loader, device, target_scale, args.loss_beta, args.max_weight)

        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_loss:.6f} val_loss={val_stats['loss']:.6f} "
            f"pred(min/max/mean)=({val_stats['pred_min']:.6g}/{val_stats['pred_max']:.6g}/{val_stats['pred_mean']:.6g}) "
            f"target(min/max/mean)=({val_stats['target_min']:.6g}/{val_stats['target_max']:.6g}/{val_stats['target_mean']:.6g}) "
            f"pred_neg_frac={val_stats['pred_neg_frac']:.6f} target_scale={target_scale:.6g}"
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": float(val_stats["loss"]),
            "test_loss": float(test_stats["loss"]),
            "target_scale": target_scale,
        }
        history.append(row)

        if val_stats["loss"] < best_val:
            best_val = float(val_stats["loss"])
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": vars(args),
                "n_neurons": n_neurons,
                "T_in": args.T_in,
                "T_out": args.T_out,
                "d_local": args.d_local,
                "d_pop": args.d_pop,
                "dropout": args.dropout,
                "decay_init": args.decay_init,
                "target_scale": target_scale,
                "dataset_path": str(Path(args.csv_path).as_posix()),
                "split_info": regions.to_dict(),
            }
            torch.save(checkpoint, best_path)
            print(f"[checkpoint] saved best model -> {best_path} (val_loss={best_val:.6f})")

    history_path = output_dir / "train_history.json"
    history_path.write_text(json.dumps(history, indent=2))
    print(f"[done] history saved to {history_path}")
    print(f"[done] best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
