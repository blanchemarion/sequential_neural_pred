"""
Train a naive non-transformer baseline: linear autoregressive model.

The model predicts one next timestep from the last T_in timesteps and is
rolled out autoregressively for T_out steps during validation.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import sys

_SRC_ROOT = Path(__file__).resolve().parent.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from helpers.preprocess_helpers import (
    create_dataloaders,
    filter_examples_by_nan,
    load_data,
    normalize_after_split_input_only,
    reshape_to_examples,
    split_by_sequences,
)
from models.model_var_baseline import create_var_baseline


def plot_learning_curves(train_tf, val_tf, val_ar, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(train_tf) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_tf, label="Train TF loss", linewidth=2)
    plt.plot(epochs, val_tf, label="Val TF loss", linewidth=2)
    plt.plot(epochs, val_ar, label="Val AR rollout loss", linewidth=2)
    best_idx = int(np.argmin(val_ar))
    plt.scatter(
        [epochs[best_idx]],
        [val_ar[best_idx]],
        s=70,
        color="gold",
        edgecolors="black",
        label=f"Best AR: {val_ar[best_idx]:.4f} (epoch {epochs[best_idx]})",
        zorder=5,
    )
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("Linear AR Baseline Learning Curves")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "learning_curves_detailed.png", dpi=300, bbox_inches="tight")
    plt.close()


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running = 0.0
    n_batches = 0

    for x, y in tqdm(loader, desc="Training", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model.forward_teacher_forcing(x, y)
        loss = criterion(pred, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running += float(loss.item())
        n_batches += 1

    return running / max(n_batches, 1)


@torch.no_grad()
def validate_epoch(model, loader, criterion, device):
    model.eval()
    tf_running = 0.0
    ar_running = 0.0
    n_batches = 0

    for x, y in tqdm(loader, desc="Validation", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred_tf = model.forward_teacher_forcing(x, y)
        pred_ar = model.forward_autoregressive(x)

        tf_running += float(criterion(pred_tf, y).item())
        ar_running += float(criterion(pred_ar, y).item())
        n_batches += 1

    denom = max(n_batches, 1)
    return tf_running / denom, ar_running / denom


def main():
    print("=" * 80)
    print("TRAINING LINEAR AUTOREGRESSIVE BASELINE")
    print("=" * 80)

    config = {
        "model_name": "VAR_BASELINE",
        "data_path": "data_processed/data100_ba16.npy",
        "T_in": 90,
        "T_out": 90,
        "n_vars": 16,
        "batch_size": 2048,
        "num_workers": 8,
        "num_epochs": 150,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "train_ratio": 0.8,
        "split_seed": 101,
        "training_seed": 103,  # vary across 101, 102, 103
        "save_dir": "checkpoints_VAR_BASELINE_90_90_seed103",
        "early_stop_patience": 25,
        "early_stop_min_delta": 1e-6,
        "checkpoint_first_epoch": 5,
        "checkpoint_every": 25,
    }

    # Windows uses "spawn" multiprocessing and can fail when pickling large in-memory
    # datasets for DataLoader workers. Force single-process loading for stability.
    effective_num_workers = int(config["num_workers"])
    if os.name == "nt" and effective_num_workers > 0:
        print(
            f"[INFO] Windows detected: overriding num_workers "
            f"{effective_num_workers} -> 0 for DataLoader stability."
        )
        effective_num_workers = 0

    torch.manual_seed(config["training_seed"])
    np.random.seed(config["training_seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["training_seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(config["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device}")
    print(f"Saving to: {save_dir}")
    print(f"Split seed: {config['split_seed']}")
    print(f"Training seed: {config['training_seed']}")

    print("\nPreparing data...")
    data_array = load_data(config["data_path"])
    examples, sequence_indices = reshape_to_examples(data_array)
    del data_array

    examples, sequence_indices = filter_examples_by_nan(
        examples,
        sequence_indices,
        T_in=config["T_in"],
        T_out=config["T_out"],
    )

    train_examples, val_examples, _, val_seq_indices = split_by_sequences(
        examples,
        sequence_indices,
        train_ratio=config["train_ratio"],
        random_seed=config["split_seed"],
    )
    del examples, sequence_indices

    run_stem = (
        f"{Path(config['data_path']).stem}"
        f"_Tin{config['T_in']}_Tout{config['T_out']}_splitseed{config['split_seed']}_var"
    )
    proc_root = Path("data_processed")
    proc_root.mkdir(parents=True, exist_ok=True)
    stats_npy = proc_root / f"train_norm_stats_{run_stem}.npy"
    val_npy = proc_root / f"processed_val_{run_stem}.npy"
    val_seq_npy = proc_root / f"processed_val_seq_indices_{run_stem}.npy"

    train_examples, val_examples, _, _ = normalize_after_split_input_only(
        train_examples,
        val_examples,
        T_in=config["T_in"],
        eps=1e-6,
        save_stats_path=str(stats_npy),
    )
    np.save(val_npy, val_examples)
    np.save(val_seq_npy, val_seq_indices.astype(np.int64, copy=False))

    config["normalization_stats_base"] = str(stats_npy.with_suffix("").resolve())
    config["processed_val_examples_path"] = str(val_npy.resolve())
    config["processed_val_seq_indices_path"] = str(val_seq_npy.resolve())

    # split_by_sequences resets NumPy's RNG; restore independent training state.
    torch.manual_seed(config["training_seed"])
    np.random.seed(config["training_seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["training_seed"])

    train_loader, val_loader = create_dataloaders(
        train_examples,
        val_examples,
        T_in=config["T_in"],
        T_out=config["T_out"],
        batch_size=config["batch_size"],
        num_workers=effective_num_workers,
    )
    del train_examples, val_examples

    model = create_var_baseline(
        n_vars=config["n_vars"],
        T_in=config["T_in"],
        T_out=config["T_out"],
        bias=True,
        device=device,
    )
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    print(f"\nModel parameters: {model.count_parameters():,}")

    train_hist, val_tf_hist, val_ar_hist = [], [], []
    best_val_ar = float("inf")
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, config["num_epochs"] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_tf, val_ar = validate_epoch(model, val_loader, criterion, device)

        train_hist.append(train_loss)
        val_tf_hist.append(val_tf)
        val_ar_hist.append(val_ar)

        print(
            f"Epoch {epoch:03d}/{config['num_epochs']} | "
            f"Train(TF) {train_loss:.6f} | Val(TF) {val_tf:.6f} | Val(AR) {val_ar:.6f}"
        )

        if epoch == config["checkpoint_first_epoch"] or (
            config["checkpoint_every"] > 0 and epoch % config["checkpoint_every"] == 0
        ):
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_tf_loss": val_tf,
                    "val_ar_loss": val_ar,
                    "config": config,
                },
                save_dir / f"checkpoint_fixed_epoch_{epoch}.pt",
            )

        if val_ar < best_val_ar - config["early_stop_min_delta"]:
            best_val_ar = val_ar
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_tf_loss": val_tf,
                    "val_ar_loss": val_ar,
                    "config": config,
                },
                save_dir / "best_model.pt",
            )
        else:
            no_improve += 1

        if no_improve >= config["early_stop_patience"]:
            print(f"Early stopping at epoch {epoch} (best AR val at epoch {best_epoch}).")
            break

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_hist,
            "val_tf_losses": val_tf_hist,
            "val_ar_losses": val_ar_hist,
            "best_epoch": best_epoch,
            "best_val_ar_loss": best_val_ar,
            "config": config,
        },
        save_dir / "final_model.pt",
    )

    plot_learning_curves(train_hist, val_tf_hist, val_ar_hist, save_dir)
    print("\nTraining complete.")
    print(f"Best AR validation loss: {best_val_ar:.6f} (epoch {best_epoch})")
    print(f"Artifacts: {save_dir}")


if __name__ == "__main__":
    try:
        main()
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
