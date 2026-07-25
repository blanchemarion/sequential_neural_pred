"""Train the teacher-forced GRU_AR baseline with AR validation."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

_SRC_ROOT = Path(__file__).resolve().parent.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from helpers.globals import load_globals, merge_paths_section, repo_root, resolve_repo_relative
from helpers.preprocess_helpers import (
    TimeSeriesDataset,
    create_dataloaders,
    filter_examples_by_nan,
    load_data,
    normalize_after_split_input_only,
    reshape_to_examples,
    split_by_sequences,
)
from models.model_gru_ar import create_gru_ar


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--globals", type=Path, default=None)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Training device; auto selects CUDA when available",
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> dict:
    root = repo_root()
    full_globals = load_globals(args.globals)
    paths = merge_paths_section(full_globals)
    section = full_globals.get("train_gru_ar")
    if not isinstance(section, dict):
        raise ValueError("globals.json is missing the train_gru_ar configuration")

    config = dict(section)
    data_path = (
        args.data_path.resolve()
        if args.data_path is not None
        else resolve_repo_relative(root, Path(paths["data_processed"]) / config["data_filename"])
    )
    save_dir = (
        args.save_dir.resolve()
        if args.save_dir is not None
        else resolve_repo_relative(root, config["save_dir"])
    )
    if args.epochs is not None:
        config["num_epochs"] = int(args.epochs)
    if args.batch_size is not None:
        config["batch_size"] = int(args.batch_size)

    config["data_path"] = str(data_path)
    config["save_dir"] = str(save_dir)
    config["globals_path"] = str(
        (args.globals.resolve() if args.globals is not None else root / "globals.json")
    )
    return config


def set_deterministic_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _fallback_nonempty_train_loader(
    train_examples: np.ndarray,
    val_examples: np.ndarray,
    config: dict,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    """Preserve configured batching unless drop_last would erase the train set."""
    train_loader, val_loader = create_dataloaders(
        train_examples,
        val_examples,
        T_in=int(config["T_in"]),
        T_out=int(config["T_out"]),
        batch_size=int(config["batch_size"]),
        num_workers=num_workers,
    )
    if len(train_loader) > 0:
        config["drop_last_train"] = True
        return train_loader, val_loader

    print(
        "[WARN] The configured physical batch exceeds this data partition; "
        "using one non-dropped training batch."
    )
    dataset = TimeSeriesDataset(
        train_examples, T_in=int(config["T_in"]), T_out=int(config["T_out"])
    )
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=False,
    )
    config["drop_last_train"] = False
    return train_loader, val_loader


def prepare_data(config: dict, num_workers: int):
    data_array = load_data(config["data_path"])
    examples, sequence_indices = reshape_to_examples(data_array)
    del data_array

    examples, sequence_indices = filter_examples_by_nan(
        examples,
        sequence_indices,
        T_in=int(config["T_in"]),
        T_out=int(config["T_out"]),
    )
    train_examples, val_examples, _, val_sequence_indices = split_by_sequences(
        examples,
        sequence_indices,
        train_ratio=float(config["train_ratio"]),
        random_seed=int(config["random_seed"]),
    )
    del examples, sequence_indices

    data_stem = Path(config["data_path"]).stem
    run_stem = (
        f"{data_stem}_Tin{config['T_in']}_Tout{config['T_out']}"
        f"_seed{config['random_seed']}_gru_ar"
    )
    processed_root = Path(config["data_path"]).parent
    stats_path = processed_root / f"train_norm_stats_{run_stem}.npy"
    val_path = processed_root / f"processed_val_{run_stem}.npy"
    val_sequence_path = processed_root / f"processed_val_seq_indices_{run_stem}.npy"

    train_examples, val_examples, _, _ = normalize_after_split_input_only(
        train_examples,
        val_examples,
        T_in=int(config["T_in"]),
        eps=1e-6,
        save_stats_path=str(stats_path),
    )
    np.save(val_path, val_examples.astype(np.float32, copy=False))
    np.save(val_sequence_path, val_sequence_indices.astype(np.int64, copy=False))

    config["normalization_stats_base"] = str(stats_path.with_suffix("").resolve())
    config["processed_val_examples_path"] = str(val_path.resolve())
    config["processed_val_seq_indices_path"] = str(val_sequence_path.resolve())
    config["data_layout"] = "NCT_on_disk_BTC_in_model"
    config["split_method"] = "sequence_id"
    config["normalization"] = "per_region_train_input_only_zscore"

    loaders = _fallback_nonempty_train_loader(
        train_examples, val_examples, config, num_workers
    )
    del train_examples, val_examples
    return loaders


def train_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    criterion,
    device,
    accumulation_steps: int,
    gradient_clip: float,
    use_amp: bool,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    total = 0.0
    count = 0

    for batch_index, (context, target) in enumerate(tqdm(loader, desc="Training", leave=False)):
        context = context.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            prediction = model.forward_teacher_forced(context, target)
            unscaled_loss = criterion(prediction, target)
            loss = unscaled_loss / accumulation_steps

        scaler.scale(loss).backward()
        should_step = (batch_index + 1) % accumulation_steps == 0 or (
            batch_index + 1 == len(loader)
        )
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        total += float(unscaled_loss.item())
        count += 1

    if count == 0:
        raise RuntimeError("Training loader produced no batches")
    return total / count


@torch.no_grad()
def validate_epoch(model, loader, criterion, device, use_amp: bool) -> tuple[float, float]:
    model.eval()
    amp_enabled = bool(use_amp and device.type == "cuda")
    teacher_forced_total = 0.0
    autoregressive_total = 0.0
    count = 0

    for context, target in tqdm(loader, desc="Validation", leave=False):
        context = context.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            teacher_forced = model.forward_teacher_forced(context, target)
            autoregressive = model.forecast_autoregressive(context, horizon=target.shape[1])
            teacher_forced_loss = criterion(teacher_forced, target)
            autoregressive_loss = criterion(autoregressive, target)
        teacher_forced_total += float(teacher_forced_loss.item())
        autoregressive_total += float(autoregressive_loss.item())
        count += 1

    if count == 0:
        raise RuntimeError("Validation loader produced no batches")
    return teacher_forced_total / count, autoregressive_total / count


def checkpoint_payload(
    model,
    optimizer,
    scheduler,
    config: dict,
    epoch: int,
    train_loss: float,
    val_teacher_forced_loss: float,
    val_autoregressive_loss: float,
) -> dict:
    return {
        "epoch": int(epoch),
        "mode": "GRU_AR",
        "model_class": "GRUAR",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_loss": float(train_loss),
        "val_teacher_forced_loss": float(val_teacher_forced_loss),
        "val_autoregressive_loss": float(val_autoregressive_loss),
        "parameter_count": int(model.count_parameters()),
        "config": dict(config),
    }


def plot_learning_curves(history: list[dict], save_dir: Path) -> None:
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, [row["train_teacher_forced_mae"] for row in history], label="Train TF")
    plt.plot(
        epochs,
        [row["val_teacher_forced_mae"] for row in history],
        label="Validation TF",
    )
    plt.plot(
        epochs,
        [row["val_autoregressive_mae"] for row in history],
        label="Validation AR",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Unweighted MAE")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "learning_curves_detailed.png", dpi=300)
    plt.close()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = build_config(args)
    device = resolve_device(args.device)
    workers = int(args.num_workers)
    if os.name == "nt" and workers > 0:
        print(f"[INFO] Windows: overriding num_workers {workers} -> 0")
        workers = 0

    set_deterministic_seed(int(config["random_seed"]))
    save_dir = Path(config["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TRAINING GRU_AR (100% TEACHER FORCING; AR CHECKPOINT SELECTION)")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Data: {config['data_path']}")
    print(f"Checkpoints: {save_dir}")

    train_loader, val_loader = prepare_data(config, workers)
    model = create_gru_ar(
        n_vars=int(config["n_vars"]),
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
        context_length=int(config["T_in"]),
        forecast_length=int(config["T_out"]),
        device=device,
    )
    parameter_count = model.count_parameters()
    config["parameter_count"] = parameter_count
    transformer_reference = config["transformer_reference"]
    if float(config["dropout"]) != float(transformer_reference["dropout"]):
        raise ValueError("GRU dropout must match the reference Transformer dropout")
    print(
        f"GRU_AR: hidden_dim={config['hidden_dim']}, layers={config['num_layers']}, "
        f"dropout={config['dropout']}, parameters={parameter_count:,}"
    )
    print(
        "Transformer reference: "
        f"d_model={transformer_reference['d_model']}, "
        f"layers={transformer_reference['n_layers']}, "
        f"heads={transformer_reference['n_heads']}, "
        f"d_ff={transformer_reference['d_ff']}, "
        f"dropout={transformer_reference['dropout']}"
    )

    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    updates_per_epoch = max(
        1,
        (len(train_loader) + int(config["accumulation_steps"]) - 1)
        // int(config["accumulation_steps"]),
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(config["max_lr"]),
        epochs=int(config["num_epochs"]),
        steps_per_epoch=updates_per_epoch,
        pct_start=float(config["scheduler_pct_start"]),
        div_factor=float(config["scheduler_div_factor"]),
        final_div_factor=float(config["scheduler_final_div_factor"]),
        anneal_strategy=str(config["scheduler_anneal_strategy"]),
        three_phase=False,
    )

    history: list[dict] = []
    best_ar_loss = float("inf")
    best_epoch = 0
    no_improvement = 0
    last_payload = None

    for epoch in range(1, int(config["num_epochs"]) + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            device,
            accumulation_steps=int(config["accumulation_steps"]),
            gradient_clip=float(config["gradient_clip"]),
            use_amp=bool(config["use_mixed_precision"]),
        )
        val_tf_loss, val_ar_loss = validate_epoch(
            model,
            val_loader,
            criterion,
            device,
            use_amp=bool(config["use_mixed_precision"]),
        )
        row = {
            "epoch": epoch,
            "train_teacher_forced_mae": train_loss,
            "val_teacher_forced_mae": val_tf_loss,
            "val_autoregressive_mae": val_ar_loss,
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{config['num_epochs']} | "
            f"Train TF MAE {train_loss:.6f} | Val TF MAE {val_tf_loss:.6f} | "
            f"Val AR MAE {val_ar_loss:.6f}"
        )

        last_payload = checkpoint_payload(
            model,
            optimizer,
            scheduler,
            config,
            epoch,
            train_loss,
            val_tf_loss,
            val_ar_loss,
        )
        if epoch == int(config["checkpoint_first_epoch"]) or (
            int(config["checkpoint_every"]) > 0
            and epoch % int(config["checkpoint_every"]) == 0
        ):
            torch.save(last_payload, save_dir / f"checkpoint_fixed_epoch_{epoch}.pt")

        if val_ar_loss < best_ar_loss - float(config["early_stop_min_delta"]):
            best_ar_loss = val_ar_loss
            best_epoch = epoch
            no_improvement = 0
            torch.save(last_payload, save_dir / "best_model.pt")
        else:
            no_improvement += 1

        if no_improvement >= int(config["early_stop_patience"]):
            print(f"Early stopping: best AR validation was epoch {best_epoch}.")
            break

    if last_payload is None:
        raise RuntimeError("No training epoch completed")
    last_payload["history"] = history
    last_payload["best_epoch"] = best_epoch
    last_payload["best_val_autoregressive_loss"] = best_ar_loss
    torch.save(last_payload, save_dir / "final_model.pt")
    (save_dir / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    plot_learning_curves(history, save_dir)
    print(f"Best AR validation MAE: {best_ar_loss:.6f} (epoch {best_epoch})")


if __name__ == "__main__":
    try:
        main()
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
