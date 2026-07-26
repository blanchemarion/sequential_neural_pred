"""Train cDMM_SSM with a conditional ELBO and mean-forecast validation."""

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
from models.model_cdmm_ssm import create_cdmm_ssm


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--globals", type=Path, default=None)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--latent-dim", type=int, choices=(4, 8, 12), default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> dict:
    root = repo_root()
    full_globals = load_globals(args.globals)
    paths = merge_paths_section(full_globals)
    section = full_globals.get("train_cdmm_ssm")
    if not isinstance(section, dict):
        raise ValueError("globals.json is missing train_cdmm_ssm")
    config = dict(section)
    config.setdefault("split_seed", int(config.get("random_seed", 101)))
    config.setdefault("training_seed", int(config.get("random_seed", 101)))
    config.setdefault("validation_num_samples", 16)
    config.setdefault("validation_seed", int(config["training_seed"]))
    config["decoder_hidden_dims"] = list(config["decoder_hidden_dims"])
    config["supported_latent_dims"] = list(config["supported_latent_dims"])

    if args.latent_dim is not None:
        config["latent_dim"] = int(args.latent_dim)
    if int(config["latent_dim"]) not in {
        int(value) for value in config["supported_latent_dims"]
    }:
        raise ValueError("Configured latent dimension is not supported")
    if args.epochs is not None:
        config["num_epochs"] = int(args.epochs)
    if args.batch_size is not None:
        config["batch_size"] = int(args.batch_size)

    data_path = (
        args.data_path.resolve()
        if args.data_path is not None
        else resolve_repo_relative(
            root, Path(paths["data_processed"]) / config["data_filename"]
        )
    )
    if args.save_dir is not None:
        save_dir = args.save_dir.resolve()
    else:
        save_name = (
            f"checkpoints_cDMM_SSM_z{config['latent_dim']}_"
            f"{config['T_in']}_{config['T_out']}"
        )
        save_dir = resolve_repo_relative(root, save_name)
    config["data_path"] = str(data_path)
    config["save_dir"] = str(save_dir)
    config["globals_path"] = str(
        args.globals.resolve() if args.globals is not None else root / "globals.json"
    )
    return config


def set_seed(seed: int) -> None:
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


def beta_for_epoch(
    epoch: int, num_epochs: int, beta_max: float, anneal_fraction: float
) -> float:
    """Linearly anneal beta from zero over the first training fraction."""
    if not 1 <= epoch <= num_epochs:
        raise ValueError("epoch must lie within the training run")
    anneal_epochs = max(1, int(np.ceil(num_epochs * anneal_fraction)))
    if anneal_epochs == 1:
        return float(beta_max)
    fraction = min(1.0, (epoch - 1) / (anneal_epochs - 1))
    return float(beta_max) * fraction


def _create_loaders(
    train_examples: np.ndarray,
    val_examples: np.ndarray,
    config: dict,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
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
    print("[WARN] Using a non-dropped training batch for this small data partition.")
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
        random_seed=int(config["split_seed"]),
    )
    del examples, sequence_indices

    run_stem = (
        f"{Path(config['data_path']).stem}_Tin{config['T_in']}_Tout{config['T_out']}"
        f"_splitseed{config['split_seed']}_cdmm_z{config['latent_dim']}"
    )
    processed_root = Path(config["data_path"]).parent
    stats_path = processed_root / f"train_norm_stats_{run_stem}.npy"
    val_path = processed_root / f"processed_val_{run_stem}.npy"
    sequence_path = processed_root / f"processed_val_seq_indices_{run_stem}.npy"
    train_examples, val_examples, _, _ = normalize_after_split_input_only(
        train_examples,
        val_examples,
        T_in=int(config["T_in"]),
        eps=1e-6,
        save_stats_path=str(stats_path),
    )
    np.save(val_path, val_examples.astype(np.float32, copy=False))
    np.save(sequence_path, val_sequence_indices.astype(np.int64, copy=False))

    config["normalization_stats_base"] = str(stats_path.with_suffix("").resolve())
    config["processed_val_examples_path"] = str(val_path.resolve())
    config["processed_val_seq_indices_path"] = str(sequence_path.resolve())
    config["data_layout"] = "NCT_on_disk_BTC_in_model"
    config["split_method"] = "sequence_id"
    config["normalization"] = "per_region_train_input_only_zscore"
    loaders = _create_loaders(
        train_examples, val_examples, config, num_workers
    )
    del train_examples, val_examples
    return loaders


def create_model_from_config(config: dict, device: torch.device):
    return create_cdmm_ssm(
        n_vars=int(config["n_vars"]),
        latent_dim=int(config["latent_dim"]),
        encoder_hidden_dim=int(config["encoder_hidden_dim"]),
        posterior_hidden_dim=int(config["posterior_hidden_dim"]),
        transition_hidden_dim=int(config["transition_hidden_dim"]),
        decoder_hidden_dims=tuple(int(value) for value in config["decoder_hidden_dims"]),
        context_length=int(config["T_in"]),
        forecast_length=int(config["T_out"]),
        spectral_radius=float(config["spectral_radius"]),
        min_scale=float(config["min_scale"]),
        min_log_variance=float(config["min_log_variance"]),
        max_log_variance=float(config["max_log_variance"]),
        device=device,
    )


def train_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    device: torch.device,
    *,
    beta: float,
    free_bits: float,
    accumulation_steps: int,
    gradient_clip: float,
    use_amp: bool,
    overshooting_horizons: tuple[int, ...],
    overshooting_weight: float,
    overshooting_num_anchors: int,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    totals = {
        "loss": 0.0,
        "reconstruction_nll": 0.0,
        "kl_raw": 0.0,
        "kl_free_bits": 0.0,
        "kl_per_latent_step": 0.0,
        "posterior_reconstruction_mse": 0.0,
        "posterior_reconstruction_mae": 0.0,
        "transition_std": 0.0,
        "emission_std": 0.0,
        "overshooting_kl_raw": 0.0,
        "overshooting_kl_free_bits": 0.0,
        "kl_regularizer_total": 0.0,
    }
    count = 0

    for batch_index, (context, target) in enumerate(
        tqdm(loader, desc="Training", leave=False)
    ):
        context = context.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            full_loss, diagnostics = model.conditional_elbo(
                context,
                target,
                beta=beta,
                free_bits=free_bits,
                overshooting_horizons=overshooting_horizons,
                overshooting_weight=overshooting_weight,
                overshooting_num_anchors=overshooting_num_anchors,
            )
            loss = full_loss / accumulation_steps
        scaler.scale(loss).backward()

        should_step = (batch_index + 1) % accumulation_steps == 0 or (
            batch_index + 1 == len(loader)
        )
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=gradient_clip
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        for key in totals:
            totals[key] += float(diagnostics[key].item())
        count += 1

    if count == 0:
        raise RuntimeError("Training loader produced no batches")
    return {key: value / count for key, value in totals.items()}


@torch.no_grad()
def validate_mean_forecast(
    model,
    loader,
    device: torch.device,
    use_amp: bool,
    *,
    num_samples: int,
    seed: int,
) -> dict[str, float]:
    model.eval()
    amp_enabled = bool(use_amp and device.type == "cuda")

    squared_error = 0.0
    absolute_error = 0.0
    zero_squared_error = 0.0
    persistence_squared_error = 0.0
    target_std_sum = 0.0
    forecast_std_sum = 0.0
    value_count = 0
    std_count = 0

    for batch_index, (context, target) in enumerate(
        tqdm(loader, desc="Validation", leave=False)
    ):
        context = context.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=amp_enabled,
        ):
            prediction = model.forecast_predictive_mean(
                context,
                horizon=target.shape[1],
                num_samples=num_samples,
                seed=seed + batch_index,
            )

        prediction = prediction.float()
        target = target.float()
        error = prediction - target

        squared_error += error.square().sum().item()
        absolute_error += error.abs().sum().item()
        zero_squared_error += target.square().sum().item()

        persistence = context[:, -1:, :].float().expand_as(target)
        persistence_squared_error += (
            persistence - target
        ).square().sum().item()

        target_std_sum += target.std(
            dim=1, unbiased=False
        ).sum().item()

        forecast_std_sum += prediction.std(
            dim=1, unbiased=False
        ).sum().item()

        value_count += target.numel()
        std_count += target.shape[0] * target.shape[2]

    if value_count == 0:
        raise RuntimeError("Validation loader produced no values")

    return {
        "mse": squared_error / value_count,
        "mae": absolute_error / value_count,
        "zero_mse": zero_squared_error / value_count,
        "persistence_mse": persistence_squared_error / value_count,
        "target_temporal_std": target_std_sum / std_count,
        "forecast_temporal_std": forecast_std_sum / std_count,
    }

def checkpoint_payload(
    model,
    optimizer,
    scheduler,
    config: dict,
    epoch: int,
    train_metrics: dict[str, float],
    validation_metrics: float,
    beta: float,
) -> dict:
    return {
        "epoch": int(epoch),
        "mode": "cDMM_SSM",
        "model_class": "ConditionalDeepMarkovSSM",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_metrics": dict(train_metrics),
        "validation_metrics": dict(validation_metrics),
        "validation_mean_forecast_mse":
            float(validation_metrics["mse"]),
        "beta": float(beta),
        "parameter_count": int(model.count_parameters()),
        "transition_spectral_norm": model.transition_spectral_norm(),
        "config": dict(config),
    }


def plot_history(history: list[dict], save_dir: Path) -> None:
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="ELBO loss")
    axes[0].plot(
        epochs,
        [row["train_reconstruction_nll"] for row in history],
        label="Reconstruction NLL",
    )
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(
        epochs,
        [row["validation_mean_forecast_mse"] for row in history],
        label="Validation mean MSE",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(save_dir / "learning_curves_detailed.png", dpi=300)
    plt.close(figure)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = build_config(args)
    device = resolve_device(args.device)
    workers = int(args.num_workers)
    if os.name == "nt" and workers > 0:
        print(f"[INFO] Windows: overriding num_workers {workers} -> 0")
        workers = 0
    set_seed(int(config["training_seed"]))
    save_dir = Path(config["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    overshooting_horizons = tuple(
        int(h) for h in config["overshooting_horizons"]
    )
    overshooting_weight = float(config["overshooting_weight"])
    overshooting_num_anchors = int(
        config["overshooting_num_anchors"]
    )
    
    print("=" * 80)
    print("TRAINING cDMM_SSM (CONDITIONAL ELBO; GENERATIVE-MEAN SELECTION)")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Latent dimension: {config['latent_dim']}")
    print(f"Data: {config['data_path']}")
    print(f"Checkpoints: {save_dir}")
    print(f"Split seed: {config['split_seed']}")
    print(f"Training seed: {config['training_seed']}")
    train_loader, val_loader = prepare_data(config, workers)
    # split_by_sequences resets NumPy's RNG; restore independent training state.
    set_seed(int(config["training_seed"]))
    model = create_model_from_config(config, device)
    config["parameter_count"] = model.count_parameters()
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Effective transition spectral norm: {model.transition_spectral_norm():.6f}")

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
    best_validation_mse = float("inf")
    best_epoch = 0
    no_improvement = 0
    last_payload = None
    for epoch in range(1, int(config["num_epochs"]) + 1):
        beta = beta_for_epoch(
            epoch,
            int(config["num_epochs"]),
            float(config["beta_max"]),
            float(config["beta_anneal_fraction"]),
        )
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            beta=beta,
            free_bits=float(config["free_bits"]),
            accumulation_steps=int(config["accumulation_steps"]),
            gradient_clip=float(config["gradient_clip"]),
            use_amp=bool(config["use_mixed_precision"]),
            overshooting_horizons=overshooting_horizons,
            overshooting_weight=overshooting_weight,
            overshooting_num_anchors=overshooting_num_anchors,
        )
        validation_metrics = validate_mean_forecast(
            model,
            val_loader,
            device,
            use_amp=bool(config["use_mixed_precision"]),
            num_samples=int(config["validation_num_samples"]),
            seed=int(config["validation_seed"]),
        )

        validation_mse = validation_metrics["mse"]
        row = {
            "epoch": epoch,
            "beta": beta,
            "train_loss": train_metrics["loss"],
            "train_reconstruction_nll":
                train_metrics["reconstruction_nll"],
            "train_kl_raw": train_metrics["kl_raw"],
            "train_kl_free_bits": train_metrics["kl_free_bits"],
            "train_kl_per_latent_step":
                train_metrics["kl_per_latent_step"],
            "train_posterior_reconstruction_mse":
                train_metrics["posterior_reconstruction_mse"],
            "train_posterior_reconstruction_mae":
                train_metrics["posterior_reconstruction_mae"],
            "train_transition_std":
                train_metrics["transition_std"],
            "train_emission_std":
                train_metrics["emission_std"],
            "validation_mean_forecast_mse":
                validation_metrics["mse"],
            "validation_mean_forecast_mae":
                validation_metrics["mae"],
            "validation_zero_mse":
                validation_metrics["zero_mse"],
            "validation_persistence_mse":
                validation_metrics["persistence_mse"],
            "validation_target_temporal_std":
                validation_metrics["target_temporal_std"],
            "validation_forecast_temporal_std":
                validation_metrics["forecast_temporal_std"],
            "train_overshooting_kl_raw":
                train_metrics["overshooting_kl_raw"],
            "train_overshooting_kl_free_bits":
                train_metrics["overshooting_kl_free_bits"],
            "train_kl_regularizer_total":
                train_metrics["kl_regularizer_total"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{config['num_epochs']} | "
            f"beta {beta:.4f} | "
            f"ELBO {train_metrics['loss']:.3f} | "
            f"Recon NLL {train_metrics['reconstruction_nll']:.3f} | "
            f"KL/step {train_metrics['kl_per_latent_step']:.4f} | "
            f"Post MSE {train_metrics['posterior_reconstruction_mse']:.4f} | "
            f"Val MSE {validation_metrics['mse']:.4f} | "
            f"Val MAE {validation_metrics['mae']:.4f} | "
            f"Persistence {validation_metrics['persistence_mse']:.4f} | "
            f"Forecast std {validation_metrics['forecast_temporal_std']:.4f}"
            f"OverKL {train_metrics['overshooting_kl_free_bits']:.4f} | "
        )

        last_payload = checkpoint_payload(
            model,
            optimizer,
            scheduler,
            config,
            epoch,
            train_metrics,
            validation_metrics,
            beta,
        )
        if epoch == int(config["checkpoint_first_epoch"]) or (
            int(config["checkpoint_every"]) > 0
            and epoch % int(config["checkpoint_every"]) == 0
        ):
            torch.save(last_payload, save_dir / f"checkpoint_fixed_epoch_{epoch}.pt")

        if validation_mse < best_validation_mse - float(
            config["early_stop_min_delta"]
        ):
            best_validation_mse = validation_mse
            best_epoch = epoch
            no_improvement = 0
            torch.save(last_payload, save_dir / "best_model.pt")
        else:
            no_improvement += 1
        if no_improvement >= int(config["early_stop_patience"]):
            print(f"Early stopping: best mean-forecast MSE at epoch {best_epoch}.")
            break

    if last_payload is None:
        raise RuntimeError("No training epoch completed")
    last_payload["history"] = history
    last_payload["best_epoch"] = best_epoch
    last_payload["best_validation_mean_forecast_mse"] = best_validation_mse
    torch.save(last_payload, save_dir / "final_model.pt")
    (save_dir / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    plot_history(history, save_dir)
    print(
        f"Best context-only generative mean MSE: {best_validation_mse:.6f} "
        f"(epoch {best_epoch})"
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
