"""
Training script for Decoder-Only Transformer on multi-step time series prediction.

Scaling-law mode: by default, all .json / .yaml / .yml files under the repo ``configs/`` directory
are used as run configs (or pass explicit paths). Each run file must define T_in, data_path,
random_seed, num_epochs, n_vars. Other hyperparameters come from the shared base config unless
overridden in the same file.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F

_SRC_ROOT = Path(__file__).resolve().parent.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from helpers.preprocess_helpers import (
    load_data,
    reshape_to_examples,
    split_by_sequences,
    create_dataloaders,
    filter_examples_by_nan,
    normalize_after_split_input_only,
    verify_data_loading,
)
from helpers.scaling_law_globals import (
    load_scaling_law_globals,
    merge_paths_section,
    train_base_config_from_globals,
)
#from models.model_KV_cached import create_model_cached
from models.models import create_model

_SCALING_LAW_GLOBALS_PATH: Path | None = None


def set_scaling_law_globals_path(path: Path | None) -> None:
    global _SCALING_LAW_GLOBALS_PATH
    _SCALING_LAW_GLOBALS_PATH = Path(path) if path is not None else None


# =========================
# 1) CombinedLoss: compute components even when weights are 0 (for logging)
# =========================
class CombinedLoss(nn.Module):
    def __init__(
        self,
        mae_weight=1.0,
        shape_weight=0.0,
        deriv_weight=0.0,
        cross_weight=0.0,
        var_weight=0.0,
        log_all_terms: bool = True,   # <-- NEW
    ):
        super().__init__()
        self.weights = {
            'mae': mae_weight,
            'shape': shape_weight,
            'deriv': deriv_weight,
            'cross': cross_weight,
            'var': var_weight,
        }
        self.log_all_terms = log_all_terms  # <-- NEW
        self.mae = nn.L1Loss()

        self.register_buffer('running_norms', torch.ones(5))
        self.initialized = False

    def forward(self, predictions, targets):
        B, T, V = predictions.shape
        device = predictions.device
        dtype = predictions.dtype
        eps = 1e-6

        mae_raw = self.mae(predictions, targets)

        # Default zeros
        shape_raw = torch.zeros((), device=device, dtype=dtype)
        deriv_raw = torch.zeros((), device=device, dtype=dtype)
        cross_raw = torch.zeros((), device=device, dtype=dtype)
        var_raw   = torch.zeros((), device=device, dtype=dtype)

        # Compute if weight>0 OR if we want it for logging
        need_shape = (self.weights['shape'] != 0) or self.log_all_terms
        need_deriv = (self.weights['deriv'] != 0) or self.log_all_terms
        need_cross = (self.weights['cross'] != 0) or self.log_all_terms
        need_var   = (self.weights['var']   != 0) or self.log_all_terms

        if need_shape:
            pred_norm = predictions - predictions.mean(dim=1, keepdim=True)
            targ_norm = targets - targets.mean(dim=1, keepdim=True)
            corr = (pred_norm * targ_norm).sum(dim=1) / (
                torch.sqrt((pred_norm**2).sum(dim=1) + 1e-8) *
                torch.sqrt((targ_norm**2).sum(dim=1) + 1e-8)
            )
            shape_raw = 1 - corr.mean()

        if need_deriv:
            d_pred = predictions[:, 1:, :] - predictions[:, :-1, :]
            d_targ = targets[:, 1:, :] - targets[:, :-1, :]
            deriv_raw = torch.mean(torch.abs(d_pred - d_targ))

        if need_cross:
            pred_cov = torch.bmm(predictions.transpose(1, 2), predictions)
            targ_cov = torch.bmm(targets.transpose(1, 2), targets)
            cross_raw = self.mae(pred_cov, targ_cov)

        if need_var:
            pred_ctr = predictions - predictions.mean(dim=1, keepdim=True)
            targ_ctr = targets     - targets.mean(dim=1, keepdim=True)

            bins = [0, T//3, 2*T//3, T]
            var_terms = []
            for a, b in zip(bins[:-1], bins[1:]):
                ps = pred_ctr[:, a:b, :].std(dim=1, unbiased=False)
                ts = targ_ctr[:, a:b, :].std(dim=1, unbiased=False)

                log_ratio = torch.log(ps + eps) - torch.log(ts + eps)
                under = F.relu(-log_ratio)
                over  = F.relu(log_ratio)
                var_terms.append(under.mean() + 0.25 * over.mean())

            var_raw = torch.stack(var_terms).mean()


        # Init norms (safe even if some terms not computed)
        if not self.initialized and self.training:
            self.running_norms[0] = mae_raw.detach() + 1e-8
            self.running_norms[1] = (shape_raw.detach() + 1e-8) if need_shape else torch.tensor(1.0, device=device)
            self.running_norms[2] = (deriv_raw.detach() + 1e-8) if need_deriv else torch.tensor(1.0, device=device)
            self.running_norms[3] = (cross_raw.detach() + 1e-8) if need_cross else torch.tensor(1.0, device=device)
            self.running_norms[4] = (var_raw.detach()   + 1e-8) if need_var   else torch.tensor(1.0, device=device)
            self.initialized = True

        total = self.weights['mae'] * (mae_raw / self.running_norms[0])
        if self.weights['shape'] != 0: total = total + self.weights['shape'] * (shape_raw / self.running_norms[1])
        if self.weights['deriv'] != 0: total = total + self.weights['deriv'] * (deriv_raw / self.running_norms[2])
        if self.weights['cross'] != 0: total = total + self.weights['cross'] * (cross_raw / self.running_norms[3])
        if self.weights['var']   != 0: total = total + self.weights['var']   * (var_raw   / self.running_norms[4])

        loss_dict_norm = {
            'mae':   (mae_raw   / self.running_norms[0]),
            'shape': (shape_raw / self.running_norms[1]) if need_shape else torch.zeros((), device=device),
            'deriv': (deriv_raw / self.running_norms[2]) if need_deriv else torch.zeros((), device=device),
            'cross': (cross_raw / self.running_norms[3]) if need_cross else torch.zeros((), device=device),
            'var':   (var_raw   / self.running_norms[4]) if need_var   else torch.zeros((), device=device),
            'total': total
        }
        loss_dict_raw = {
            'mae':   mae_raw,
            'shape': (shape_raw) if need_shape else torch.zeros((), device=device),
            'deriv': (deriv_raw) if need_deriv else torch.zeros((), device=device),
            'cross': (cross_raw) if need_cross else torch.zeros((), device=device),
            'var':   (var_raw) if need_var   else torch.zeros((), device=device),
            'total': total
        }

        return total, loss_dict_norm, loss_dict_raw


def build_loss(config):
    if config['loss_type'] == 'combined':
        return CombinedLoss(
            mae_weight=config.get('loss_mae_weight', 1.0),
            shape_weight=config.get('loss_shape_weight', 0.0),
            deriv_weight=config.get('loss_deriv_weight', 0.0),
            cross_weight=config.get('loss_cross_weight', 0.0),
            var_weight=config.get('loss_var_weight', 0.0),
            log_all_terms=config.get('log_all_loss_terms', True),  
        )


# --- Scaling-law: per-run config files (YAML / JSON) ---------------------------------

REQUIRED_SCALING_RUN_KEYS = frozenset(
    {"T_in", "data_path", "random_seed", "num_epochs", "n_vars"}
)


def _fallback_training_base_config() -> dict:
    """Used only if ``scaling_law_globals.json`` has no ``train_scaling_law.base_config``."""
    return {
        "data_path": "data_processed/data25_ba2.npy",
        "T_in": 30,
        "T_out": 10,
        "n_vars": 2,
        "d_model": 64,
        "n_heads": 8,
        "n_layers": 8,
        "d_ff": 128,
        "dropout": 0.05,
        "patch_len": 1,
        "effective_batch_size": 100,
        "batch_size": 100,
        "accumulation_steps": 1,
        "use_mixed_precision": True,
        "loss_type": "combined",
        "loss_mae_weight": 1.0,
        "loss_shape_weight": 0.0,
        "loss_deriv_weight": 0.0,
        "loss_cross_weight": 0.0,
        "loss_var_weight": 0.0,
        "learning_rate": 1e-4,
        "max_lr": 0.0003,
        "num_epochs": 1,
        "train_ratio": 0.8,
        "random_seed": 101,
        "save_dir": "checkpoints",
        "checkpoint_flat_dir": "checkpoints",
        "plot_dir": "evaluation",
        "log_all_loss_terms": True,
        "checkpoint_first_epoch": 400,
        "checkpoint_every": 400,
        "save_every": 400,
        "early_stop_patience": 400,
        "early_stop_min_delta": 1e-6,
        "log_per_region": False,
        "scheduler_pct_start": 0.3,
        "scheduler_div_factor": 25,
        "scheduler_final_div_factor": 100,
        "scheduler_anneal_strategy": "cos",
        "use_torch_compile": True,
        "save_processed_split_examples": True,
    }


def get_default_base_config() -> dict:
    """Shared hyperparameters from ``scaling_law_globals.json`` with Python fallback."""
    full = load_scaling_law_globals(_SCALING_LAW_GLOBALS_PATH)
    loaded = train_base_config_from_globals(full)
    return loaded if loaded is not None else _fallback_training_base_config()


def load_run_config_file(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(raw)
    else:
        raise ValueError(f"Unsupported config extension {path.suffix!r} (use .json)")

    if data is None:
        raise ValueError(f"Config file is empty: {path}")
    if not isinstance(data, dict):
        raise TypeError(f"Config root must be a mapping, got {type(data).__name__} in {path}")
    return data


def validate_scaling_run_dict(overrides: dict, path: Path) -> None:
    missing = REQUIRED_SCALING_RUN_KEYS - set(overrides.keys())
    if missing:
        raise ValueError(
            f"{path}: missing required keys {sorted(missing)}. "
            f"Required: {sorted(REQUIRED_SCALING_RUN_KEYS)}"
        )


def merge_scaling_config(
    base: dict,
    overrides: dict,
    *,
    run_label: str | None = None,
) -> dict:
    """Deep copy base, apply overrides. If save_dir not in overrides, suffix with run_label."""
    merged = copy.deepcopy(base)
    merged.update(overrides)
    if run_label is not None and "save_dir" not in overrides:
        merged["save_dir"] = f"{base['save_dir']}_{run_label}"
    return merged


def scaling_law_project_root() -> Path:
    """Repository root (parent of ``src/``); this file lives under ``src/train/``."""
    return Path(__file__).resolve().parents[2]


def train_norm_stats_npy_path(config_json_path: Path) -> Path:
    """
    Filesystem-safe path for train normalization stats tied to one run config file.
    Writes ``train_norm_stats_{config_stem}.npy`` (mean/std use the same base; see preprocess_helpers).
    """
    stem = Path(config_json_path).stem
    if not stem:
        stem = "run"
    return scaling_law_project_root() / "data_processed" / f"train_norm_stats_{stem}.npy"


def processed_split_examples_paths(config_json_path: Path) -> tuple[Path, Path, Path, Path]:
    """
    Paths for normalized train/val arrays, val sequence-id vector, and JSON sidecar.

    Returns (train_npy, val_npy, val_seq_indices_npy, meta_json).
    """
    stem = Path(config_json_path).stem
    if not stem:
        stem = "run"
    root = scaling_law_project_root() / "data_processed"
    train_p = root / f"processed_train_{stem}.npy"
    val_p = root / f"processed_val_{stem}.npy"
    val_seq_p = root / f"processed_val_seq_indices_{stem}.npy"
    meta_p = root / f"processed_split_{stem}_meta.json"
    return train_p, val_p, val_seq_p, meta_p


def save_processed_split_for_inference(
    config_json_path: Path,
    train_examples: np.ndarray,
    val_examples: np.ndarray,
    val_seq_indices: np.ndarray,
    config: dict,
) -> None:
    """Persist normalized train/val splits and val row→logical-sequence mapping for long-horizon inference."""
    train_p, val_p, val_seq_p, meta_p = processed_split_examples_paths(config_json_path)
    train_p.parent.mkdir(parents=True, exist_ok=True)

    np.save(train_p, train_examples)
    np.save(val_p, val_examples)
    np.save(val_seq_p, val_seq_indices.astype(np.int64, copy=False))

    meta = {
        "layout": "NCT",
        "train_shape": list(train_examples.shape),
        "val_shape": list(val_examples.shape),
        "T_in": int(config["T_in"]),
        "T_out": int(config["T_out"]),
        "train_ratio": float(config["train_ratio"]),
        "split_random_seed": int(config["random_seed"]),
        "source_data_path": config["data_path"],
        "normalization_stats_base": config.get("normalization_stats_base"),
        "processed_train_examples_path": str(train_p.resolve()),
        "processed_val_examples_path": str(val_p.resolve()),
        "processed_val_seq_indices_path": str(val_seq_p.resolve()),
        "note": "val_seq_indices[i] is the logical sequence id for processed_val row i; "
        "concatenate all val rows with the same id in row order to rebuild full sequences.",
    }
    meta_p.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    config["processed_train_examples_path"] = meta["processed_train_examples_path"]
    config["processed_val_examples_path"] = meta["processed_val_examples_path"]
    config["processed_val_seq_indices_path"] = meta["processed_val_seq_indices_path"]
    config["processed_split_meta_path"] = str(meta_p.resolve())

    print("Saved processed splits (normalized, same arrays as DataLoaders):")
    print(f"  Train: {train_p}")
    print(f"  Val:   {val_p}")
    print(f"  Val seq ids: {val_seq_p}")
    print(f"  Meta:  {meta_p}")


def default_scaling_configs_dir() -> Path:
    full = load_scaling_law_globals(_SCALING_LAW_GLOBALS_PATH)
    paths = merge_paths_section(full)
    return scaling_law_project_root() / paths["configs"]


def discover_run_configs_in_dir(dir_path: Path) -> list[Path]:
    """Sorted list of *.json, *.yaml, *.yml files in ``dir_path`` (non-recursive)."""
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        return []
    found: list[Path] = []
    for pattern in ("*.json", "*.yaml", "*.yml"):
        found.extend(p for p in dir_path.glob(pattern) if p.is_file())
    return sorted(found, key=lambda p: p.name.lower())


def scaling_checkpoint_milestones(num_epochs: int) -> list[int]:
    """Epochs (1-based) at 0.1*n, 0.3*n, and n; deduplicated, in order."""
    n = int(num_epochs)
    if n < 1:
        return []
    e1 = max(1, min(n, int(round(0.1 * n))))
    e2 = max(1, min(n, int(round(0.3 * n))))
    e3 = n
    out: list[int] = []
    seen: set[int] = set()
    for e in (e1, e2, e3):
        if e not in seen:
            out.append(e)
            seen.add(e)
    return out


def build_scaling_checkpoint_filename(mode_tag: str, config: dict, epoch: int) -> str:
    """
    e.g. checkpoints_AR_KV_data50_ba4_Tin30_seed101_epoch3.pt
    """
    stem = Path(config["data_path"]).stem
    tin = int(config["T_in"])
    seed = int(config["random_seed"])
    return f"checkpoints_{mode_tag}_{stem}_Tin{tin}_seed{seed}_epoch{epoch}.pt"


def save_scaling_checkpoint_pt(
    model,
    optimizer,
    scheduler,
    config: dict,
    mode_tag: str,
    mode_label: str,
    epoch: int,
    train_loss: float,
    val_loss: float,
    out_dir: Path,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = build_scaling_checkpoint_filename(mode_tag, config, epoch)
    path = out_dir / name
    to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save(
        {
            "epoch": epoch,
            "mode": mode_tag,
            "mode_label": mode_label,
            "model_state_dict": to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "config": config,
        },
        path,
    )
    return path


def run_model_forward(model, inputs, targets, current_p, mode):

    mode = mode.upper()

    if mode == "AR_KV":
        if not hasattr(model, "forward_autoregressive_kvcache"):
            raise AttributeError("Model has no forward_autoregressive_kvcache()")
        return model.forward_autoregressive_kvcache(inputs)

    if mode == "TF":
        return model.forward_teacher_forcing(inputs, targets)

    raise ValueError(f"Unknown forward mode: {mode} (expected 'TL', 'AR', 'TF)")



def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, accumulation_steps=1, use_amp=True, forward_mode="TL", current_p=0.5):
    """
    Train for one epoch with gradient accumulation and mixed precision.
    
    Args:
        accumulation_steps: Number of batches to accumulate gradients over
        use_amp: Use automatic mixed precision (fp16) to reduce memory
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    component_totals_norm = {}
    component_totals_raw = {}

    scaler = torch.cuda.amp.GradScaler() if (use_amp and torch.cuda.is_available()) else None
    optimizer.zero_grad()


    for batch_idx, (input_batch, target_batch) in enumerate(tqdm(train_loader, desc="Training")):

        input_batch = input_batch.to(device, non_blocking=True)
        target_batch = target_batch.to(device, non_blocking=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                predictions = run_model_forward(model, input_batch, target_batch, current_p, forward_mode)
                loss, loss_dict_norm, loss_dict_raw = criterion(predictions, target_batch)
                loss = loss / accumulation_steps
        else:
            predictions = run_model_forward(model, input_batch, target_batch, current_p, forward_mode)
            loss, loss_dict_norm, loss_dict_raw = criterion(predictions, target_batch)
            loss = loss / accumulation_steps

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # ---- BACKWARD ----
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # aggregate BOTH
        for k, v in loss_dict_norm.items():
            component_totals_norm[k] = component_totals_norm.get(k, 0.0) + float(v.item())
        for k, v in loss_dict_raw.items():
            component_totals_raw[k] = component_totals_raw.get(k, 0.0) + float(v.item())

        del predictions

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

        total_loss += loss.item() * accumulation_steps
        n_batches += 1
    

    avg_main_loss = total_loss / n_batches
    avg_norm = {k: v / n_batches for k, v in component_totals_norm.items()}
    avg_raw  = {k: v / n_batches for k, v in component_totals_raw.items()}
    return avg_main_loss, avg_norm, avg_raw




def validate(model, val_loader, criterion, device, compute_per_region=False, n_vars=16, use_amp=True,  forward_mode="TL", current_p=0.5):
    """
    Validate the model.
    
    Args:
        model: Model to validate
        val_loader: Validation dataloader
        criterion: Loss function
        device: Device to run on
        compute_per_region: If True, compute per-region losses
        n_vars: Number of variables/regions
        use_amp: Use mixed precision
        
    Returns:
        avg_loss: Average validation loss
        per_region_losses: Dict of per-region losses (if compute_per_region=True)
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    component_totals_norm = {}
    component_totals_raw = {}

    if compute_per_region:
        per_region_losses = torch.zeros(n_vars, device=device)
        per_region_counts = torch.zeros(n_vars, device=device)

    with torch.no_grad():
        for input_batch, target_batch in tqdm(val_loader, desc="Validation"):
            input_batch = input_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)

            if use_amp and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    predictions = run_model_forward(model, input_batch, target_batch, current_p, forward_mode)
                    loss, loss_dict_norm, loss_dict_raw = criterion(predictions, target_batch)
            else:
                predictions = run_model_forward(model, input_batch, target_batch, current_p, forward_mode)
                loss, loss_dict_norm, loss_dict_raw = criterion(predictions, target_batch)

            total_loss += float(loss.item())

            for k, v in loss_dict_norm.items():
                component_totals_norm[k] = component_totals_norm.get(k, 0.0) + float(v.item())
            for k, v in loss_dict_raw.items():
                component_totals_raw[k] = component_totals_raw.get(k, 0.0) + float(v.item())

            if compute_per_region:
                region_losses = torch.mean(torch.abs(predictions - target_batch), dim=(0, 1))
                per_region_losses += region_losses
                per_region_counts += 1

            n_batches += 1
            del input_batch, target_batch, predictions
            """if n_batches % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()"""

    avg_loss = total_loss / n_batches
    avg_norm = {k: v / n_batches for k, v in component_totals_norm.items()}
    avg_raw  = {k: v / n_batches for k, v in component_totals_raw.items()}

    if compute_per_region:
        per_region_losses = (per_region_losses / per_region_counts).cpu().numpy()
        return avg_loss, avg_norm, avg_raw, per_region_losses
    else:
        return avg_loss, avg_norm, avg_raw, None


def train_single_mode(base_config, train_loader, val_loader, device, mode_tag, mode_label):
    config = dict(base_config)
    ckpt_dir = Path(config.get("checkpoint_flat_dir", "checkpoints"))
    milestones = scaling_checkpoint_milestones(config["num_epochs"])
    saved_checkpoints: list[Path] = []
    print(f"[{mode_tag}] Milestone checkpoints (epochs): {milestones}")

    """model = create_model_cached(
        n_vars=config['n_vars'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        T_in=config['T_in'],
        T_out=config['T_out'],
        device=device
    )"""
    model = create_model(
        n_vars=config['n_vars'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        T_in=config['T_in'],
        T_out=config['T_out'],
        device=device
    )

    criterion = build_loss(config)
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])

    total_batches = len(train_loader)
    total_steps = (total_batches + config['accumulation_steps'] - 1) // config['accumulation_steps']
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['max_lr'],
        epochs=config['num_epochs'],
        steps_per_epoch=total_steps,
        pct_start=config['scheduler_pct_start'],
        div_factor=config['scheduler_div_factor'],
        final_div_factor=config['scheduler_final_div_factor'],
        anneal_strategy=config['scheduler_anneal_strategy'],
        three_phase=False,
    )

    best_val_loss = float('inf')
    best_epoch = 0

    no_improve_count = 0
    early_stop_patience = config.get('early_stop_patience', 10)
    early_stop_min_delta = config.get('early_stop_min_delta', 1e-6)
    compute_per_region = config.get('log_per_region', True)

    for epoch in range(1, config['num_epochs'] + 1):
        current_p = 0.5
        criterion.weights['mae'] = 1.0

        print(f"[{mode_tag}] Epoch={epoch}/{config['num_epochs']}")
        train_loss, train_comp, train_comp_raw = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device,
            accumulation_steps=config['accumulation_steps'],
            use_amp=config.get('use_mixed_precision', True),
            forward_mode=mode_tag, current_p=current_p,
        )

        if mode_tag == "TF":
            val_loss, val_comp, val_comp_raw, per_region_losses = validate(
                model, val_loader, criterion, device,
                compute_per_region=compute_per_region,
                n_vars=config['n_vars'],
                use_amp=config.get('use_mixed_precision', True),
                forward_mode="TF", current_p=current_p,
            )
        else:
            # ---- Primary validation (matches training regime)
            val_loss, val_comp, val_comp_raw, per_region_losses = validate(
                model, val_loader, criterion, device,
                compute_per_region=compute_per_region,
                n_vars=config['n_vars'],
                use_amp=config.get('use_mixed_precision', True),
                forward_mode=mode_tag, current_p=current_p,
            )


        # Console prints
        print(
            f"  Train: Total {train_loss:.4f} | MAE {train_comp['mae']:.4f} | Shape {train_comp['shape']:.4f} "
            f"| Deriv {train_comp['deriv']:.4f} | Cross {train_comp['cross']:.4f} | Var {train_comp['var']:.4f}"
        )
        print(
            f"  Val({mode_tag}): Total {val_loss:.4f} | MAE {val_comp['mae']:.4f} | Shape {val_comp['shape']:.4f} "
            f"| Deriv {val_comp['deriv']:.4f} | Cross {val_comp['cross']:.4f} | Var {val_comp['var']:.4f}"
        )

        if epoch in milestones:
            path = save_scaling_checkpoint_pt(
                model,
                optimizer,
                scheduler,
                config,
                mode_tag,
                mode_label,
                epoch,
                float(train_loss),
                float(val_loss),
                ckpt_dir,
            )
            saved_checkpoints.append(path)
            print(f"  [Saved checkpoint] {path.name}")

        # Best model selection (metrics only; checkpoints are milestone-based)
        if val_loss < best_val_loss - early_stop_min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= early_stop_patience:
            print(f"  [{mode_tag}] Early stopping triggered after {no_improve_count} epochs without improvement.")
            break

    print(f"\n[{mode_tag}] Training complete! Best val loss {best_val_loss:.6f} (epoch {best_epoch})")

    return {
        'mode': mode_tag,
        'label': mode_label,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'checkpoint_dir': ckpt_dir,
        'checkpoint_paths': saved_checkpoints,
    }


def _print_training_config_summary(config: dict) -> None:
    print("\nConfiguration:")
    print("  Model:")
    print(f"    d_model: {config['d_model']}")
    print(f"    n_heads: {config['n_heads']}")
    print(f"    n_layers: {config['n_layers']}")
    print(f"    d_ff (d_hid): {config['d_ff']}")
    print(f"    dropout: {config['dropout']}")
    print(f"    patch_len: {config['patch_len']}")
    print("  Run-specific:")
    print(f"    data_path: {config['data_path']}")
    print(f"    T_in: {config['T_in']}  T_out: {config['T_out']}  n_vars: {config['n_vars']}")
    print(f"    num_epochs: {config['num_epochs']}  random_seed: {config['random_seed']}")
    print(f"    save_dir: {config['save_dir']}")
    print(f"    checkpoint_flat_dir: {config.get('checkpoint_flat_dir', 'checkpoints')}")
    print(f"    save_processed_split_examples: {config.get('save_processed_split_examples', True)}")
    print("  Training:")
    print(f"    effective_batch_size: {config['effective_batch_size']} (via gradient accumulation)")
    print(f"    physical_batch_size: {config['batch_size']}")
    print(f"    accumulation_steps: {config['accumulation_steps']}")
    print(f"    mixed_precision (FP16): {config.get('use_mixed_precision', False)}")
    print(f"    optimizer: AdamW")
    print(f"    scheduler: OneCycleLR (max_lr={config['max_lr']})")
    if config.get("loss_type", "l1") == "combined":
        loss_desc = (
            f"CombinedLoss(MAE={config.get('loss_mae_weight', 1.0)}, "
            f"MSE={config.get('loss_mse_weight', 0.0)}, "
            f"Corr={config.get('loss_corr_weight', 0.0)}, "
            f"Deriv={config.get('loss_deriv_weight', 0.0)}, "
            f"AutoCorr={config.get('loss_autocorr_weight', 0.0)}, "
            f"CrossReg={config.get('loss_cross_region_weight', 0.0)}, "
            f"Dist={config.get('loss_distribution_weight', 0.0)})"
        )
    else:
        loss_desc = config.get("loss_type", "l1").upper()
    print(f"    criterion: {loss_desc}")
    print(f"    early_stopping: {config['early_stop_patience']} epochs")
    print(f"    save_every: {config['save_every']} epochs")


def train_single_scaling_run(
    path: Path,
    config: dict,
    device: torch.device | None = None,
    *,
    training_variants: list[dict] | None = None,
) -> list[dict]:
    """
    One full training run: set seeds, build dataloaders from config, train each variant.

    `config` must be a fully merged dict (base + per-run overrides).
    """
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _print_training_config_summary(config)

    torch.manual_seed(int(config["random_seed"]))
    np.random.seed(int(config["random_seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config["random_seed"]))

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    print("\n" + "=" * 80)
    print("PREPARING DATA")
    print("=" * 80)
    data_array = load_data(config["data_path"])
    examples, sequence_indices = reshape_to_examples(data_array)
    del data_array
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    examples, sequence_indices = filter_examples_by_nan(
        examples,
        sequence_indices,
        T_in=config["T_in"],
        T_out=config["T_out"],
    )

    train_examples, val_examples, _train_seq_indices, val_seq_indices = split_by_sequences(
        examples,
        sequence_indices,
        train_ratio=config["train_ratio"],
        random_seed=config["random_seed"],
    )

    stats_npy = train_norm_stats_npy_path(path)
    # Base path without ``.npy`` — mean/std/meta are ``{base}_mean.npy``, etc. (see preprocess_helpers).
    config["normalization_stats_base"] = str(stats_npy.with_suffix(""))

    train_examples, val_examples, _test_examples, _norm = normalize_after_split_input_only(
        train_examples,
        val_examples,
        T_in=config["T_in"],
        eps=1e-6,
        save_stats_path=str(stats_npy),
    )

    if config.get("save_processed_split_examples", True):
        save_processed_split_for_inference(
            path, train_examples, val_examples, val_seq_indices, config
        )

    del examples, sequence_indices
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_loader, val_loader = create_dataloaders(
        train_examples,
        val_examples,
        T_in=config["T_in"],
        T_out=config["T_out"],
        batch_size=config["batch_size"],
        num_workers=8,
    )

    if training_variants is None:
        training_variants = [
            #{"tag": "AR_KV", "label": "KV Autoregressive"},
            {"tag": "TF", "label": "Teacher Forced"},
        ]

    histories = []
    for variant in training_variants:
        history = train_single_mode(
            config,
            train_loader,
            val_loader,
            device,
            mode_tag=variant["tag"],
            mode_label=variant["label"],
        )
        histories.append(history)

    return histories


def run_scaling_law_from_config_files(
    config_paths: list[Path],
    *,
    base_config: dict | None = None,
    training_variants: list[dict] | None = None,
) -> list[dict]:
    """
    Train one model (per file × training variant) for each YAML/JSON run config.
    Each file must define: T_in, data_path, random_seed, num_epochs, n_vars.
    """
    base = copy.deepcopy(base_config) if base_config is not None else get_default_base_config()
    results = []
    for path in config_paths:
        path = Path(path)
        overrides = load_run_config_file(path)
        validate_scaling_run_dict(overrides, path)
        merged = merge_scaling_config(base, overrides, run_label=path.stem)
        print("\n" + "=" * 80)
        print(f"SCALING RUN: {path}")
        print("=" * 80)
        histories = train_single_scaling_run(
            path, merged, training_variants=training_variants
        )
        results.append(
            {
                "config_path": str(path.resolve()),
                "histories": histories,
            }
        )
    return results


def main(argv: list[str] | None = None) -> None:
    """CLI: default is all run configs in ``configs/``; optional explicit paths or --no-configs-dir."""
    parser = argparse.ArgumentParser(
        description="Train AR_KV model(s). Defaults to every .json/.yaml/.yml in configs/.",
    )
    parser.add_argument(
        "configs",
        nargs="*",
        type=str,
        help="Optional explicit per-run config paths (overrides configs-dir scan)",
    )
    parser.add_argument(
        "--scaling-law-globals",
        type=Path,
        default=None,
        help="Path to scaling_law_globals.json (default: <repo>/scaling_law_globals.json)",
    )
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=None,
        help="Directory to scan for run configs (default: from scaling_law_globals.json paths.configs)",
    )
    parser.add_argument(
        "--no-configs-dir",
        action="store_true",
        help="Do not scan configs/; run a single training with the default base only (no run JSON)",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=None,
        help="Optional JSON/YAML merged on top of the built-in default base (shared hyperparameters)",
    )
    argv = argv if argv is not None else sys.argv[1:]
    args = parser.parse_args(argv)

    set_scaling_law_globals_path(args.scaling_law_globals)

    base = get_default_base_config()
    if args.base_config is not None:
        extra = load_run_config_file(args.base_config)
        base.update(extra)

    import gc

    print("=" * 80)
    print("TRAINING SIMPLE DECODER-ONLY TRANSFORMER (scaling law)")
    print("=" * 80)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("\n[INFO] Cleared GPU cache before starting")
        print(f"[INFO] GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"[INFO] GPU Memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

    if args.no_configs_dir:
        train_single_scaling_run(Path("base"), base)
        return

    if args.configs:
        config_paths = [Path(p) for p in args.configs]
    else:
        cfg_dir = args.configs_dir if args.configs_dir is not None else default_scaling_configs_dir()
        config_paths = discover_run_configs_in_dir(cfg_dir)
        if not config_paths:
            print(
                f"\n[ERROR] No run configs found in {cfg_dir.resolve()} "
                f"(expected .json / .yaml / .yml). Add files under configs/ or pass paths on the CLI."
            )
            sys.exit(1)
        print(f"\nUsing {len(config_paths)} run config(s) from {cfg_dir.resolve()}:")
        for p in config_paths:
            print(f"  - {p}")

    run_scaling_law_from_config_files(config_paths, base_config=base)


if __name__ == "__main__":
    import gc
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        raise
    finally:
        # Clean up memory
        print("\n[INFO] Final memory cleanup...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("[INFO] GPU cache cleared")