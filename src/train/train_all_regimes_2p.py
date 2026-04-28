"""
Training script for Decoder-Only Transformer on multi-step time series prediction.
"""

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
import sys
import math
import os
import json
import argparse


_SRC_ROOT = Path(__file__).resolve().parent.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from helpers.preprocess_helpers import (
    load_data,
    reshape_to_examples,
    create_dataloaders,
    filter_examples_by_nan,
    verify_data_loading,
    compute_train_stats_input_only,
    apply_normalization_nct,
)
from models.model_KV_cached_2p import create_model_cached


def apply_robust_std_floor_numpy(
    std: np.ndarray,
    eps: float,
    floor_abs: float,
    floor_frac_median: float,
) -> tuple[np.ndarray, float, float]:
    """
    Raise near-zero per-channel stds so z-scoring does not explode on silent neurons.

    floor = max(eps, floor_abs, floor_frac_median * median(per-channel std))
    """
    flat = std.astype(np.float64).reshape(-1)
    med = float(np.median(flat))
    floor = max(float(eps), float(floor_abs), float(floor_frac_median) * med)
    out = np.maximum(std.astype(np.float64), floor).astype(np.float32)
    return out, floor, med


def normalize_split_robust_2p(
    train_examples: np.ndarray,
    val_examples: np.ndarray,
    test_examples: np.ndarray | None,
    T_in: int,
    eps: float,
    floor_abs: float,
    floor_frac_median: float,
    normalization_mode: str = "robust_zscore",
    save_stats_path: str | None = None,
):
    """
    Train-only input-window mean/std (same as preprocess_helpers), then robust std floor, then z-score.
    Saves *_mean.npy / *_std.npy / *_meta.json compatible with inference loaders.
    """
    mode = str(normalization_mode).lower()
    if mode == "none":
        print("[norm 2p] normalization_mode=none -> using raw continuous traces")
        return train_examples, val_examples, test_examples, (None, None, "none")

    mean, std = compute_train_stats_input_only(train_examples, T_in=T_in, eps=eps)
    std, floor_used, med_before = apply_robust_std_floor_numpy(
        std, eps=eps, floor_abs=floor_abs, floor_frac_median=floor_frac_median
    )
    print(
        f"[norm 2p] median std (train input window, pre-floor): {med_before:.6g} | "
        f"robust floor applied: {floor_used:.6g}"
    )

    train_norm = apply_normalization_nct(train_examples, mean, std)
    val_norm = apply_normalization_nct(val_examples, mean, std)
    test_norm = None if test_examples is None else apply_normalization_nct(test_examples, mean, std)

    if save_stats_path is not None:
        save_path = Path(save_stats_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        base_path = save_path.with_suffix("")
        mean_path = base_path.parent / f"{base_path.name}_mean.npy"
        std_path = base_path.parent / f"{base_path.name}_std.npy"
        meta_path = base_path.parent / f"{base_path.name}_meta.json"
        np.save(str(mean_path), mean.astype(np.float32))
        np.save(str(std_path), std.astype(np.float32))
        meta = {
            "layout": "NCT",
            "computed_on": "TRAIN_ONLY",
            "timepoints_used_for_stats": f"[0:{T_in})",
            "eps": float(eps),
            "robust_std_floor": float(floor_used),
            "median_std_before_floor": float(med_before),
            "floor_frac_median": float(floor_frac_median),
            "floor_abs": float(floor_abs),
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        print("Saved normalization stats to:")
        print(f"  Mean: {mean_path}")
        print(f"  Std:  {std_path}")
        print(f"  Meta: {meta_path}")

    return train_norm, val_norm, test_norm, (mean, std, "NCT_robust")


def summarize_continuous_split(name: str, arr: np.ndarray, near_zero_eps: float = 1e-8) -> None:
    v = arr.reshape(-1)
    print(
        f"[{name}] min={float(np.min(v)):.6g} max={float(np.max(v)):.6g} "
        f"mean={float(np.mean(v)):.6g} std={float(np.std(v)):.6g}"
    )
    zero_frac = float(np.mean(v == 0))
    near_zero_frac = float(np.mean(np.abs(v) <= near_zero_eps))
    neg_frac = float(np.mean(v < 0))
    print(
        f"[{name}] frac_zero={zero_frac:.4f} frac_near_zero(|x|<={near_zero_eps:g})={near_zero_frac:.4f} "
        f"frac_negative={neg_frac:.4f}"
    )
    if neg_frac > 0:
        print(f"[WARNING] {name} contains negative target values ({100*neg_frac:.2f}%).")


# =========================
# 1) CombinedLoss: compute components even when weights are 0 (for logging)
# =========================
"""class CombinedLoss(nn.Module):
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
"""

class CombinedLoss(nn.Module):
    def __init__(
        self,
        loss_spike_weight_beta=4.0,
        loss_spike_weight_scale=1.0,
        loss_spike_weight_clip=10.0,
        loss_onset_weight=3.0,
        loss_onset_threshold=0.1,
        loss_onset_min_target=0.0,
        loss_onset_value_weight=1.0,
        loss_underprediction_weight=1.0,
        loss_high_target_threshold=0.5,
        loss_false_positive_weight=0.1,
        loss_false_positive_threshold=0.5,
        eps=1e-8,
    ):
        super().__init__()
        self.loss_spike_weight_beta = float(loss_spike_weight_beta)
        self.loss_spike_weight_scale = float(max(loss_spike_weight_scale, eps))
        self.loss_spike_weight_clip = float(max(loss_spike_weight_clip, 1.0))
        self.loss_onset_weight = float(loss_onset_weight)
        self.loss_onset_threshold = float(loss_onset_threshold)
        self.loss_onset_min_target = float(loss_onset_min_target)
        self.loss_onset_value_weight = float(loss_onset_value_weight)
        self.loss_underprediction_weight = float(loss_underprediction_weight)
        self.loss_high_target_threshold = float(loss_high_target_threshold)
        self.loss_false_positive_weight = float(loss_false_positive_weight)
        self.loss_false_positive_threshold = float(loss_false_positive_threshold)
        self.eps = float(eps)

    def _masked_mae(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denom = mask.float().sum()
        if float(denom.item()) <= 0:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
        return (torch.abs(pred - target) * mask.float()).sum() / denom.clamp_min(self.eps)

    def forward(self, predictions, targets, context_last=None):
        """
        predictions, targets: (B, T, C)
        context_last: (B, C) or (B, 1, C)
        """
        device = predictions.device
        dtype = predictions.dtype
        B, T, C = predictions.shape
        if context_last is None:
            context_last = targets[:, :1, :]
        if context_last.ndim == 2:
            context_last = context_last.unsqueeze(1)

        abs_err = torch.abs(predictions - targets)
        base_mae = abs_err.mean()

        # Peak/high activity weighted MAE
        target_for_weight = torch.clamp(targets, min=0.0)
        rel_level = torch.clamp(target_for_weight / self.loss_spike_weight_scale, min=0.0, max=self.loss_spike_weight_clip)
        weight = 1.0 + self.loss_spike_weight_beta * rel_level
        weighted_mae = (weight * abs_err).mean()
        weighted_mae_term = weighted_mae - base_mae

        # Derivatives with context-aware first step
        dy_target = torch.zeros_like(targets)
        dy_pred = torch.zeros_like(predictions)
        dy_target[:, 0:1, :] = targets[:, 0:1, :] - context_last
        dy_pred[:, 0:1, :] = predictions[:, 0:1, :] - context_last
        if T > 1:
            dy_target[:, 1:, :] = targets[:, 1:, :] - targets[:, :-1, :]
            dy_pred[:, 1:, :] = predictions[:, 1:, :] - predictions[:, :-1, :]

        onset_mask = (dy_target > self.loss_onset_threshold) & (targets > self.loss_onset_min_target)
        onset_derivative_loss = self._masked_mae(dy_pred, dy_target, onset_mask)
        onset_value_loss = self._masked_mae(predictions, targets, onset_mask)

        high_target_mask = targets > self.loss_high_target_threshold
        important_mask = onset_mask | high_target_mask
        under_error = torch.relu(targets - predictions)
        if float(important_mask.float().sum().item()) > 0:
            underprediction_loss = (under_error * important_mask.float()).sum() / important_mask.float().sum().clamp_min(self.eps)
        else:
            underprediction_loss = torch.zeros((), device=device, dtype=dtype)

        # Weak false-positive penalty by default (for conservative behavior control)
        inactive_mask = targets <= self.loss_false_positive_threshold
        false_positive_raw = (torch.relu(predictions - self.loss_false_positive_threshold) * inactive_mask.float()).mean()

        total = (
            base_mae
            + weighted_mae_term
            + self.loss_onset_weight * onset_derivative_loss
            + self.loss_onset_value_weight * onset_value_loss
            + self.loss_underprediction_weight * underprediction_loss
            + self.loss_false_positive_weight * false_positive_raw
        )

        pred_min = predictions.min()
        pred_max = predictions.max()
        pred_mean = predictions.mean()
        tgt_min = targets.min()
        tgt_max = targets.max()
        tgt_mean = targets.mean()
        onset_frac = onset_mask.float().mean()
        high_target_frac = high_target_mask.float().mean()
        pred_negative_frac = (predictions < 0).float().mean()

        # No normalized terms now; keep interface for existing training loop.
        loss_dict_norm = {
            "base_mae": base_mae,
            "weighted_mae": weighted_mae,
            "weighted_mae_term": weighted_mae_term,
            "onset_derivative_loss": onset_derivative_loss,
            "onset_value_loss": onset_value_loss,
            "underprediction_loss": underprediction_loss,
            "false_positive_loss": false_positive_raw,
            "onset_fraction": onset_frac,
            "high_target_fraction": high_target_frac,
            "pred_min": pred_min,
            "pred_max": pred_max,
            "pred_mean": pred_mean,
            "target_min": tgt_min,
            "target_max": tgt_max,
            "target_mean": tgt_mean,
            "pred_negative_fraction": pred_negative_frac,
            "total": total,
        }
        loss_dict_raw = dict(loss_dict_norm)
        return total, loss_dict_norm, loss_dict_raw


def build_loss(config):
    if config["loss_type"] == "combined":
        return CombinedLoss(
            loss_spike_weight_beta=config.get("loss_spike_weight_beta", 4.0),
            loss_spike_weight_scale=config.get("loss_spike_weight_scale", 1.0),
            loss_spike_weight_clip=config.get("loss_spike_weight_clip", 10.0),
            loss_onset_weight=config.get("loss_onset_weight", 3.0),
            loss_onset_threshold=config.get("loss_onset_threshold", 0.1),
            loss_onset_min_target=config.get("loss_onset_min_target", 0.0),
            loss_onset_value_weight=config.get("loss_onset_value_weight", 1.0),
            loss_underprediction_weight=config.get("loss_underprediction_weight", 1.0),
            loss_high_target_threshold=config.get("loss_high_target_threshold", 0.5),
            loss_false_positive_weight=config.get("loss_false_positive_weight", 0.1),
            loss_false_positive_threshold=config.get("loss_false_positive_z_threshold", 0.5),
            eps=config.get("loss_eps", 1e-8),
        )
    
def plot_learning_curves(train_history, val_history, save_dir):
    if len(train_history) == 0 or len(val_history) == 0:
        print("[WARNING] Empty histories, skipping plot.")
        return

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    epochs = np.arange(1, len(train_history) + 1)

    # -------------------------
    # Panel 1: Total loss curves
    # -------------------------
    train_total = [h['total'] for h in train_history]

    # Determine primary validation total key
    if 'total' in val_history[0]:
        val_total_key = 'total'
    else:
        # Expect keys like 'tl_total' or 'ar_total' (primary regime stored with prefix)
        if 'tl_total' in val_history[0]:
            val_total_key = 'tl_total'
        elif 'ar_total' in val_history[0]:
            val_total_key = 'ar_total'
        else:
            candidates = [k for k in val_history[0].keys() if k.endswith('_total')]
            if not candidates:
                raise ValueError(f"No total key found in val_history[0]. Keys: {list(val_history[0].keys())}")
            val_total_key = candidates[0]

    val_total = [h[val_total_key] for h in val_history]

    best_idx = int(np.argmin(val_total))
    best_epoch = epochs[best_idx]
    best_val = val_total[best_idx]

    # -------------------------
    # Panel 2: Component curves
    # -------------------------
    # Train keys are unprefixed
    train_comp_keys = [k for k in train_history[0].keys() if k != 'total']

    # Val may be:
    # - old: unprefixed keys + 'total'
    # - new: prefixed keys ('tl_mae', 'ar_mae', ...) + ('tl_total', 'ar_total', ...)
    has_prefixed = any(
        k.startswith('tl_') or k.startswith('ar_') or k.startswith('tf_')
        for k in val_history[0].keys()
    )

    val_prefixes = []
    if has_prefixed:
        if any(k.startswith('tl_') for k in val_history[0].keys()): val_prefixes.append('tl')
        if any(k.startswith('ar_') for k in val_history[0].keys()): val_prefixes.append('ar')
        if any(k.startswith('tf_') for k in val_history[0].keys()): val_prefixes.append('tf')
    else:
        val_prefixes.append(None)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 12), sharex=True)

    # --- Panel 1 ---
    ax1.plot(epochs, train_total, label='Train total', alpha=0.4, linewidth=1.5)
    ax1.plot(epochs, val_total, label=f'Val total ({val_total_key})', linewidth=2.5)
    ax1.axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.5)
    ax1.scatter(best_epoch, best_val, color='gold', edgecolor='black', s=80, zorder=5,
                label=f'Best: {best_val:.4f} (Ep {best_epoch})')
    ax1.set_title('Global Objective Convergence', fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(True, alpha=0.2)
    ax1.legend(loc='upper right')

    # --- Panel 2 ---
    cmap = plt.get_cmap('Set1')

    for i, comp in enumerate(train_comp_keys):
        color = cmap(i % 9)

        # Train (unprefixed)
        t_comp = [h[comp] for h in train_history]
        ax2.plot(epochs, t_comp, color=color, linestyle=':', alpha=0.35, linewidth=1.2,
                 label=f'Train {comp.upper()}')

        # Val: for each prefix that exists
        for p in val_prefixes:
            if p is None:
                # old format
                if comp in val_history[0]:
                    v_comp = [h[comp] for h in val_history]
                    ax2.plot(epochs, v_comp, color=color, linewidth=2.0,
                             label=f'Val {comp.upper()}')
            else:
                key = f"{p}_{comp}"
                if key in val_history[0]:
                    v_comp = [h[key] for h in val_history]
                    ls = '-' if p == 'tl' else '--'   # TL solid, AR dashed
                    ax2.plot(epochs, v_comp, color=color, linestyle=ls, linewidth=2.0,
                             label=f'Val({p.upper()}) {comp.upper()}')

    ax2.axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.5)
    ax2.set_title('Loss Component Breakdown', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss (Log Scale)', fontsize=12)
    ax2.set_yscale('log')
    ax2.grid(True, which="both", ls="-", alpha=0.1)

    # Legend outside
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, fontsize=9)

    plt.tight_layout()
    out = save_dir / 'learning_curves_detailed.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[INFO] Detailed learning curves saved to {out}")




def should_save_fixed_checkpoint(epoch: int, first_epoch: int = 5, every: int = 25) -> bool:
    if epoch == first_epoch:
        return True
    if every is not None and every > 0 and (epoch % every == 0):
        return True
    return False



def run_model_forward(model, inputs, targets, current_p, mode):

    mode = mode.upper()

    if mode == "TL":
        return model.forward_oneshot(inputs)

    if mode == "AR":
        if not hasattr(model, "forward_autoregressive"):
            raise AttributeError("Model has no forward_autoregressive()")
        return model.forward_autoregressive(inputs)

    if mode == "AR_KV":
        if not hasattr(model, "forward_autoregressive_kvcache"):
            raise AttributeError("Model has no forward_autoregressive_kvcache()")
        return model.forward_autoregressive_kvcache(inputs)

    if mode == "MIX_TF_AR_KV":
        if not hasattr(model, "forward_teacher_forcing"):
            raise AttributeError("Model has no forward_teacher_forcing()")
        if not hasattr(model, "forward_autoregressive_kvcache"):
            raise AttributeError("Model has no forward_autoregressive_kvcache()")
        p_tf = float(np.clip(current_p, 0.0, 1.0))
        if np.random.rand() < p_tf:
            return model.forward_teacher_forcing(inputs, targets)
        return model.forward_autoregressive_kvcache(inputs)

    if mode == "TF":
        return model.forward_teacher_forcing(inputs, targets)

    if mode == "AR_SSM":
        return model.forward_autoregressive(inputs)

    if mode == "OS_SSM":
        return model.forward_oneshot(inputs)

    if mode == "TF_SSM":
        return model.forward_teacher_forcing(inputs, targets)

    raise ValueError(f"Unknown forward mode: {mode} (expected 'TL', 'AR', 'AR_KV', 'TF', 'MIX_TF_AR_KV')")



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

    scaler = torch.amp.GradScaler("cuda") if (use_amp and torch.cuda.is_available()) else None
    optimizer.zero_grad()

    import time
    t_data = t_h2d = t_fwd = t_bwd = t_opt = 0.0

    end = time.time()


    for batch_idx, (input_batch, target_batch) in enumerate(tqdm(train_loader, desc="Training")):

        # ---- DATA WAIT (time since last batch finished) ----
        t0 = time.time()
        t_data += t0 - end

        # ---- H2D COPY (optional to track) ----
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.time()

        input_batch = input_batch.to(device, non_blocking=True)
        target_batch = target_batch.to(device, non_blocking=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t2 = time.time()
        t_h2d += t2 - t1

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                predictions = run_model_forward(model, input_batch, target_batch, current_p, forward_mode)
                context_last = input_batch[:, -1, :]
                loss, loss_dict_norm, loss_dict_raw = criterion(predictions, target_batch, context_last=context_last)
                loss = loss / accumulation_steps
        else:
            predictions = run_model_forward(model, input_batch, target_batch, current_p, forward_mode)
            context_last = input_batch[:, -1, :]
            loss, loss_dict_norm, loss_dict_raw = criterion(predictions, target_batch, context_last=context_last)
            loss = loss / accumulation_steps

        if torch.isnan(loss).any():
            raise RuntimeError("NaN detected in training loss")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t3 = time.time()
        t_fwd += t3 - t2

        # ---- BACKWARD ----
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t4 = time.time()
        t_bwd += t4 - t3

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
            t5 = time.time()
            t_opt += t5 - t4
            end = t5
        else:
            # update end for correct per-batch "data wait"
            end = time.time()

        total_loss += loss.item() * accumulation_steps
        n_batches += 1
    
    print(
        f"data_wait {t_data:.1f}s | h2d {t_h2d:.1f}s | "
        f"fwd {t_fwd:.1f}s | bwd {t_bwd:.1f}s | opt {t_opt:.1f}s"
    )

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
                with torch.amp.autocast(device_type="cuda"):
                    predictions = run_model_forward(model, input_batch, target_batch, current_p, forward_mode)
                    context_last = input_batch[:, -1, :]
                    loss, loss_dict_norm, loss_dict_raw = criterion(predictions, target_batch, context_last=context_last)
            else:
                predictions = run_model_forward(model, input_batch, target_batch, current_p, forward_mode)
                context_last = input_batch[:, -1, :]
                loss, loss_dict_norm, loss_dict_raw = criterion(predictions, target_batch, context_last=context_last)

            if torch.isnan(loss).any():
                raise RuntimeError("NaN detected in validation loss")

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
    save_dir = Path(f"{config['save_dir']}_{mode_tag}")
    save_dir.mkdir(parents=True, exist_ok=True)

    model = create_model_cached(
        n_vars=config['n_vars'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        T_in=config['T_in'],
        T_out=config['T_out'],
        device=device,
        nonnegative_output=config.get("nonnegative_output", True),
        output_activation=config.get("output_activation", "softplus"),
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
    train_losses = []
    val_losses = []  # we'll store dicts; for TL they will contain both TL + AR eval entries
    train_losses_raw  = []
    val_losses_raw    = []

    no_improve_count = 0
    early_stop_patience = config.get('early_stop_patience', 10)
    early_stop_min_delta = config.get('early_stop_min_delta', 1e-6)
    compute_per_region = config.get('log_per_region', True)

    # NEW: checkpoint schedule parameters
    ckpt_first = config.get('checkpoint_first_epoch', 5)
    ckpt_every = config.get('checkpoint_every', 25)

    # Load TF pretrained weights as starting point
    """pretrain_path = Path("checkpoints_TF/best_model.pt")
    if pretrain_path.exists():
        ckpt = torch.load(pretrain_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"[INFO] Loaded TF pretrained weights")"""

    for epoch in range(1, config['num_epochs'] + 1):
        current_p = 0.5

        print(f"[{mode_tag}] Epoch={epoch}/{config['num_epochs']}")
        train_loss, train_comp, train_comp_raw = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device,
            accumulation_steps=config['accumulation_steps'],
            use_amp=config.get('use_mixed_precision', True),
            forward_mode=mode_tag, current_p=current_p,
        )

        # ---- Primary validation (matches training regime)
        val_loss, val_comp, val_comp_raw, per_region_losses = validate(
            model, val_loader, criterion, device,
            compute_per_region=compute_per_region,
            n_vars=config['n_vars'],
            use_amp=config.get('use_mixed_precision', True),
            forward_mode=mode_tag, current_p=current_p,
        )

        # In train_single_mode after primary validation:
        val_tl_loss = None
        val_tl_comp = None
        val_tl_comp_raw = None 
        if mode_tag.upper() == "AR":
            val_tl_loss, val_tl_comp, val_tl_comp_raw, _ = validate(
                model, val_loader, criterion, device,
                compute_per_region=False,
                n_vars=config['n_vars'],
                use_amp=config.get('use_mixed_precision', True),
                forward_mode="TL", current_p=current_p,
            )


        # ---- NEW: AR-rollout validation loss dict for TL mode
        val_ar_loss = None
        val_ar_comp = None
        val_ar_comp_raw = None
        #if mode_tag.upper() == "TL":
        if mode_tag.upper() in ("TL", "TF", "AR_KV", "AR_SSM", "TF_SSM", "OS_SSM"):
            val_ar_loss, val_ar_comp, val_ar_comp_raw, _ = validate(
                model, val_loader, criterion, device,
                compute_per_region=False,  # usually skip per-region for the AR-eval
                n_vars=config['n_vars'],
                use_amp=config.get('use_mixed_precision', True),
                #forward_mode="AR", current_p=current_p,
                forward_mode="AR_KV", current_p=current_p,
            )

        # Logging dicts
        train_metrics = dict(train_comp)
        train_metrics_raw = dict(train_comp_raw)
        train_metrics['total'] = float(train_loss)
        train_metrics_raw['total'] = float(train_loss)
        train_losses.append(train_metrics)
        train_losses_raw.append(train_metrics_raw)

        # store BOTH TL-eval and AR-eval in the same epoch row (distinct prefixes)
        val_metrics = {}
        # primary
        for k, v in val_comp.items():
            val_metrics[f"{mode_tag.lower()}_{k}"] = float(v)
        val_metrics[f"{mode_tag.lower()}_total"] = float(val_loss)

        # optional AR-eval
        if val_ar_comp is not None:
            for k, v in val_ar_comp.items():
                val_metrics[f"ar_{k}"] = float(v)
            val_metrics["ar_total"] = float(val_ar_loss)

        val_losses.append(val_metrics)

        val_metrics_raw = {}
        # primary
        for k, v in val_comp_raw.items():
            val_metrics_raw[f"{mode_tag.lower()}_{k}"] = float(v)
        val_metrics_raw[f"{mode_tag.lower()}_total"] = float(val_loss)

        # optional AR-eval
        if val_ar_comp_raw is not None:
            for k, v in val_ar_comp_raw.items():
                val_metrics_raw[f"ar_{k}"] = float(v)
            val_metrics_raw["ar_total"] = float(val_ar_loss)

        val_losses_raw.append(val_metrics_raw)

        # optional TL-eval (only when mode_tag == AR)
        if val_tl_comp is not None:
            for k, v in val_tl_comp.items():
                val_metrics[f"tl_{k}"] = float(v)
            val_metrics["tl_total"] = float(val_tl_loss)
        if val_tl_comp is not None:
            print(
                f"  Val(TL rollout): Total {val_tl_loss:.4f} | "
                f"base_mae {val_tl_comp.get('base_mae', float('nan')):.4f} | "
                f"onset_frac {val_tl_comp.get('onset_fraction', float('nan')):.4f}"
            )


        # Console prints
        print(
            f"  Train: Total {train_loss:.4f} | "
            f"base_mae {train_comp.get('base_mae', float('nan')):.4f} | "
            f"weighted_mae {train_comp.get('weighted_mae', float('nan')):.4f} | "
            f"onset_d {train_comp.get('onset_derivative_loss', float('nan')):.4f} | "
            f"onset_v {train_comp.get('onset_value_loss', float('nan')):.4f} | "
            f"under {train_comp.get('underprediction_loss', float('nan')):.4f}"
        )
        print(
            f"  Val({mode_tag}): Total {val_loss:.4f} | "
            f"base_mae {val_comp.get('base_mae', float('nan')):.4f} | "
            f"weighted_mae {val_comp.get('weighted_mae', float('nan')):.4f} | "
            f"onset_d {val_comp.get('onset_derivative_loss', float('nan')):.4f} | "
            f"onset_v {val_comp.get('onset_value_loss', float('nan')):.4f} | "
            f"under {val_comp.get('underprediction_loss', float('nan')):.4f}"
        )
        if val_ar_comp is not None:
            print(
                f"  Val(AR rollout): Total {val_ar_loss:.4f} | "
                f"base_mae {val_ar_comp.get('base_mae', float('nan')):.4f} | "
                f"onset_frac {val_ar_comp.get('onset_fraction', float('nan')):.4f}"
            )

        # -------------------------
        # NEW: fixed checkpoint schedule (epoch 5, then every 25)
        # Save *after* validation so the checkpoint corresponds to the logged metrics
        # -------------------------
        if should_save_fixed_checkpoint(epoch, first_epoch=ckpt_first, every=ckpt_every):

            to_save = model._orig_mod if hasattr(model, "_orig_mod") else model

            torch.save({
                'epoch': epoch,
                'mode': mode_tag,
                'mode_label': mode_label,
                'model_state_dict': to_save.state_dict(), #'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': float(train_loss),
                'val_loss_primary': float(val_loss),
                'val_comp_primary': {k: float(v) for k, v in val_comp.items()},
                'val_comp_primary_raw': {k: float(v) for k, v in val_comp_raw.items()},
                'val_loss_ar': float(val_ar_loss) if val_ar_loss is not None else None,
                'val_comp_ar': {k: float(v) for k, v in val_ar_comp.items()} if val_ar_comp is not None else None,
                'val_comp_ar_raw': {k: float(v) for k, v in val_ar_comp_raw.items()} if val_ar_comp_raw is not None else None,
                'config': config,
            }, save_dir / f'checkpoint_fixed_epoch_{epoch}.pt')
            print(f"  [Saved fixed checkpoint @ epoch {epoch}]")

        # -------------------------
        # Best model selection:
        # keep it based on PRIMARY val loss (regime-matched)
        # (You can switch to AR loss later if you want.)
        # -------------------------
        if val_loss < best_val_loss - early_stop_min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve_count = 0
            torch.save({
                'epoch': epoch,
                'mode': mode_tag,
                'mode_label': mode_label,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': float(train_loss),
                'val_loss': float(val_loss),
                'config': config,
            }, save_dir / 'best_model.pt')
            print(f"  [Saved best model with val_loss: {val_loss:.6f}]")
        else:
            no_improve_count += 1

        if no_improve_count >= early_stop_patience:
            print(f"  [{mode_tag}] Early stopping triggered after {no_improve_count} epochs without improvement.")
            break

        # (Optional) keep your old periodic saving if you want, but it becomes redundant
        # if epoch % config['save_every'] == 0: ...

    torch.save({
        'epoch': epoch if epoch < config['num_epochs'] else config['num_epochs'],
        'mode': mode_tag,
        'mode_label': mode_label,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': float(train_loss),
        'val_loss': float(val_loss),
        'config': config,
        'train_losses': train_losses,
        'train_losses_raw': train_losses_raw,
        'val_losses': val_losses,
        'val_losses_raw': val_losses_raw,
    }, save_dir / 'final_model.pt')

    plot_learning_curves(train_losses, val_losses, save_dir)
    print(f"\n[{mode_tag}] Training complete! Best val loss {best_val_loss:.6f} (epoch {best_epoch})")

    return {
        'mode': mode_tag,
        'label': mode_label,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'save_dir': save_dir,
    }


def _allocate_split_counts(n_items, train_ratio, val_ratio, test_ratio):
    """Allocate n_items to train/val/test with ratio matching and minimum presence when possible."""
    ratios = np.array([train_ratio, val_ratio, test_ratio], dtype=np.float64)
    raw = ratios * n_items
    counts = np.floor(raw).astype(int)
    remainder = int(n_items - counts.sum())

    if remainder > 0:
        frac = raw - counts
        order = np.argsort(-frac)
        for i in order[:remainder]:
            counts[i] += 1

    # If enough items, ensure each split gets at least one item
    # so early/mid/late bins can be represented everywhere.
    if n_items >= 3 and np.all(ratios > 0):
        for i in range(3):
            if counts[i] == 0:
                donor = int(np.argmax(counts))
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[i] += 1

    # Final guard to keep exact total
    diff = int(n_items - counts.sum())
    if diff != 0:
        counts[0] += diff

    return int(counts[0]), int(counts[1]), int(counts[2])


def split_examples_temporal_stratified_train_val_test(
    examples,
    sequence_indices,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    random_seed=42,
):
    """
    Split examples into train/val/test while mixing early/mid/late subsequences in all splits.

    For each sequence:
      - keep subsequence order using flattened example index order
      - split ordered subsequences into 3 temporal bins: beginning/middle/end
      - randomly assign subsequences within each bin to train/val/test
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(
            f"train/val/test ratios must sum to 1.0, got "
            f"{train_ratio} + {val_ratio} + {test_ratio} = {total_ratio}"
        )

    rng = np.random.default_rng(random_seed)
    unique_sequences = np.unique(sequence_indices)

    train_idx_parts = []
    val_idx_parts = []
    test_idx_parts = []

    for seq_id in unique_sequences:
        seq_flat_idx = np.where(sequence_indices == seq_id)[0]
        seq_flat_idx = np.sort(seq_flat_idx)
        temporal_bins = np.array_split(seq_flat_idx, 3)

        for bin_idx in temporal_bins:
            n_bin = len(bin_idx)
            if n_bin == 0:
                continue

            shuffled = rng.permutation(bin_idx)
            n_train, n_val, n_test = _allocate_split_counts(
                n_bin, train_ratio, val_ratio, test_ratio
            )

            train_idx_parts.append(shuffled[:n_train])
            val_idx_parts.append(shuffled[n_train:n_train + n_val])
            test_idx_parts.append(shuffled[n_train + n_val:n_train + n_val + n_test])

    train_idx = np.sort(np.concatenate(train_idx_parts)) if train_idx_parts else np.array([], dtype=int)
    val_idx = np.sort(np.concatenate(val_idx_parts)) if val_idx_parts else np.array([], dtype=int)
    test_idx = np.sort(np.concatenate(test_idx_parts)) if test_idx_parts else np.array([], dtype=int)

    train_examples = examples[train_idx]
    val_examples = examples[val_idx]
    test_examples = examples[test_idx]
    train_seq_indices = sequence_indices[train_idx]
    val_seq_indices = sequence_indices[val_idx]
    test_seq_indices = sequence_indices[test_idx]

    print("\nTemporal-stratified split (beginning/middle/end mixed in all splits):")
    print(f"  Total examples: {len(examples)}")
    print(f"  Train: {len(train_examples)} ({100 * len(train_examples) / max(1, len(examples)):.1f}%)")
    print(f"  Val:   {len(val_examples)} ({100 * len(val_examples) / max(1, len(examples)):.1f}%)")
    print(f"  Test:  {len(test_examples)} ({100 * len(test_examples) / max(1, len(examples)):.1f}%)")

    return (
        train_examples,
        val_examples,
        test_examples,
        train_seq_indices,
        val_seq_indices,
        test_seq_indices,
    )


def split_examples_blocked_with_gap(
    examples,
    sequence_indices,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    gap_examples=0,
):
    """
    Contiguous blocked split to avoid overlap leakage between train/val/test.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(
            f"train/val/test ratios must sum to 1.0, got "
            f"{train_ratio} + {val_ratio} + {test_ratio} = {total_ratio}"
        )
    n = len(examples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    g = int(max(0, gap_examples))

    train_end = max(0, min(n, n_train))
    val_start = max(train_end + g, 0)
    val_end = max(val_start, min(n, val_start + n_val))
    test_start = max(val_end + g, 0)

    train_idx = np.arange(0, train_end, dtype=np.int64)
    val_idx = np.arange(val_start, val_end, dtype=np.int64)
    test_idx = np.arange(test_start, n, dtype=np.int64)

    train_examples = examples[train_idx]
    val_examples = examples[val_idx]
    test_examples = examples[test_idx]
    train_seq_indices = sequence_indices[train_idx]
    val_seq_indices = sequence_indices[val_idx]
    test_seq_indices = sequence_indices[test_idx]

    print("\nBlocked split with gap:")
    print(f"  Total examples: {n}")
    print(f"  Gap examples between splits: {g}")
    print(f"  Train: {len(train_examples)} ({100 * len(train_examples) / max(1, n):.1f}%)")
    print(f"  Val:   {len(val_examples)} ({100 * len(val_examples) / max(1, n):.1f}%)")
    print(f"  Test:  {len(test_examples)} ({100 * len(test_examples) / max(1, n):.1f}%)")

    return (
        train_examples,
        val_examples,
        test_examples,
        train_seq_indices,
        val_seq_indices,
        test_seq_indices,
    )



def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train 2p continuous spike-like trace model.")
    parser.add_argument("--normalization_mode", type=str, default="none", choices=("none", "robust_zscore"))
    parser.add_argument("--train_regime", type=str, default="AR_KV")
    parser.add_argument("--split_mode", type=str, default="blocked", choices=("blocked", "random"))
    parser.add_argument("--split_gap_timesteps", type=int, default=120)
    parser.add_argument("--nonnegative_output", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=True)
    parser.add_argument("--output_activation", type=str, default="softplus", choices=("softplus", "relu", "clamp", "identity"))
    parser.add_argument("--loss_onset_weight", type=float, default=3.0)
    parser.add_argument("--loss_onset_value_weight", type=float, default=1.0)
    parser.add_argument("--loss_onset_quantile", type=float, default=0.95)
    parser.add_argument("--loss_onset_threshold", type=float, default=0.1)
    parser.add_argument("--loss_onset_min_target", type=float, default=0.0)
    parser.add_argument("--loss_spike_weight_beta", type=float, default=4.0)
    parser.add_argument("--loss_spike_weight_scale_quantile", type=float, default=0.95)
    parser.add_argument("--loss_spike_weight_clip", type=float, default=10.0)
    parser.add_argument("--loss_underprediction_weight", type=float, default=1.0)
    parser.add_argument("--loss_high_target_quantile", type=float, default=0.95)
    parser.add_argument("--loss_false_positive_weight", type=float, default=0.1)
    args = parser.parse_args()

    print("="*80)
    print("TRAINING SIMPLE DECODER-ONLY TRANSFORMER")
    print("="*80)
    
    # Clear any existing GPU memory before starting
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("\n[INFO] Cleared GPU cache before training")
        # Print GPU memory stats
        print(f"[INFO] GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"[INFO] GPU Memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

    # Configuration (matching netho-hp-search-9-run-2.yaml)
    config = {
        'data_path': 'data_processed/data2p_A_3_1189451_S10_C_dec_10Hz.npy', #'data_processed/data100_ba16.npy',
        'T_in': 90, #90,
        'T_out': 1, # next-step for rollout realism
        'n_vars': 88,
        'd_model': 256,           # model_spec.d_model
        'n_heads': 8,            # model_spec.nhead
        'n_layers': 4,           # model_spec.nlayers
        'd_ff': 256,             # model_spec.d_hid (feedforward dimension)
        'dropout': 0.1,       # training_spec.dropout
        'patch_len': 1,          # Number of time-steps per patch (Timer-XL style)
        'effective_batch_size': 32, #2000,  # Original batch size from YAML (effective via accumulation)
        'batch_size': 32, #2048, #512,        # Physical batch size (teacher forcing is memory efficient)
        'accumulation_steps': 1, #4,  # 64 * 32 ≈ 2048 effective batch size
        'use_mixed_precision': True,  # Use FP16 to reduce memory by ~50%
        'loss_type': 'combined',
        'loss_mae_weight': 1.0,
        'loss_shape_weight': 0.0,  # High weight to force "wiggles"
        'loss_deriv_weight': 0.0,  # Moderate weight to force smooth transitions
        'loss_cross_weight': 0.0,  # Low weight to maintain population structure
        'loss_var_weight': 0.0,
        # Continuous spike-like trace objectives / scaling
        'normalization_mode': args.normalization_mode,
        'norm_eps': 1e-5,
        'norm_std_floor_abs': 1e-4,
        'norm_std_floor_frac_median': 0.25,
        'loss_spike_weight_beta': float(args.loss_spike_weight_beta),
        'loss_spike_weight_scale_quantile': float(args.loss_spike_weight_scale_quantile),
        'loss_spike_weight_clip': float(args.loss_spike_weight_clip),
        'loss_spike_weight_scale': 1.0,
        'loss_onset_weight': float(args.loss_onset_weight),
        'loss_onset_value_weight': float(args.loss_onset_value_weight),
        'loss_onset_quantile': float(args.loss_onset_quantile),
        'loss_onset_threshold': float(args.loss_onset_threshold),
        'loss_onset_min_target': float(args.loss_onset_min_target),
        'loss_underprediction_weight': float(args.loss_underprediction_weight),
        'loss_high_target_quantile': float(args.loss_high_target_quantile),
        'loss_high_target_threshold': 1.0,
        'loss_false_positive_weight': float(args.loss_false_positive_weight),
        'loss_false_positive_z_threshold': 1.0,
        'train_regime': args.train_regime,
        'split_mode': args.split_mode,
        'split_gap_timesteps': int(args.split_gap_timesteps),
        'nonnegative_output': bool(args.nonnegative_output),
        'output_activation': str(args.output_activation),
        'learning_rate': 1e-4,    # training_spec.lr (will use scheduler instead)
        'max_lr': 0.0003,         # scheduler.max_lr
        'num_epochs': 150, #150 ,     # training_spec.epochs
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        'random_seed': 101,      # seed from YAML
        'save_dir': 'checkpoints',
        'plot_dir': 'evaluation',
        'log_all_loss_terms': False,
        'checkpoint_first_epoch': 5,     # <-- NEW
        'checkpoint_every': 25, 
        'save_every': 15,       # training_spec.iter_save
        'early_stop_patience': 150, #15,  # training_spec.early_stopping_epochs
        'early_stop_min_delta': 1e-6,  # Minimum change to qualify as improvement
        'log_per_region': False,  # Log per-region loss breakdown
        # Scheduler parameters (OneCycleLR)
        # Updated Scheduler parameters (OneCycleLR)
        'scheduler_pct_start': 0.3,           # Spend 30% of time warming up (90 epochs)
        'scheduler_div_factor': 25,           # Start at max_lr / 25
        'scheduler_final_div_factor': 100,    # End at max_lr / 100 (keep it higher)
        'scheduler_anneal_strategy': 'cos',
        'use_torch_compile': False,
    }
    
    # Print configuration summary
    print("\nConfiguration (from YAML netho-hp-search-9-run-2):")
    print(f"  Model:")
    print(f"    d_model: {config['d_model']}")
    print(f"    n_heads: {config['n_heads']}")
    print(f"    n_layers: {config['n_layers']}")
    print(f"    d_ff (d_hid): {config['d_ff']}")
    print(f"    dropout: {config['dropout']}")
    print(f"    patch_len: {config['patch_len']}")
    print(f"  Training:")
    print(f"    effective_batch_size: {config['effective_batch_size']} (via gradient accumulation)")
    print(f"    physical_batch_size: {config['batch_size']}")
    print(f"    accumulation_steps: {config['accumulation_steps']}")
    print(f"    mixed_precision (FP16): {config.get('use_mixed_precision', False)}")
    print(f"    epochs: {config['num_epochs']}")
    print(f"    optimizer: AdamW")
    print(f"    scheduler: OneCycleLR (max_lr={config['max_lr']})")
    if config.get('loss_type', 'l1') == 'combined':
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
        loss_desc = config.get('loss_type', 'l1').upper()
    print(f"    criterion: {loss_desc}")
    print(f"    early_stopping: {config['early_stop_patience']} epochs")
    print(f"    save_every: {config['save_every']} epochs")
    print(f"    seed: {config['random_seed']}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['random_seed'])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load and prepare data
    print("\n" + "="*80)
    print("PREPARING DATA")
    print("="*80)
    data_array = load_data(config['data_path'])
    data_n_vars = int(data_array.shape[2])
    if config.get('n_vars') != data_n_vars:
        print(
            f"[INFO] Overriding n_vars from {config.get('n_vars')} to {data_n_vars} "
            f"to match loaded data shape."
        )
        config['n_vars'] = data_n_vars
    examples, sequence_indices = reshape_to_examples(data_array)
    
    # Free the original data array from memory
    del data_array
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Filter out NaN examples
    examples, sequence_indices = filter_examples_by_nan(
        examples, sequence_indices,
        T_in=config['T_in'],
        T_out=config['T_out']
    )
    
    # Convert split gap in timesteps to examples using prepare metadata window_stride when available.
    gap_examples = int(config.get("split_gap_timesteps", 0))
    metadata_path = Path(str(config["data_path"])).with_suffix("").with_name(Path(str(config["data_path"])).stem + "_metadata.json")
    if metadata_path.exists():
        try:
            md = json.loads(metadata_path.read_text())
            stride = int(md.get("partition", {}).get("window_stride", 1))
            if stride > 0:
                gap_examples = int(math.ceil(config.get("split_gap_timesteps", 0) / stride))
                print(f"[split] using window_stride={stride} -> gap_examples={gap_examples}")
        except Exception:
            pass

    if str(config.get("split_mode", "blocked")).lower() == "blocked":
        train_examples, val_examples, test_examples, train_seq_indices, val_seq_indices, test_seq_indices = (
            split_examples_blocked_with_gap(
                examples,
                sequence_indices,
                train_ratio=config['train_ratio'],
                val_ratio=config['val_ratio'],
                test_ratio=config['test_ratio'],
                gap_examples=gap_examples,
            )
        )
    else:
        train_examples, val_examples, test_examples, train_seq_indices, val_seq_indices, test_seq_indices = (
            split_examples_temporal_stratified_train_val_test(
                examples,
                sequence_indices,
                train_ratio=config['train_ratio'],
                val_ratio=config['val_ratio'],
                test_ratio=config['test_ratio'],
                random_seed=config['random_seed'],
            )
        )


    run_stem = (
        f"{Path(config['data_path']).stem}"
        f"_Tin{config['T_in']}_Tout{config['T_out']}_seed{config['random_seed']}"
    )

    proc_root = Path("data_processed")
    proc_root.mkdir(parents=True, exist_ok=True)

    stats_npy = proc_root / f"train_norm_stats_{run_stem}.npy"
    val_npy = proc_root / f"processed_val_{run_stem}.npy"
    val_seq_npy = proc_root / f"processed_val_seq_indices_{run_stem}.npy"
    test_npy = proc_root / f"processed_test_{run_stem}.npy"
    test_seq_npy = proc_root / f"processed_test_seq_indices_{run_stem}.npy"

    train_examples, val_examples, test_examples, norm = normalize_split_robust_2p(
        train_examples,
        val_examples,
        test_examples,
        T_in=config["T_in"],
        eps=float(config["norm_eps"]),
        floor_abs=float(config["norm_std_floor_abs"]),
        floor_frac_median=float(config["norm_std_floor_frac_median"]),
        normalization_mode=str(config.get("normalization_mode", "robust_zscore")),
        save_stats_path=None if str(config.get("normalization_mode", "robust_zscore")).lower() == "none" else str(stats_npy),
    )

    summarize_continuous_split("train_targets", train_examples[:, config["T_in"]:config["T_in"] + config["T_out"], :])
    summarize_continuous_split("val_targets", val_examples[:, config["T_in"]:config["T_in"] + config["T_out"], :])
    summarize_continuous_split("test_targets", test_examples[:, config["T_in"]:config["T_in"] + config["T_out"], :])

    # Derive robust thresholds/scales from training targets for onset/peak weighting.
    train_targets = train_examples[:, config["T_in"]:config["T_in"] + config["T_out"], :]
    pos_vals = train_targets[train_targets > 0]
    if pos_vals.size > 0:
        config["loss_spike_weight_scale"] = float(np.quantile(pos_vals, config.get("loss_spike_weight_scale_quantile", 0.95)))
        config["loss_high_target_threshold"] = float(np.quantile(pos_vals, config.get("loss_high_target_quantile", 0.95)))
    else:
        config["loss_spike_weight_scale"] = 1.0
        config["loss_high_target_threshold"] = 1.0

    # Onset threshold from positive derivatives if requested.
    context_last_np = train_examples[:, config["T_in"] - 1:config["T_in"], :]
    dy0 = train_targets[:, 0:1, :] - context_last_np
    if train_targets.shape[1] > 1:
        dyn = train_targets[:, 1:, :] - train_targets[:, :-1, :]
        dy = np.concatenate([dy0, dyn], axis=1)
    else:
        dy = dy0
    pos_dy = dy[dy > 0]
    if pos_dy.size > 0 and config.get("loss_onset_quantile", None) is not None:
        config["loss_onset_threshold"] = float(np.quantile(pos_dy, config.get("loss_onset_quantile", 0.95)))
    print(
        f"[loss scales] spike_scale={config['loss_spike_weight_scale']:.6g} "
        f"onset_thr={config['loss_onset_threshold']:.6g} "
        f"high_target_thr={config['loss_high_target_threshold']:.6g}"
    )

    np.save(val_npy, val_examples)  # processed val (raw if normalization_mode=none)
    np.save(val_seq_npy, val_seq_indices.astype(np.int64, copy=False))
    np.save(test_npy, test_examples)  # processed test (raw if normalization_mode=none)
    np.save(test_seq_npy, test_seq_indices.astype(np.int64, copy=False))

    config["normalization_stats_base"] = (
        str(stats_npy.with_suffix("").resolve())
        if str(config.get("normalization_mode", "robust_zscore")).lower() != "none"
        else None
    )
    config["processed_val_examples_path"] = str(val_npy.resolve())
    config["processed_val_seq_indices_path"] = str(val_seq_npy.resolve())
    config["processed_test_examples_path"] = str(test_npy.resolve())
    config["processed_test_seq_indices_path"] = str(test_seq_npy.resolve())
    
    # Free examples array after splitting
    del examples, sequence_indices
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Create dataloaders
    # Guard for small datasets: create_dataloaders uses drop_last=True for train,
    # so batch_size must not exceed number of train examples or we get 0 train batches.
    if len(train_examples) == 0:
        raise ValueError("No training examples after split/filtering. Cannot train.")
    effective_batch_size = int(min(config['batch_size'], len(train_examples)))
    if effective_batch_size < config['batch_size']:
        print(
            f"[INFO] Reducing batch_size from {config['batch_size']} to {effective_batch_size} "
            f"to avoid zero training batches on this dataset."
        )
    config['batch_size'] = effective_batch_size

    train_loader, val_loader = create_dataloaders(
        train_examples,
        val_examples,
        T_in=config['T_in'],
        T_out=config['T_out'],
        batch_size=config['batch_size'],
        num_workers=6,
    )

    #verify_data_loading(train_loader, val_loader)
    
    # Train both modes sequentially
    train_regime = str(config.get("train_regime", "AR_KV")).upper()
    if train_regime in ("MIX_TF_AR_KV", "MIXED"):
        training_variants = [{"tag": "MIX_TF_AR_KV", "label": "Mixed Teacher Forcing + AR_KV"}]
    elif train_regime == "TF":
        training_variants = [{"tag": "TF", "label": "Teacher Forcing"}]
    else:
        training_variants = [{"tag": "AR_KV", "label": "KV Autoregressive"}]

    
    histories = []
    for variant in training_variants:

        history = train_single_mode(
            config, 
            train_loader, 
            val_loader, 
            device,
            mode_tag=variant['tag'],
            mode_label=variant['label'],
        )
        histories.append(history)


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