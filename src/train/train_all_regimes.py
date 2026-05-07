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
from models.model_KV_cached import create_model_cached



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
        mae_weight=1.0,
        shape_weight=0.0,
        deriv_weight=0.0,
        cross_weight=0.0,
        var_weight=0.0,
        kl_weight=0.0,
        qnt_weight=0.0,
        trj_weight=0.0,
        log_all_terms: bool = True,
        kl_bins: int = 33,
        kl_eps: float = 1e-8,
        kl_support_low: float = 0.001,
        kl_support_high: float = 0.999,
        kl_use_q10: bool = True,
        kl_q10_weight: float = 0.5,
        qnt_q_lo: float = 0.01,
        qnt_q_hi: float = 0.99,
        qnt_n_q: int = 99,
        qnt_tail_lo: float = 0.10,
        qnt_tail_hi: float = 0.90,
        qnt_top_q_regions: float = 0.25,
        qnt_eps: float = 1e-8,
    ):
        super().__init__()
        self.weights = {
            'mae': mae_weight,
            'shape': shape_weight,
            'deriv': deriv_weight,
            'cross': cross_weight,
            'var': var_weight,
            'kl': kl_weight,
            'qnt': qnt_weight,
        }
        self.log_all_terms = log_all_terms
        self.mae = nn.L1Loss()

        # KL params
        self.kl_bins = kl_bins
        self.kl_eps = kl_eps
        self.kl_support_low = kl_support_low
        self.kl_support_high = kl_support_high
        self.kl_use_q10 = kl_use_q10
        self.kl_q10_weight = kl_q10_weight

        # QNT params
        self.qnt_q_lo = qnt_q_lo
        self.qnt_q_hi = qnt_q_hi
        self.qnt_n_q = qnt_n_q
        self.qnt_tail_lo = qnt_tail_lo
        self.qnt_tail_hi = qnt_tail_hi
        self.qnt_top_q_regions = qnt_top_q_regions
        self.qnt_eps = qnt_eps

        # mae, shape, deriv, cross, var, kl, qnt, trj
        self.register_buffer('running_norms', torch.ones(8))
        self.initialized = False

    def _soft_histogram_probs(self, x, lo, hi, bins, eps):
        """
        x:  (B, T, V)
        lo: (V,)
        hi: (V,)
        returns probs: (B, V, K)
        """
        B, T, V = x.shape
        device = x.device
        dtype = x.dtype
        K = bins

        widths = (hi - lo).clamp_min(1e-6) / K
        centers = lo.unsqueeze(-1) + (torch.arange(K, device=device, dtype=dtype) + 0.5) * widths.unsqueeze(-1)

        x_exp = x.unsqueeze(-1)                    # (B, T, V, 1)
        c_exp = centers.unsqueeze(0).unsqueeze(0) # (1, 1, V, K)

        sigma = 0.5 * widths
        sigma_exp = sigma.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # (1, 1, V, 1)

        weights = torch.exp(-0.5 * ((x_exp - c_exp) / sigma_exp) ** 2)
        probs = weights.sum(dim=1)  # (B, V, K)
        probs = probs + eps
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return probs

    def _soft_histogram_1d(self, x, lo, hi, bins, eps):
        """
        x: (N,)
        returns probs: (K,)
        """
        x = x.reshape(-1)
        dtype = x.dtype
        device = x.device

        lo = torch.as_tensor(lo, device=device, dtype=dtype)
        hi = torch.as_tensor(hi, device=device, dtype=dtype)
        width = (hi - lo).clamp_min(1e-6) / bins
        centers = lo + (torch.arange(bins, device=device, dtype=dtype) + 0.5) * width
        sigma = 0.5 * width

        w = torch.exp(-0.5 * ((x[:, None] - centers[None, :]) / sigma) ** 2)
        probs = w.sum(dim=0) + eps
        probs = probs / probs.sum()
        return probs

    def _score_from_distance_torch(self, distance, eps=1e-8):
        return 1.0 / (1.0 + torch.clamp(distance, min=0.0))

    def _corr_score_torch(self, x, y, eps=1e-8):
        x = x.reshape(-1)
        y = y.reshape(-1)
        x = x - x.mean()
        y = y - y.mean()
        denom = torch.sqrt((x * x).sum() + eps) * torch.sqrt((y * y).sum() + eps)
        return (x * y).sum() / denom

    def _batch_corr_score_torch(self, x, y, eps=1e-8):
        """
        x, y: (B, T)
        returns: (B,)
        """
        x = x - x.mean(dim=1, keepdim=True)
        y = y - y.mean(dim=1, keepdim=True)
        denom = torch.sqrt((x * x).sum(dim=1) + eps) * torch.sqrt((y * y).sum(dim=1) + eps)
        return (x * y).sum(dim=1) / denom

    def _quantile_distance_torch(self, x, y, qs, eps=1e-8):
        x = x.float().reshape(-1)
        y = y.float().reshape(-1)
        q = torch.tensor(qs, device=x.device, dtype=x.dtype)
        qx = torch.quantile(x, q)
        qy = torch.quantile(y, q)
        denom = torch.mean(torch.abs(qx)) + eps
        return torch.mean(torch.abs(qx - qy) / denom)

    def _weighted_mean_available_torch(self, values, weights):
        """
        values: dict[str, scalar tensor]
        weights: dict[str, float]
        """
        num = None
        den = 0.0
        for key, value in values.items():
            w = float(weights.get(key, 0.0))
            if w <= 0:
                continue
            if num is None:
                num = value * w
            else:
                num = num + value * w
            den += w
        if num is None or den <= 0:
            return torch.tensor(float('nan'), device=next(iter(values.values())).device)
        return num / den

    def _subsample_rows(self, x, max_points):
        """
        x: (N, D)
        returns: (min(N, max_points), D)
        """
        if max_points is None or max_points <= 0 or x.shape[0] <= max_points:
            return x
        idx = torch.randperm(x.shape[0], device=x.device)[:max_points]
        return x.index_select(0, idx)

    def _subsample_pair_1d(self, x, y, max_points):
        """
        x, y: (N,)
        returns same-index subsample of both
        """
        x = x.reshape(-1)
        y = y.reshape(-1)
        n = x.shape[0]
        if max_points is None or max_points <= 0 or n <= max_points:
            return x, y
        idx = torch.randperm(n, device=x.device)[:max_points]
        return x.index_select(0, idx), y.index_select(0, idx)

    def _kl_distribution_loss(self, predictions, targets):
        """
        Benchmark-inspired surrogate for KL_score01_avg:
          - per-sequence, per-region soft histograms
          - symmetric KL
          - similarity = 1 / (1 + KL_sym)
          - geometric mean across regions
          - average with lower-tail q10 over sequences
        """
        B, T, V = predictions.shape
        device = predictions.device
        dtype = predictions.dtype
        eps = self.kl_eps

        pooled = torch.cat([targets.detach(), predictions.detach()], dim=1)  # (B, 2T, V)
        pooled_flat = pooled.reshape(-1, V).float()
        lo = torch.quantile(pooled_flat, self.kl_support_low, dim=0)
        hi = torch.quantile(pooled_flat, self.kl_support_high, dim=0)
        hi = torch.maximum(hi, lo + 1e-6)

        p = self._soft_histogram_probs(targets, lo.to(dtype), hi.to(dtype), self.kl_bins, eps)
        q = self._soft_histogram_probs(predictions, lo.to(dtype), hi.to(dtype), self.kl_bins, eps)

        kl_pq = (p * (torch.log(p + eps) - torch.log(q + eps))).sum(dim=-1)  # (B, V)
        kl_qp = (q * (torch.log(q + eps) - torch.log(p + eps))).sum(dim=-1)  # (B, V)
        kl_sym = 0.5 * (kl_pq + kl_qp)

        sim = 1.0 / (1.0 + kl_sym)
        sim = sim.clamp(min=eps, max=1.0)

        kl_geo_seq = torch.exp(torch.mean(torch.log(sim), dim=1))  # (B,)

        kl_mean = kl_geo_seq.mean()
        if self.kl_use_q10:
            kl_q10 = torch.quantile(kl_geo_seq.float(), 0.10).to(dtype)
            kl_score01_avg = self.kl_q10_weight * kl_mean + (1.0 - self.kl_q10_weight) * kl_q10
        else:
            kl_q10 = torch.zeros((), device=device, dtype=dtype)
            kl_score01_avg = kl_mean

        kl_loss = 1.0 - kl_score01_avg
        return kl_loss, kl_mean, kl_q10, kl_score01_avg

    def _qnt_distribution_loss(self, predictions, targets):
        """
        Benchmark-inspired surrogate for QNT_score01:
          Q_gt(b,v,q) = quantile_q of targets[b,:,v]
          Q_pr(b,v,q) = quantile_q of preds[b,:,v]
          d_tail(b,v) = mean over tail quantiles of |Q_gt - Q_pr| / (IQR_GT(v) + eps)
          D_b         = mean of top-q worst regions for sequence b
          D           = mean_b D_b
          QNT_score01 = 1 / (1 + D)

        Loss = 1 - QNT_score01
        """
        B, T, V = predictions.shape
        device = predictions.device
        dtype = predictions.dtype
        eps = self.qnt_eps

        targets_f = targets.float()
        predictions_f = predictions.float()

        quantiles = torch.linspace(
            self.qnt_q_lo,
            self.qnt_q_hi,
            self.qnt_n_q,
            device=device,
            dtype=targets_f.dtype,
        )

        tail_mask = (quantiles <= self.qnt_tail_lo) | (quantiles >= self.qnt_tail_hi)
        tail_idx = torch.where(tail_mask)[0]
        if tail_idx.numel() == 0:
            raise ValueError("QNT tail mask is empty. Check qnt_tail_lo/qnt_tail_hi settings.")

        targ_flat = targets_f.reshape(-1, V)
        q25 = torch.quantile(targ_flat, 0.25, dim=0)
        q75 = torch.quantile(targ_flat, 0.75, dim=0)
        iqr_gt = (q75 - q25).clamp_min(eps)

        q_gt = torch.quantile(targets_f, quantiles, dim=1)      # (Q, B, V)
        q_pr = torch.quantile(predictions_f, quantiles, dim=1)  # (Q, B, V)

        q_gt = q_gt.permute(1, 2, 0)  # (B, V, Q)
        q_pr = q_pr.permute(1, 2, 0)  # (B, V, Q)

        dq = torch.abs(q_gt - q_pr) / iqr_gt.view(1, V, 1)
        d_tail = dq[:, :, tail_idx].mean(dim=-1)  # (B, V)

        k = max(1, int(math.ceil(self.qnt_top_q_regions * V)))
        topk_vals, _ = torch.topk(d_tail, k=k, dim=1, largest=True, sorted=False)
        D_seq = topk_vals.mean(dim=1)  # (B,)

        D = D_seq.mean()
        qnt_score01 = 1.0 / (1.0 + D)
        qnt_loss = 1.0 - qnt_score01

        return qnt_loss, D, qnt_score01, D_seq, d_tail

    def forward(self, predictions, targets):
        B, T, V = predictions.shape
        device = predictions.device
        dtype = predictions.dtype
        eps = 1e-6

        mae_raw = self.mae(predictions, targets)

        shape_raw = torch.zeros((), device=device, dtype=dtype)
        deriv_raw = torch.zeros((), device=device, dtype=dtype)
        cross_raw = torch.zeros((), device=device, dtype=dtype)
        var_raw   = torch.zeros((), device=device, dtype=dtype)
        kl_raw    = torch.zeros((), device=device, dtype=dtype)
        qnt_raw   = torch.zeros((), device=device, dtype=dtype)
        trj_raw   = torch.zeros((), device=device, dtype=dtype)

        kl_mean_raw = torch.zeros((), device=device, dtype=dtype)
        kl_q10_raw = torch.zeros((), device=device, dtype=dtype)
        kl_score01_avg_raw = torch.zeros((), device=device, dtype=dtype)

        qnt_D_raw = torch.zeros((), device=device, dtype=dtype)
        qnt_score01_raw = torch.zeros((), device=device, dtype=dtype)

        need_shape = (self.weights['shape'] != 0) or self.log_all_terms
        need_deriv = (self.weights['deriv'] != 0) or self.log_all_terms
        need_cross = (self.weights['cross'] != 0) or self.log_all_terms
        need_var   = (self.weights['var']   != 0) or self.log_all_terms
        need_kl    = (self.weights['kl']    != 0) or self.log_all_terms
        need_qnt   = (self.weights['qnt']   != 0) or self.log_all_terms

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

        if need_kl:
            kl_raw, kl_mean_raw, kl_q10_raw, kl_score01_avg_raw = self._kl_distribution_loss(predictions, targets)

        if need_qnt:
            qnt_raw, qnt_D_raw, qnt_score01_raw, _, _ = self._qnt_distribution_loss(predictions, targets)

        if not self.initialized and self.training:
            self.running_norms[0] = mae_raw.detach() + 1e-8
            self.running_norms[1] = (shape_raw.detach() + 1e-8) if need_shape else torch.tensor(1.0, device=device)
            self.running_norms[2] = (deriv_raw.detach() + 1e-8) if need_deriv else torch.tensor(1.0, device=device)
            self.running_norms[3] = (cross_raw.detach() + 1e-8) if need_cross else torch.tensor(1.0, device=device)
            self.running_norms[4] = (var_raw.detach()   + 1e-8) if need_var   else torch.tensor(1.0, device=device)
            self.running_norms[5] = (kl_raw.detach()    + 1e-8) if need_kl    else torch.tensor(1.0, device=device)
            self.running_norms[6] = (qnt_raw.detach()   + 1e-8) if need_qnt   else torch.tensor(1.0, device=device)
            self.running_norms[7] = (trj_raw.detach()   + 1e-8) if need_trj   else torch.tensor(1.0, device=device)
            self.initialized = True

        total = self.weights['mae'] * (mae_raw / self.running_norms[0])
        if self.weights['shape'] != 0:
            total = total + self.weights['shape'] * (shape_raw / self.running_norms[1])
        if self.weights['deriv'] != 0:
            total = total + self.weights['deriv'] * (deriv_raw / self.running_norms[2])
        if self.weights['cross'] != 0:
            total = total + self.weights['cross'] * (cross_raw / self.running_norms[3])
        if self.weights['var'] != 0:
            total = total + self.weights['var'] * (var_raw / self.running_norms[4])
        if self.weights['kl'] != 0:
            total = total + self.weights['kl'] * (kl_raw / self.running_norms[5])
        if self.weights['qnt'] != 0:
            total = total + self.weights['qnt'] * (qnt_raw / self.running_norms[6])

        loss_dict_norm = {
            'mae':   (mae_raw   / self.running_norms[0]),
            'shape': (shape_raw / self.running_norms[1]) if need_shape else torch.zeros((), device=device),
            'deriv': (deriv_raw / self.running_norms[2]) if need_deriv else torch.zeros((), device=device),
            'cross': (cross_raw / self.running_norms[3]) if need_cross else torch.zeros((), device=device),
            'var':   (var_raw   / self.running_norms[4]) if need_var   else torch.zeros((), device=device),
            'kl':    (kl_raw    / self.running_norms[5]) if need_kl    else torch.zeros((), device=device),
            'qnt':   (qnt_raw   / self.running_norms[6]) if need_qnt   else torch.zeros((), device=device),
            'total': total
        }

        loss_dict_raw = {
            'mae': mae_raw,
            'shape': shape_raw if need_shape else torch.zeros((), device=device),
            'deriv': deriv_raw if need_deriv else torch.zeros((), device=device),
            'cross': cross_raw if need_cross else torch.zeros((), device=device),
            'var': var_raw if need_var else torch.zeros((), device=device),

            'kl': kl_raw if need_kl else torch.zeros((), device=device),
            'kl_mean': kl_mean_raw if need_kl else torch.zeros((), device=device),
            'kl_q10': kl_q10_raw if need_kl else torch.zeros((), device=device),
            'kl_score01_avg': kl_score01_avg_raw if need_kl else torch.zeros((), device=device),

            'qnt': qnt_raw if need_qnt else torch.zeros((), device=device),
            'qnt_D': qnt_D_raw if need_qnt else torch.zeros((), device=device),
            'qnt_score01': qnt_score01_raw if need_qnt else torch.zeros((), device=device),

            'total': total,
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

            kl_weight=config.get('loss_kl_weight', 0.02),
            qnt_weight=config.get('loss_qnt_weight', 0.08),

            log_all_terms=config.get('log_all_loss_terms', True),

            kl_bins=config.get('loss_kl_bins', 33),
            kl_eps=config.get('loss_kl_eps', 1e-8),
            kl_support_low=config.get('loss_kl_support_low', 0.001),
            kl_support_high=config.get('loss_kl_support_high', 0.999),
            kl_use_q10=config.get('loss_kl_use_q10', True),
            kl_q10_weight=config.get('loss_kl_q10_weight', 0.5),

            qnt_q_lo=config.get('loss_qnt_q_lo', 0.01),
            qnt_q_hi=config.get('loss_qnt_q_hi', 0.99),
            qnt_n_q=config.get('loss_qnt_n_q', 99),
            qnt_tail_lo=config.get('loss_qnt_tail_lo', 0.10),
            qnt_tail_hi=config.get('loss_qnt_tail_hi', 0.90),
            qnt_top_q_regions=config.get('loss_qnt_top_q_regions', 0.25),
            qnt_eps=config.get('loss_qnt_eps', 1e-8),
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
                loss, loss_dict_norm, loss_dict_raw = criterion(predictions, target_batch)
                loss = loss / accumulation_steps
        else:
            predictions = run_model_forward(model, input_batch, target_batch, current_p, forward_mode)
            loss, loss_dict_norm, loss_dict_raw = criterion(predictions, target_batch)
            loss = loss / accumulation_steps

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
                f"  Val(TL rollout): Total {val_tl_loss:.4f} | MAE {val_tl_comp['mae']:.4f} | Shape {val_tl_comp['shape']:.4f} "
                f"| Deriv {val_tl_comp['deriv']:.4f} | Cross {val_tl_comp['cross']:.4f} | Var {val_tl_comp['var']:.4f}"
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
        if val_ar_comp is not None:
            print(
                f"  Val(AR rollout): Total {val_ar_loss:.4f} | MAE {val_ar_comp['mae']:.4f} | Shape {val_ar_comp['shape']:.4f} "
                f"| Deriv {val_ar_comp['deriv']:.4f} | Cross {val_ar_comp['cross']:.4f} | Var {val_ar_comp['var']:.4f}"
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



def main():
    """Main training function."""
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
        'data_path': 'data_processed/data2p_A_3_1189451_S10_F_dff_10Hz.npy', #'data_processed/data100_ba16.npy',
        'T_in': 30, #90,
        'T_out': 30, #90,
        'n_vars': 16,
        'd_model': 64,           # model_spec.d_model
        'n_heads': 8,            # model_spec.nhead
        'n_layers': 8,           # model_spec.nlayers
        'd_ff': 128,             # model_spec.d_hid (feedforward dimension)
        'dropout': 0.05,       # training_spec.dropout
        'patch_len': 1,          # Number of time-steps per patch (Timer-XL style)
        'effective_batch_size': 2000, #2000,  # Original batch size from YAML (effective via accumulation)
        'batch_size': 1024,#2048, #512,        # Physical batch size (teacher forcing is memory efficient)
        'accumulation_steps': 2, #4,  # 64 * 32 ≈ 2048 effective batch size
        'use_mixed_precision': True,  # Use FP16 to reduce memory by ~50%
        'loss_type': 'combined',
        'loss_mae_weight': 1.0,
        'loss_shape_weight': 0.0,  # High weight to force "wiggles"
        'loss_deriv_weight': 0.0,  # Moderate weight to force smooth transitions
        'loss_cross_weight': 0.0,  # Low weight to maintain population structure
        'loss_var_weight': 0.0,
        'learning_rate': 1e-4,    # training_spec.lr (will use scheduler instead)
        'max_lr': 0.0003,         # scheduler.max_lr
        'num_epochs': 150, #150 ,     # training_spec.epochs
        'train_ratio': 0.8,
        'random_seed': 101,      # seed from YAML
        'save_dir': 'checkpoints',
        'plot_dir': 'evaluation',
        'log_all_loss_terms': True,      # <-- NEW (so mae/shape/deriv/cross/var are computed even if weights=0)
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
        'use_torch_compile': True,
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
    
    train_examples, val_examples, train_seq_indices, val_seq_indices = split_by_sequences(
        examples, sequence_indices, 
        train_ratio=config['train_ratio'], 
        random_seed=config['random_seed']
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

    train_examples, val_examples, test_examples, norm = normalize_after_split_input_only(
        train_examples,
        val_examples,
        T_in=config["T_in"],
        eps=1e-6,
        save_stats_path=str(stats_npy),   # this writes _mean/_std
    )

    np.save(val_npy, val_examples)  # normalized val
    np.save(val_seq_npy, val_seq_indices.astype(np.int64, copy=False))

    config["normalization_stats_base"] = str(stats_npy.with_suffix("").resolve())
    config["processed_val_examples_path"] = str(val_npy.resolve())
    config["processed_val_seq_indices_path"] = str(val_seq_npy.resolve())

    
    # Free examples array after splitting
    del examples, sequence_indices
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Create dataloaders
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
    training_variants = [
        {'tag': 'AR_KV', 'label': 'KV Autoregressive'},
        #{'tag': 'MIX_TF_AR_KV', 'label': 'Mixed Teacher Forcing and Autoregressive'},
        #{'tag': 'AR', 'label': 'Autoregressive'},
        #{'tag': 'TF', 'label': 'Teacher Forcing'},
    ]

    
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