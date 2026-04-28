"""
Simple baselines for continuous nonnegative 2p spike-like forecasting.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _apply_nonnegative(x: torch.Tensor, output_activation: str) -> torch.Tensor:
    act = str(output_activation).lower()
    if act == "softplus":
        return F.softplus(x)
    if act == "clamp":
        return x.clamp_min(0.0)
    if act == "relu":
        return F.relu(x)
    if act == "identity":
        return x
    raise ValueError(f"Unknown output_activation: {output_activation}")


class ExpDecayBaseline(nn.Module):
    """
    pred_next = alpha * y_last + bias
    alpha is per-neuron and constrained to [0, 1] via sigmoid.
    """

    def __init__(self, n_vars: int, output_activation: str = "softplus"):
        super().__init__()
        self.n_vars = int(n_vars)
        self.output_activation = output_activation
        self.alpha_logit = nn.Parameter(torch.zeros(self.n_vars))
        self.bias = nn.Parameter(torch.zeros(self.n_vars))

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        # context: (B, T_in, n_vars)
        y_last = context[:, -1, :]  # (B, n_vars)
        alpha = torch.sigmoid(self.alpha_logit).unsqueeze(0)  # (1, n_vars)
        pred_next = alpha * y_last + self.bias.unsqueeze(0)
        pred_next = _apply_nonnegative(pred_next, self.output_activation)
        return pred_next.unsqueeze(1)  # (B, 1, n_vars)


class RectifiedVARBaseline(nn.Module):
    """
    Linear autoregressive forecaster over last K lags across all neurons.
    """

    def __init__(
        self,
        n_vars: int,
        T_in: int,
        K_lags: int | None = None,
        dropout: float = 0.0,
        output_activation: str = "softplus",
    ):
        super().__init__()
        self.n_vars = int(n_vars)
        self.T_in = int(T_in)
        self.K_lags = int(K_lags) if K_lags is not None else int(T_in)
        if self.K_lags <= 0 or self.K_lags > self.T_in:
            raise ValueError(f"K_lags must be in [1, T_in], got K_lags={self.K_lags}, T_in={self.T_in}")
        self.output_activation = output_activation

        in_dim = self.K_lags * self.n_vars
        self.dropout = nn.Dropout(float(dropout))
        self.linear = nn.Linear(in_dim, self.n_vars)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        # context: (B, T_in, n_vars)
        x = context[:, -self.K_lags :, :].reshape(context.shape[0], -1)
        x = self.dropout(x)
        pred_next = self.linear(x)
        pred_next = _apply_nonnegative(pred_next, self.output_activation)
        return pred_next.unsqueeze(1)  # (B, 1, n_vars)


class TinyGRUForecaster(nn.Module):
    """
    Tiny recurrent one-step forecaster.
    """

    def __init__(
        self,
        n_vars: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        output_activation: str = "softplus",
    ):
        super().__init__()
        self.n_vars = int(n_vars)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.output_activation = output_activation

        self.gru = nn.GRU(
            input_size=self.n_vars,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=float(dropout) if self.num_layers > 1 else 0.0,
        )
        self.out = nn.Linear(self.hidden_size, self.n_vars)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        # context: (B, T_in, n_vars)
        h, _ = self.gru(context)
        last_h = h[:, -1, :]  # (B, hidden_size)
        pred_next = self.out(last_h)
        pred_next = _apply_nonnegative(pred_next, self.output_activation)
        return pred_next.unsqueeze(1)  # (B, 1, n_vars)


@torch.no_grad()
def rollout_model(model: nn.Module, initial_context: torch.Tensor, pred_len: int, T_in: int) -> torch.Tensor:
    """
    Rollout one-step model autoregressively.
    Returns shape (B, T_in + pred_len, n_vars).
    """
    model.eval()
    seq = initial_context
    current = initial_context
    for _ in range(int(pred_len)):
        pred_next = model(current).clamp_min(0.0)  # final safety clamp
        seq = torch.cat([seq, pred_next], dim=1)
        current = seq[:, -int(T_in) :, :]
    return seq

