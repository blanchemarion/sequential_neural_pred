"""Forecast-only alignment helpers for NeuroBench scoring scripts."""

from __future__ import annotations

import numpy as np


def scoring_start_for_prediction(
    pred_arr: np.ndarray,
    *,
    full_sequence_length: int = 810,
    context_steps: int = 90,
) -> int:
    """Return 90 for full context-plus-forecast arrays and 0 otherwise."""
    if pred_arr.ndim != 3:
        raise ValueError(
            f"Expected 3D prediction array [sequence,time,region], got {pred_arr.shape}"
        )
    return context_steps if pred_arr.shape[1] == full_sequence_length else 0


def align_forecast_only(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    *,
    full_sequence_length: int = 810,
    context_steps: int = 90,
) -> tuple[np.ndarray, np.ndarray]:
    """Align GT/prediction after dropping context from both when required."""
    if gt_arr.ndim != 3:
        raise ValueError(
            f"Expected 3D ground-truth array [sequence,time,region], got {gt_arr.shape}"
        )
    score_start = scoring_start_for_prediction(
        pred_arr,
        full_sequence_length=full_sequence_length,
        context_steps=context_steps,
    )
    if score_start >= gt_arr.shape[1] or score_start >= pred_arr.shape[1]:
        raise ValueError(
            f"Scoring start {score_start} is outside "
            f"GT={gt_arr.shape} or prediction={pred_arr.shape}"
        )

    n_seq = min(gt_arr.shape[0], pred_arr.shape[0])
    n_time = min(
        gt_arr.shape[1] - score_start,
        pred_arr.shape[1] - score_start,
    )
    n_reg = min(gt_arr.shape[2], pred_arr.shape[2])
    stop = score_start + n_time
    return (
        gt_arr[:n_seq, score_start:stop, :n_reg],
        pred_arr[:n_seq, score_start:stop, :n_reg],
    )


def horizon_window_for_prediction(
    pred_arr: np.ndarray,
    horizon: int,
    *,
    full_sequence_length: int = 810,
    context_steps: int = 90,
) -> tuple[int, int]:
    """Return the source-array slice for a forecast horizon."""
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    start = scoring_start_for_prediction(
        pred_arr,
        full_sequence_length=full_sequence_length,
        context_steps=context_steps,
    )
    return start, start + int(horizon)
