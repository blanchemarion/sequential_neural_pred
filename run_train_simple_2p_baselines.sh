#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/lyceum/blanche/sequential_neural_pred"
cd "$PROJECT_DIR"

echo "[INFO] Prepare 2p traces (raw, no transform)"
python src/prepare/prepare_2p_traces.py \
  --trace-transform identity \
  --split-mode blocked \
  --split-gap-timesteps 120

echo "[INFO] Train DECAY baseline"
python src/train/train_simple_2p_baselines.py \
  --model_type decay \
  --T_in 90 \
  --T_out_train 1 \
  --num_epochs 8 \
  --batch_size 2048 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --output_activation softplus \
  --loss_beta 2.0 \
  --target_scale_quantile 0.95 \
  --loss_max_weight 10.0

echo "[INFO] Train RVAR baseline"
python src/train/train_simple_2p_baselines.py \
  --model_type rectified_var \
  --T_in 90 \
  --T_out_train 1 \
  --K_lags 60 \
  --num_epochs 30 \
  --batch_size 2048 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --output_activation softplus \
  --loss_beta 2.0 \
  --target_scale_quantile 0.95 \
  --loss_max_weight 10.0

echo "[INFO] Train GRU baseline"
python src/train/train_simple_2p_baselines.py \
  --model_type gru \
  --T_in 90 \
  --T_out_train 1 \
  --hidden_size 128 \
  --num_layers 1 \
  --num_epochs 30 \
  --batch_size 1024 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --output_activation softplus \
  --loss_beta 2.0 \
  --target_scale_quantile 0.95 \
  --loss_max_weight 10.0

echo "[INFO] Run long-rollout inference for DECAY/RVAR/GRU"
python src/infer/inference_simple_2p_baselines.py \
  --checkpoints checkpoints_DECAY/best_model.pt checkpoints_RVAR/best_model.pt checkpoints_GRU/best_model.pt \
  --long_pred_len 420 \
  --num_sequences 10 \
  --seed 102

echo "[INFO] Done"

