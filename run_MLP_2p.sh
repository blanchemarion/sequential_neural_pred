#!/usr/bin/env bash
set -euo pipefail

python src/train/train_MLP_2p.py \
  --csv_path data_raw/2p_traces/Data_Valence_FULL_A_3_1189451_Session_10_C_dec_10Hz.csv \
  --T_in 90 \
  --stride 1 \
  --batch_size 1024 \
  --epochs 30 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --d_local 64 \
  --d_pop 128 \
  --loss_beta 2.0 \
  --target_scale_quantile 0.95 \
  --output_dir checkpoints_MLP_2P

python src/infer/inference_MLP_2p.py \
  --csv_path data_raw/2p_traces/Data_Valence_FULL_A_3_1189451_Session_10_C_dec_10Hz.csv \
  --checkpoint checkpoints_MLP_2P/best_model.pt \
  --T_in 90 \
  --pred_len 720 \
  --num_sequences 10 \
  --output_dir evaluation_results/mlp_2p
