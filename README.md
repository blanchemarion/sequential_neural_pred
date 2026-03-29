# Sequential Models for neural forecasting

## Scaling-law pipeline (run in order)

1. **`src/prepare/prepare_data.py`**: Read raw parquet, build `.npy` splits under `data_processed/` (and metadata) for training.

2. **`src/prepare/generate_scaling_configs.py`**: Scan `data_processed/` and write JSON run configs under `configs/` (e.g. `data_path`, `T_in`, seeds, epochs).

3. **`src/train/train_scaling_law.py`**: Train on those configs; writes checkpoints under `checkpoints/` and normalized train/val arrays (plus val sequence-id sidecars) under `data_processed/` for long-horizon inference.

4. **`src/infer/inference_scaling_law.py`**: Load checkpoints and the saved val split + `processed_val_seq_indices_*` file; write CSVs (and optional plots) under `predictions/`.


```bash
python src/prepare/prepare_data.py
python src/prepare/generate_scaling_configs.py
python src/train/train_scaling_law.py
python src/infer/inference_scaling_law.py
```