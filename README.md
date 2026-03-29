# Sequential Models for neural forecasting

## Scaling-law globals (single place to edit defaults)

Repository root file **`scaling_law_globals.json`** holds shared paths (`data_raw`, `data_processed`, `configs`, `checkpoints`, `predictions`, …), `prepare_data` knobs (parquet name, subsequence length, partitions, seeds), `generate_scaling_configs` (`t_in_choices`, `seeds`, `share_to_num_epochs`), the training **`base_config`** block merged under every run JSON, and inference defaults (`num_sequences`, `long_pred_length`, `n_plot_examples`).

All four pipeline scripts load it automatically. Override the file location with: `--scaling-law-globals /path/to/custom.json`.

If the file is missing, each script falls back to built-in defaults (training still uses a code fallback if `train_scaling_law.base_config` is absent).

## Scaling-law pipeline (run in order)

1. **`src/prepare/prepare_data.py`**: Read raw parquet, build `.npy` splits under `data_processed/` (and metadata) for training.

2. **`src/prepare/generate_scaling_configs.py`**: Scan `data_processed/` and write JSON run configs under `configs/` (e.g. `data_path`, `T_in`, seeds, epochs).

3. **`src/train/train_scaling_law.py`**: Train on those configs; writes checkpoints under `checkpoints/` and normalized train/val arrays (plus val sequence-id sidecars) under `data_processed/` for long-horizon inference.

4. **`src/infer/inference_scaling_law.py`**: Load checkpoints and the saved val split + `processed_val_seq_indices_*` file; write `predictions/pred_*_epoch*.npy` (stacked pred/GT arrays) and optional plots under `predictions/`.


```bash
python src/prepare/prepare_data.py
python src/prepare/generate_scaling_configs.py
python src/train/train_scaling_law.py
python src/infer/inference_scaling_law.py
```