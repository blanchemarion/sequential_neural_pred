# Sequential neural forecasting for widefield calcium dynamics

This repository implements **decoder-only Transformer models** and evaluation tooling for **multi-step forecasting** of population neural activity organized as multivariate time series. It targets recordings observed over many timesteps and summarized per timestep by activity in multiple brain regions or neurons.

The code supports several training–inference **regimes** (including pure autoregressive self-feedback, and teacher forcing–style supervision), **scaling analyses** over dataset size and input horizon, **linear VAR-style baselines**, and **population-conditioned MLP baselines** on dense two-photon–style traces stored as CSV. Companion **`scaling_law_globals.json`** centralizes paths and hyperparameter defaults so experiments stay reproducible and consistent across preparation, training, and inference scripts.

---

## Scientific scope

- **Task.** Predict future activity conditioned on an observed prefix of length $\(T_{\mathrm{in}}\)$. Outputs use horizon $\(T_{\mathrm{out}}\)$ (often one step during scaling-law runs; multi-step in full-regime experiments depending on configuration).
- **Data.** Wide-format tabular exports keyed by sequence and timestep identifiers (`sequenceId`, `itemPosition`), with one column per region or neuron. The preparation utilities reshape contiguous subsequences into 4D arrays suited to batched training.
- **Models.** Core neural predictor: causal Transformer encoder over concatenated history and prediction horizon (`src/models/models.py`), plus a **KV-cache variant** intended for efficient long rollouts (`src/models/model_KV_cached.py`). Auxiliary **`train_var_baseline.py`** / **`inference_var_baseline.py`** implement vector-autoregressive-style linear comparisons (`src/models/model_var_baseline.py`).
- **Two-photon–style traces.** **`train_MLP_2p.py`** and **`inference_MLP_2p.py`** train an **`MLP2P`** architecture (`src/models/mlp_2p.py`): per-neuron temporal embeddings conditioned on a learned population summary, suited to nonnegative continuous fluorescence traces supplied as matrix CSV (rows = neurons, columns = time; see script `--csv_path`).

---

## Requirements

- **Python** 3.11 recommended (see `run_scaling_law.sh` / `run_train.sh` for conda/venv bootstrap patterns).
- **PyTorch** 2.x with CUDA when available (shell helpers pin a CUDA 12.4 wheel line for reproducibility; adjust for your hardware via [pytorch.org](https://pytorch.org/get-started/locally/)).
- **`requirements.txt`** lists NumPy, Pandas, PyArrow, Matplotlib, Seaborn, tqdm, SciPy, and scikit-learn.

Install dependencies:

```bash
pip install -r requirements.txt
pip install torch  # choose CPU/GPU build appropriate for your machine
```

---

## Configuration: `scaling_law_globals.json`

The JSON file at the repository root is the **single shared configuration** for path prefixes (`paths`), parquet ingestion knobs (`prepare_data`), default Transformer hyperparameters merged into scaling-law runs (`train_scaling_law.base_config`), grid choices for generated run configs (`generate_scaling_configs`), and long-rollout inference defaults (`inference_scaling_law`).

Every pipeline stage honors **`--scaling-law-globals /path/to/custom.json`** to swap experiments without editing source. If the file is absent or incomplete, scripts merge sensible programmatic defaults (`src/helpers/scaling_law_globals.py`).

---

## Pipeline A — Scaling-law experiments (parquet → training → long inference)

Typical order:

1. **`src/prepare/prepare_scaling.py`** — ingest raw parquet + metadata, sample sequences and regions per globals, export partitioned `.npy` tensors plus JSON metadata beside them for reproducibility.
2. **`src/prepare/generate_scaling_configs.py`** — scan exported tensors and write per-run JSON specs (e.g., \(T_{\mathrm{in}}\), seeds, epochs keyed off dataset share names).
3. **`src/train/train_scaling_law.py`** — enumerate configs (defaults to JSON/YAML under the configured configs directory), train with **`CombinedLoss`** (MAE core plus optional distributional/shape terms when weighted), save checkpoints and normalized validation tensors plus sequence-ID sidecars for stitching long horizons.
4. **`src/infer/inference_scaling_law.py`** — load checkpoints and matched processed splits; run long autoregressive forecasts (length set via globals); stack predictions and aligned ground-truth into `.npy` archives suitable for downstream metrics/plots.

A bundled Bash driver **`run_scaling_law.sh`** mirrors this sequence (environment creation, PyTorch install, then the four Python stages).

Optional **`src/prepare/sample_parquet_1pct.py`** subsamples sequences (~1% rows by sequence ID) for fast debugging without altering downstream filenames unexpectedly—adjust globals/metadata accordingly before serious runs.

---

## Pipeline B — Multi-regime Transformer training (`prepare.py` → `train_all_regimes.py`)

For richer supervisory mixtures (teacher forcing, autoregressive self-feedback, one-shot blocks—see docstrings in `src/models/models.py`):

```bash
python src/prepare/prepare.py
python src/train/train_all_regimes.py
```

Defaults inside **`prepare.py`** target tensors aligned with `train_all_regimes.py` expectations (history/future lengths and brain-region cardinality documented at module top). Use **`run_train.sh`** on POSIX systems if you want conda/venv setup echoed from the scaling-law driver pattern.

---

## Evaluation and visualization

- **`src/infer/inference_all_regimes.py`** — short-window versus long-window autoregressive evaluation, qualitative overlays, and metric summaries tuned inside the script (modes list and horizons).
- **`src/visualize/plot_all_configs_learning_curves.py`** — aggregate learning curves across exported histories/checkpoints.
- **`src/visualize/neuro_metric_specific_visualizations_90_810.py`** / **`neuro_subscores_from_npy_with_sequifier_4split.py`** — neuroscience-oriented dashboards from stacked prediction arrays.
- **`src/visualize/neuro_subscores_from_npy_2p_4split.py`** — analogous tooling tuned for two-photon benchmark layouts referenced inside that script.

Point these utilities at **your exported prediction tensors** produced by the inference scripts (layouts described in each file’s module docstring).

---

## Two-photon CSV baseline (`MLP2P`)

The **`train_MLP_2p.py`** script trains directly from CSV traces using blocked chronological splits (train / validation / test gaps configurable via CLI). Training checkpoints feed **`inference_MLP_2p.py`** for multi-step recursive rollout (`pred_len`, stride-driven contexts).

Example skeleton:

```bash
python src/train/train_MLP_2p.py \
  --csv_path path/to/traces.csv \
  --T_in 90 \
  --epochs 30 \
  --output_dir path/to/checkpoints

python src/infer/inference_MLP_2p.py \
  --csv_path path/to/traces.csv \
  --checkpoint path/to/checkpoints/best_model.pt \
  --T_in 90 \
  --pred_len 720 \
  --output_dir path/to/eval_exports
```

**`src/prepare/prepare_2p_traces.py`** converts the same CSV layout into `.npy` archives aligned with the widefield tensor conventions when traces must enter the Transformer dataloaders; see its module docstring for matrix orientation and output naming.

---

## Repository layout (tracked sources)

```text
src/
  helpers/           # globals loader, preprocessing, dataloading splits / normalization
  models/            # Transformer core, KV-cached variant, VAR baseline, MLP2P
  prepare/           # parquet scaling pipeline, regime preprocessing, utilities
  train/             # scaling-law trainer, multi-regime trainer, baselines
  infer/             # scaling-law rollout engine, regime evaluator, baseline/MLP2P inference
  visualize/         # publication-style plots and neuro metric summaries
scaling_law_globals.json
requirements.txt
run_scaling_law.sh
run_train.sh
```

---

## Reproducibility notes for supplementary material

- **Seeds** appear in globals (`prepare_data`, `train_scaling_law.base_config`) and per-run JSON emitted by `generate_scaling_configs.py`.
- **Normalization** follows split-aware routines in `src/helpers/preprocess_helpers.py` (input-only normalization for scaling-law training; matching transforms carried into saved validation tensors for inference).
- **Hardware variance.** Mixed precision and optional `torch.compile` toggles live in training configs; disable if you require deterministic CPU-only traces.
- **Long-horizon stitching.** Scaling-law inference relies on sequence-ID sidecars saved during training so validation rollouts concatenate subsequences belonging to the same underlying recording.

When citing this artifact alongside a camera-ready submission, reference the paper title/authors as specified in your proceedings entry and mention the Git commit hash and the exact `scaling_law_globals.json` snapshot used for each reported figure.

---

## License

This project is released under the [MIT License](LICENSE).
