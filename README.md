# Sequential neural forecasting for widefield calcium dynamics

This repository implements **decoder-only Transformer models** and evaluation tooling for **multi-step forecasting** of population neural activity organized as multivariate time series. It targets recordings observed over many timesteps and summarized per timestep by activity in multiple brain regions or neurons.

The code supports several training–inference **regimes** (including pure autoregressive self-feedback, and teacher forcing–style supervision), **sensitivity analyses** over dataset size, brain-region cardinality, and input horizon, **linear VAR-style baselines**, and **population-conditioned MLP baselines** on dense two-photon–style traces stored as CSV. The repository-root file **`globals.json`** centralizes paths and hyperparameter defaults so experiments stay reproducible and consistent across preparation, training, and inference scripts.

---

## Scientific scope

- **Task.** Predict future activity conditioned on an observed prefix of length \(T_{\mathrm{in}}\). Outputs use horizon \(T_{\mathrm{out}}\).
- **Data.** Wide-format tabular exports keyed by sequence and timestep identifiers (`sequenceId`, `itemPosition`), with one column per region or neuron. The preparation utilities reshape contiguous subsequences into 4D arrays suited to batched training.
- **Models.** Core neural predictor: causal Transformer encoder over concatenated history and prediction horizon (`src/models/models.py`), with **KV-cache** optimization for efficient long rollouts (`src/models/model_KV_cached.py`). Auxiliary **`train_var_baseline.py`** / **`inference_var_baseline.py`** implement vector-autoregressive-style linear comparisons (`src/models/model_var_baseline.py`).
- **Two-photon–style traces.** **`train_MLP_2p.py`** and **`inference_MLP_2p.py`** train an **`MLP2P`** architecture (`src/models/mlp_2p.py`): per-neuron temporal embeddings conditioned on a learned population summary, suited to nonnegative continuous fluorescence traces supplied as matrix CSV (rows = neurons, columns = time; see script `--csv_path`).

---

## Requirements

- **Python** 3.11 recommended (see `run_sensitity.sh` / `run_train.sh` for conda/venv bootstrap patterns).
- **PyTorch** 2.x with CUDA when available (shell helpers pin a CUDA 12.4 wheel line for reproducibility; adjust for your hardware via [pytorch.org](https://pytorch.org/get-started/locally/)).
- **`requirements.txt`** lists NumPy, Pandas, PyArrow, Matplotlib, Seaborn, tqdm, SciPy, and scikit-learn.

Install dependencies:

```bash
pip install -r requirements.txt
pip install torch  # choose CPU/GPU build appropriate for your machine
```

---

## Configuration: `globals.json`

The JSON file at the repository root is the **single shared configuration**, with sections:

- **`paths`** — directory prefixes for raw/processed data, configs, checkpoints, predictions, evaluation outputs.
- **`prepare_data`** — parquet ingestion knobs (filename, partitions over data share and brain-region cardinality, subsequence length, RNG seeds).
- **`train_sensitivity.base_config`** — default Transformer hyperparameters merged under every run config (architecture, optimizer, scheduler, loss weights, checkpointing).
- **`generate_configs`** — grid choices for sensitivity sweeps (`t_in_choices`, seeds, `share_to_num_epochs`).
- **`inference_sensitivity`** — long-rollout defaults (number of validation sequences, autoregressive horizon, number of plot examples).

Every pipeline stage honors **`--globals /path/to/custom.json`** to swap experiments without editing source. If the file is absent or incomplete, scripts merge sensible programmatic defaults (`src/helpers/globals.py`).

---

## Pipeline A — Sensitivity analysis (parquet → training → long inference)

This pipeline produces controlled sweeps over dataset share, number of brain regions, input horizon \(T_{\mathrm{in}}\), and random seed, enabling **sensitivity analyses** of forecasting quality with respect to those axes.

Typical order:

1. **`src/prepare/prepare_sensitivity.py`** — ingest raw parquet + metadata, sample sequences and regions per globals, export partitioned `.npy` tensors plus JSON metadata beside them for reproducibility.
2. **`src/prepare/generate_configs.py`** — scan exported tensors and write per-run JSON specs (e.g., \(T_{\mathrm{in}}\), seeds, epochs keyed off dataset share names).
3. **`src/train/train_sensitivity.py`** — enumerate configs (defaults to JSON/YAML under the configured configs directory), train with **`CombinedLoss`** (MAE core plus optional distributional/shape terms when weighted), save checkpoints and normalized validation tensors plus sequence-ID sidecars for stitching long horizons.
4. **`src/infer/inference_sensitivity.py`** — load checkpoints and matched processed splits; run long autoregressive forecasts (length set via globals); stack predictions and aligned ground-truth into `.npy` archives suitable for downstream metrics/plots.

A bundled Bash driver **`run_sensitity.sh`** mirrors this sequence (environment creation, PyTorch install, then the four Python stages).

Optional **`src/prepare/sample_parquet_1pct.py`** subsamples sequences (~1% rows by sequence ID) for fast debugging without altering downstream filenames unexpectedly—adjust globals/metadata accordingly before serious runs.

---

## Pipeline B — Multi-regime Transformer training (`prepare.py` → `train_all_regimes.py`)

For richer supervisory mixtures (teacher forcing, autoregressive self-feedback—see docstrings in `src/models/model_KV_cached.py`):

```bash
python src/prepare/prepare.py
python src/train/train_all_regimes.py
```

Defaults inside **`prepare.py`** target tensors aligned with `train_all_regimes.py` expectations (history/future lengths and brain-region cardinality documented at module top). Use **`run_train.sh`** on POSIX systems if you want the same conda/venv bootstrap as the sensitivity-analysis driver.

---

## Evaluation and visualization

**Nethobench dependency.** Neuro visualization and scoring scripts import the **`nethobench`** Python package (`compute_neuro_scores`, neuro pipeline helpers). This repository **expects a vendored copy at the repository root**: clone or copy the Nethobench sources into **`nethobench/`** next to `src/` and `globals.json` (setuptools layout so imports resolve from `<repo>/nethobench`). Scripts under **`src/visualize/`** prepend that directory to `sys.path`, so a separate **`pip install`** is not required. Alternatively you may **`pip install -e ./nethobench`** into your environment and rely on the normal import path.

Install **`requirements.txt`** (includes `umap-learn` and `ripser` used by full neuro composites).

- **`src/infer/inference_all_regimes.py`** — short-window versus long-window autoregressive evaluation, qualitative overlays, and metric summaries tuned inside the script (modes list and horizons).
- **`src/visualize/plot_all_configs_learning_curves.py`** — aggregate learning curves across exported histories/checkpoints.
- **`src/visualize/neuro_metric_specific_visualizations_90_810.py`** / **`neuro_subscores_from_npy_with_sequifier_4split.py`** — neuroscience-oriented dashboards from stacked prediction arrays; official **`compute_neuro_scores`** paths write CSV family/submetric tables where enabled.
- **`src/visualize/neuro_subscores_from_npy_2p_4split.py`** — analogous tooling tuned for two-photon benchmark layouts referenced inside that script.

**Tensor paths.** Point these utilities at **your** stacked `.npy` exports (layouts in each script’s docstring). Defaults target **`evaluation_results/<layout>/seed_102/`** at the repo root; override with each script’s CLI or constants. Figures and caches go under **`output/`** (outside `src/`), not under `nethobench/`.

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
  prepare/           # parquet sensitivity-analysis pipeline, regime preprocessing, utilities
  train/             # sensitivity-analysis trainer, multi-regime trainer, baselines
  infer/             # sensitivity-analysis rollout engine, regime evaluator, baseline/MLP2P inference
  visualize/         # publication-style plots and neuro metric summaries
nethobench/            # vendored Nethobench package (neuro benchmark scoring)
globals.json
requirements.txt
run_sensitity.sh
run_train.sh
run_MLP_2p.sh
```

---

## Reproducibility notes for supplementary material

- **Seeds** appear in globals (`prepare_data`, `train_sensitivity.base_config`) and per-run JSON emitted by `generate_configs.py`.
- **Normalization** follows split-aware routines in `src/helpers/preprocess_helpers.py` (input-only normalization during sensitivity-analysis training; matching transforms carried into saved validation tensors for inference).
- **Hardware variance.** Mixed precision and optional `torch.compile` toggles live in training configs; disable if you require deterministic CPU-only traces.
- **Long-horizon stitching.** Sensitivity-analysis inference relies on sequence-ID sidecars saved during training so validation rollouts concatenate subsequences belonging to the same underlying recording.

When citing this artifact alongside a camera-ready submission, reference the paper title/authors as specified in your proceedings entry and mention the Git commit hash and the exact `globals.json` snapshot used for each reported figure.

---

## License

This project is released under the [MIT License](LICENSE).
