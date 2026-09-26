# Sequential neural forecasting

Code for forecasting population neural activity from an observed time series. The analyses cover widefield calcium recordings and two photon fluorescence traces. The widefield experiments compare a causal, decoder only Transformer with linear autoregression (VAR), an autoregressive GRU (RNN), and a conditional deep Markov state space model (SSM). A population conditioned MLP provides a separate baseline for two photon traces. The repository also contains long horizon evaluation and neural realism scoring code used in the accompanying article.

This README describes the analysis code and its inputs. Reported numerical results should be taken from the article and the exact experiment outputs used to prepare its figures.

## Requirements

Use Python 3.11 and install PyTorch for your CPU or CUDA platform, then the Python dependencies:

```bash
python -m pip install torch
python -m pip install -r requirements.txt
python -m pip install -e ./nethobench
```

The last command installs the Nethob  ench package included in this repository; no separate checkout is needed. Large training and scoring runs may require a CUDA GPU and substantial memory.

## Reproduce a reported analysis figure

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the pinned widefield dataset
and checkpoint releases, SHA-256 checks, and a command that rebuilds the
90/810 family-score table and bar figure from the released split-level scores.
The short path is:

```bash
python reproduce_90_810_summary.py
```

The command verifies the 424-row score summary and writes the table and figure
under `output/reproduced_90_810_summary/`. Large data and checkpoints are
downloaded separately; the bundled score cache makes this figure reproducible
without fetching either artifact.

## Data and configuration

Widefield preparation expects a Parquet table with `sequenceId`, `itemPosition`, and one numeric column per brain region. Rows within a sequence must be ordered by time. The preparation code also reads the corresponding metadata JSON. The main preparation workflow produces an array with axes `(sequence, subsequence, region, time)` and a metadata sidecar.

The two photon MLP reads a headerless CSV matrix with **neurons in rows and time points in columns**. `src/prepare/prepare_2p_traces.py` can convert this format for the array based workflows.

`globals.json` defines input and export locations, data partitions, model defaults, training seeds, and inference horizons. Edit it for your data and experiment settings before running the workflows. The sensitivity and GRU/SSM entry points also accept `--globals` to load another configuration file; consult each script's `--help` for available overrides. Keep the configuration snapshot, source revision, input data version, and selected checkpoints with any reported result.

The default widefield settings use 90 observed steps and 90 training target steps for the main model comparison. Long horizon evaluation extends forecasts to 720 future steps for the main comparison and 810 future steps for the sensitivity workflow. These are different evaluation protocols and should be reported separately.

## Widefield workflows

Run commands from the repository root after setting the data locations in `globals.json`.

### Main model comparison

```bash
python src/prepare/prepare.py
python src/train/train_all_regimes.py
python src/train/train_var_baseline.py
python src/train/train_gru_ar.py
python src/train/train_cdmm_ssm.py
```

`train_all_regimes.py` trains the Transformer with the supervision regimes selected in that file. Its implementation uses `src/models/model_KV_cached.py`, including cached key/value attention for autoregressive rollout. The other trainers implement the linear, GRU, and cDMM SSM comparisons. GRU and cDMM SSM settings are read from their sections of `globals.json`; their command line options include input array, save location, device, and training duration. The VAR trainer has its experiment settings in the script.

Run the matching inference entry points on the saved checkpoints:

```bash
python src/infer/inference_all_regimes.py
python src/infer/inference_var_baseline.py --help
python src/infer/inference_gru_ar.py --help
python src/infer/inference_cdmm_ssm.py --help
```

The Transformer evaluator selects modes and checkpoints through constants in `inference_all_regimes.py`. The baseline evaluators expose checkpoint and evaluation options on the command line. Inference exports aligned prediction and ground truth arrays; some entry points also write tabular scores, plots, or CSV traces.

### Data and input horizon sensitivity

```bash
python src/prepare/prepare_sensitivity.py
python src/prepare/generate_configs.py
python src/train/train_sensitivity.py
python src/infer/inference_sensitivity.py
```

This workflow varies the fraction of sequences, number of regions, input horizon, and seed according to `globals.json`. The current configuration specifies sequence fractions of 100%, 50%, and 25%; 4, 8, and 16 regions; and input horizons of 30, 90, and 300 steps. Generated run specifications are read by the sensitivity trainer. Its inference step uses saved validation sequence identifiers to assemble long autoregressive forecasts.

## Two photon MLP workflow

The population conditioned MLP uses chronological train, validation, and test regions with configurable gaps. For a trace matrix in the format above:

```bash
python src/train/train_MLP_2p.py --csv_path /path/to/traces.csv --output_dir /path/to/mlp_run
python src/infer/inference_MLP_2p.py --csv_path /path/to/traces.csv --checkpoint /path/to/mlp_run/best_model.pt --output_dir /path/to/mlp_evaluation
```

The trainer defaults to a 90 step context and a 16 step training target. The inference script defaults to a 720 step recursive forecast. Use `--help` to set horizons, split gaps, and other options explicitly for a particular analysis.

## Evaluation 

The scripts in `src/visualize/` consume aligned forecast and ground truth arrays. `neuro_scoring_windows.py` defines the shared forecast window logic; score comparisons should use the same sequences and forecast window for every model.

- `neuro_subscores_from_npy_with_sequifier_4split_3seeds.py` aggregates NethoBench scores over four validation sequence splits and independent training seeds. It reports between seed variation separately from within seed split variation and computes paired ranking tests from cached scores.
- `neuro_metric_specific_visualizations_90_810.py` produces metric level diagnostic figures for the 90 observed plus 720 forecast step setting.
- `analyze_pointwise_fidelity_vs_nethobench.py` compares direct forecast fidelity with structural neural realism.
- `estimate_widefield_ceiling_floor.py` estimates reference bounds for widefield scoring.
- `matched_widefield_references.py` calculates matched widefield references for the 222 evaluation targets over the 720-step forecast window.
- `neuro_subscores_from_npy_2p_4split.py` and `neuro_subscores_from_npy_2p_4split_filtered_regions.py` evaluate two photon forecasts.
- `plot_all_configs_learning_curves.py` plots training histories across configurations.

These analyses require the corresponding aligned arrays and, where applicable, NethoBench score inputs. Script level arguments and expected array names are documented in each entry point. NethoBench computes distributional, temporal, relational, geometry, and state dynamics scores; direct fidelity measures are handled alongside them. Treat realism scores and direct pointwise errors as distinct measurements.

## Source layout

| Path | Role |
| --- | --- |
| `src/prepare/` | Widefield and two photon data preparation; sensitivity run generation |
| `src/models/` | Transformer, VAR, GRU, cDMM SSM, and MLP implementations |
| `src/train/` | Model training entry points |
| `src/infer/` | Checkpoint evaluation and autoregressive forecasting |
| `src/visualize/` | Forecast diagnostics, scoring, and figure generation |
| `src/helpers/` | Configuration and preprocessing utilities |
| `nethobench/` | NethoBench scoring package used by the current analysis scripts |

## Reproducibility

For each article figure or table, record the source revision, exact `globals.json` snapshot, input data version, model checkpoint, training and evaluation seeds, forecast window, and scoring code version. The repository contains code for several analyses, so a single default run does not reproduce every article panel. 

## License

This project is released under the [MIT License](LICENSE).
