# Reproducing the widefield analysis

This repository contains the source code and the split-level score cache for the
90 observed + 720 forecast step NethoBench comparison. Large recordings and
model weights are distributed separately. Run commands from the repository root
with Python 3.11.

## Released inputs and exact versions

| Artifact | Location | Version and integrity check |
| --- | --- | --- |
| Widefield data | [Nethobench/nethobench-widefield-v1](https://huggingface.co/datasets/Nethobench/nethobench-widefield-v1) | Dataset version `1.0.0`, Git revision `6e45145353ec1b99b22984a4075187cd9bba5408`; `data/data100_ba16.npy` SHA-256 `d77a70c594ae851c969cd3242a774607238dbf37ac3e526a66f98d6685668af5` |
| Published weights | [Anonymous checkpoint archive](https://anonymous.4open.science/r/submission_checkpoints-86D5) | `MANIFEST.json` SHA-256 `96e2d74ca4562c826c1101eaa802556d8ae825b91438e54cfbf34795457f689e`; 22 `.safetensors` models |
| Four-split scores | [`reproducibility/scores_cache_90_810_4split_3seeds.json`](reproducibility/scores_cache_90_810_4split_3seeds.json) | SHA-256 `f1ed0cba4d577144ab653ceee572ae00497f3d4e5e6017646012daf7295705d4` |

The data array has shape `(287, 211, 16, 180)` with axes `(sequence,
subsequence, region, time)`. The score cache records validation seed 102,
training seeds 101–103, four sequence splits, and the forecast-only window
`[90:810)`. The analysis reports each training seed's mean over splits, then the
mean and sample SD across the three training seeds.

## Install

```bash
python -m pip install torch
python -m pip install -r requirements-repro.txt
python -m pip install -e ./nethobench
```

Choose the PyTorch build appropriate for your CPU or CUDA runtime. `nethobench/`
is part of this source repository; no additional Git checkout is needed.
The aggregation command was verified with PyTorch `2.9.0+cpu` and the direct
dependency versions in `requirements-repro.txt`. The bundled NethoBench
package reports version `0.2.0` in its `pyproject.toml`.

## Reproduce one score table and figure

```bash
python reproduce_90_810_summary.py
```

This verifies all 424 rows of the across-training-seed table against the
released reference, then writes:

```text
output/reproduced_90_810_summary/90_810_4split_3seeds_across_training_seed_summary.csv
output/reproduced_90_810_summary/bar_family_scores_90_810_4split_3seeds_across_training_seeds_std.svg
```

The input is the bundled split-level score cache, so this command reproduces
aggregation and plotting. It does **not** rerun model inference or NethoBench
metric computation. The corresponding full analysis entry point is
`src/visualize/neuro_subscores_from_npy_with_sequifier_4split_3seeds.py`;
it requires aligned prediction and ground-truth arrays under
`evaluation_results/90_810/` for all selected models and seeds.

## Obtain and verify the large artifacts

Download the pinned widefield tensor and metadata (about 0.7 GB):

```bash
python release_artifacts.py download-data
```

This creates `data_release/nethobench-widefield-v1/data/` and verifies SHA-256
before accepting either file. Add `--include-source-parquet` to retrieve the
unprepared source table as well (about 1.5 GB more). To check an existing copy:

```bash
python release_artifacts.py verify-data --root data_release/nethobench-widefield-v1
```

Open the anonymous checkpoint archive above, download its ZIP, and extract the
archive contents to `checkpoint_release/` so `checkpoint_release/MANIFEST.json`
exists. Then run:

```bash
python release_artifacts.py verify-checkpoints --root checkpoint_release
```

The manifest hash pins the exact checkpoint release. The verifier checks each
file size and SHA-256 against that manifest. The published archive contains
portable `.safetensors` weights, model configuration files, and normalization
statistics. Its README also describes original `.pt` checkpoints and a loader,
but those files are absent from the verified archive. The existing inference
entry points in `src/infer/` consume original training `.pt` files, so the
portable weights cannot be passed to those entry points directly. Training
with this source repository produces the `.pt` format used by them.

The two-photon workflows need a separate trace CSV; the widefield release does
not contain that input.

## Source revision

For a reported result, record the commit hash of this repository, the pinned
dataset revision above, the checkpoint manifest hash, the relevant training
seed and validation seed, and the forecast window. After committing this
release, use `git rev-parse HEAD` to record the code revision. The score cache
and reference table in this repository are fixed snapshots of the current
90/810 analysis.
