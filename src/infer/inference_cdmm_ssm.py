"""Context-only deterministic and stochastic inference for cDMM_SSM."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

_SRC_ROOT = Path(__file__).resolve().parent.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from helpers.globals import load_globals, repo_root, resolve_repo_relative
from infer.inference_var_baseline import (
    plot_prediction_examples,
    prepare_input_and_gt,
    save_sequences_to_neurobench_csv,
)
from models.model_cdmm_ssm import create_cdmm_ssm


def prediction_output_directory(
    output_root: Path,
    seed: int,
    context_length: int,
    target_sequence_length: int,
) -> Path:
    total_sequence_length = int(context_length) + int(target_sequence_length)
    return (
        Path(output_root)
        / f"{int(context_length)}_{total_sequence_length}"
        / f"seed_{int(seed)}"
    )


def create_inference_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    if checkpoint.get("model_class") not in (None, "ConditionalDeepMarkovSSM"):
        raise ValueError("Checkpoint does not contain a cDMM_SSM model")
    model = create_cdmm_ssm(
        n_vars=int(config["n_vars"]),
        latent_dim=int(config["latent_dim"]),
        encoder_hidden_dim=int(config["encoder_hidden_dim"]),
        posterior_hidden_dim=int(config["posterior_hidden_dim"]),
        transition_hidden_dim=int(config["transition_hidden_dim"]),
        decoder_hidden_dims=tuple(int(value) for value in config["decoder_hidden_dims"]),
        context_length=int(config["T_in"]),
        forecast_length=int(config["T_out"]),
        spectral_radius=float(config["spectral_radius"]),
        min_scale=float(config["min_scale"]),
        min_log_variance=float(config["min_log_variance"]),
        max_log_variance=float(config["max_log_variance"]),
        device=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    expected_count = config.get("parameter_count", checkpoint.get("parameter_count"))
    if expected_count is not None and model.count_parameters() != int(expected_count):
        raise ValueError("Reconstructed model parameter count does not match checkpoint")
    model.eval()
    return model, config


@torch.no_grad()
def generate_long_outputs(
    model,
    context: torch.Tensor,
    *,
    horizon: int = 720,
    num_samples: int = 6,
    generator: torch.Generator | None = None,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean (B,T,V) and samples (S,B,T,V), without target access."""
    model.eval()
    mean = model.forecast_mean_blockwise(context, horizon=horizon)
    samples = model.forecast_samples_blockwise(
        context,
        horizon=horizon,
        num_samples=num_samples,
        generator=generator,
        seed=seed,
        sample_emission = True
    )
    return mean, samples


def evaluate_long_window(
    model,
    val_examples: np.ndarray,
    val_sequence_indices: np.ndarray,
    selected_indices: np.ndarray,
    output_dir: Path,
    *,
    t_in: int,
    target_pred_length: int,
    device: torch.device,
    seed: int,
    num_samples: int,
    n_plot_examples: int,
    checkpoint_path: Path | None = None,
) -> dict[str, Path]:
    """Save existing 3-D baseline outputs plus a sample-preserving archive."""
    target_length = t_in + target_pred_length
    max_context = t_in + 50
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))

    mean_predictions = []
    sample_predictions = []
    ground_truth_rows = []
    for row, example_index in enumerate(selected_indices, start=1):
        print(f"  Sequence {row}/{len(selected_indices)}")
        input_tensor, ground_truth = prepare_input_and_gt(
            val_examples,
            val_sequence_indices,
            int(example_index),
            t_in,
            target_length,
            max_context,
        )
        input_tensor = input_tensor.to(device)
        mean_forecast, sample_forecasts = generate_long_outputs(
            model,
            input_tensor,
            horizon=target_pred_length,
            num_samples=num_samples,
            generator=generator,
        )
        full_mean = torch.cat([input_tensor, mean_forecast], dim=1)
        repeated_context = input_tensor.unsqueeze(0).expand(
            num_samples, -1, -1, -1
        )
        full_samples = torch.cat([repeated_context, sample_forecasts], dim=2)
        mean_predictions.append(full_mean.cpu().numpy()[0])
        sample_predictions.append(full_samples.cpu().numpy()[:, 0])
        ground_truth_rows.append(ground_truth)

    means = np.asarray(mean_predictions, dtype=np.float32)
    # Per-row list is (N,S,T,V); stored convention is (S,N,T,V).
    samples = np.asarray(sample_predictions, dtype=np.float32).transpose(1, 0, 2, 3)
    ground_truth = np.asarray(ground_truth_rows, dtype=np.float32)
    if means.shape != ground_truth.shape:
        raise ValueError(f"Mean/ground-truth shape mismatch: {means.shape}")
    if samples.shape != (
        num_samples,
        means.shape[0],
        means.shape[1],
        means.shape[2],
    ):
        raise ValueError(f"Unexpected sample archive shape: {samples.shape}")
    if not np.isfinite(means).all() or not np.isfinite(samples).all():
        raise ValueError("cDMM_SSM predictions contain NaN or Inf")

    config_name = f"{t_in}_{t_in + target_pred_length}"
    mean_mode = "cDMM_SSM_mean"
    sample_mode = "cDMM_SSM_samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    outputs["mean_npy"] = output_dir / (
        f"long_predictions_{config_name}_{mean_mode}.npy"
    )
    outputs["samples_npy"] = output_dir / (
        f"long_predictions_{config_name}_{sample_mode}.npy"
    )
    outputs["ground_truth_npy"] = output_dir / (
        f"long_ground_truth_{config_name}.npy"
    )
    np.save(outputs["mean_npy"], means)
    np.save(outputs["samples_npy"], samples)
    np.save(outputs["ground_truth_npy"], ground_truth)

    region_names = [f"roi_{index}" for index in range(means.shape[2])]
    mean_scored = means[:, t_in:, :]
    ground_truth_scored = ground_truth[:, t_in:, :]
    outputs["mean_csv"] = output_dir / (
        f"long_predictions_scored_{mean_mode}.csv"
    )
    outputs["ground_truth_csv"] = output_dir / "long_ground_truth_scored.csv"
    save_sequences_to_neurobench_csv(
        mean_scored, outputs["mean_csv"], region_names=region_names
    )
    save_sequences_to_neurobench_csv(
        ground_truth_scored,
        outputs["ground_truth_csv"],
        region_names=region_names,
    )

    sample_labels = []
    for sample_index in range(num_samples):
        label = f"{sample_mode}_s{sample_index + 1:02d}"
        sample_labels.append(label)
        sample_npy = output_dir / f"long_predictions_{config_name}_{label}.npy"
        sample_csv = output_dir / f"long_predictions_scored_{label}.csv"
        np.save(sample_npy, samples[sample_index])
        save_sequences_to_neurobench_csv(
            samples[sample_index, :, t_in:, :],
            sample_csv,
            region_names=region_names,
        )
        outputs[f"sample_{sample_index + 1:02d}_npy"] = sample_npy
        outputs[f"sample_{sample_index + 1:02d}_csv"] = sample_csv

    metadata = {
        "model": "cDMM_SSM",
        "dtype": "float32",
        "context_steps_in_full_arrays": int(t_in),
        "forecast_steps": int(target_pred_length),
        "mean_axes": ["sequence", "time", "region"],
        "sample_archive_axes": ["sample", "sequence", "time", "region"],
        "per_sample_axes": ["sequence", "time", "region"],
        "sample_order": sample_labels,
        "num_samples": int(num_samples),
        "seed": int(seed),
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "posterior_used_for_inference": False,
        "blockwise_protocol": {
            "block_length": int(model.forecast_length),
            "generated_blocks": int(
                np.ceil(target_pred_length / model.forecast_length)
            ),
            "sample_history": "each trajectory propagates its own generated block",
        },
    }
    outputs["metadata_json"] = output_dir / "prediction_metadata_cDMM_SSM.json"
    outputs["metadata_json"].write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    plot_prediction_examples(
        means,
        ground_truth,
        output_dir,
        mode=mean_mode,
        config_name=config_name,
        pred_start=t_in,
        seed=seed,
        n_examples=n_plot_examples,
    )
    return outputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--globals", type=Path, default=None)
    preliminary_args, _ = preliminary.parse_known_args(argv)
    full_globals = load_globals(preliminary_args.globals)
    defaults = full_globals.get("inference_cdmm_ssm", {})

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--globals", type=Path, default=None)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints_cDMM_SSM_z12_90_90") / "best_model.pt",
    )
    parser.add_argument(
        "--num-sequences", type=int, default=int(defaults.get("num_sequences", 10))
    )
    parser.add_argument(
        "--num-samples", type=int, default=int(defaults.get("num_samples", 6))
    )
    parser.add_argument("--seed", type=int, default=int(defaults.get("seed", 102)))
    parser.add_argument(
        "--long-pred-length",
        type=int,
        default=int(defaults.get("long_pred_length", 720)),
    )
    parser.add_argument(
        "--n-plot-examples",
        type=int,
        default=int(defaults.get("n_plot_examples", 10)),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(defaults.get("output_root", "evaluation_results")),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def resolve_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    root = repo_root()
    device = resolve_device(args.device)
    checkpoint_path = resolve_repo_relative(root, args.checkpoint)
    output_root = resolve_repo_relative(root, args.output_root)
    model, config = create_inference_model(checkpoint_path, device)
    t_in = int(config["T_in"])
    t_out = int(config["T_out"])
    if t_in != 90 or t_out != 90:
        raise ValueError("The common blockwise protocol requires T_in=T_out=90")

    val_path = Path(config["processed_val_examples_path"])
    sequence_path = Path(config["processed_val_seq_indices_path"])
    if not val_path.is_file():
        val_path = root / "data_processed" / val_path.name
    if not sequence_path.is_file():
        sequence_path = root / "data_processed" / sequence_path.name

    val_examples = np.load(val_path)
    val_sequence_indices = np.load(sequence_path).astype(np.int64, copy=False)
    if val_examples.ndim != 3 or val_examples.shape[1] != int(config["n_vars"]):
        raise ValueError(f"Unexpected validation tensor shape: {val_examples.shape}")
    if val_sequence_indices.shape != (val_examples.shape[0],):
        raise ValueError("Validation sequence metadata does not match examples")

    seed = int(args.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    count = min(int(args.num_sequences), len(val_examples))
    output_dir = prediction_output_directory(
        output_root, seed, t_in, int(args.long_pred_length)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir.parent / f"selected_indices_N{count}_seed{seed}.npy"
    if index_path.exists():
        selected_indices = np.load(index_path).astype(np.int64, copy=False)
    else:
        selected_indices = np.random.choice(
            len(val_examples), size=count, replace=False
        ).astype(np.int64, copy=False)
        np.save(index_path, selected_indices)

    print("=" * 80)
    print("cDMM_SSM CONTEXT-ONLY BLOCKWISE INFERENCE")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Latent dimension: {config['latent_dim']}")
    print(f"Samples per context: {args.num_samples}")
    print(f"Output: {output_dir}")
    evaluate_long_window(
        model,
        val_examples,
        val_sequence_indices,
        selected_indices,
        output_dir,
        t_in=t_in,
        target_pred_length=int(args.long_pred_length),
        device=device,
        seed=seed,
        num_samples=int(args.num_samples),
        n_plot_examples=int(args.n_plot_examples),
        checkpoint_path=checkpoint_path,
    )


if __name__ == "__main__":
    main()
