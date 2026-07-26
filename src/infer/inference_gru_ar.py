"""Blockwise autoregressive inference for GRU_AR checkpoints."""

from __future__ import annotations

import argparse
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
from models.model_gru_ar import create_gru_ar


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
    if checkpoint.get("model_class") not in (None, "GRUAR"):
        raise ValueError(
            f"Checkpoint model_class is {checkpoint.get('model_class')!r}, not 'GRUAR'"
        )

    model = create_gru_ar(
        n_vars=int(config["n_vars"]),
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
        context_length=int(config["T_in"]),
        forecast_length=int(config["T_out"]),
        device=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    expected_count = config.get("parameter_count", checkpoint.get("parameter_count"))
    if expected_count is not None and model.count_parameters() != int(expected_count):
        raise ValueError(
            f"Checkpoint parameter count {expected_count} does not match "
            f"reconstructed model {model.count_parameters()}"
        )
    model.eval()
    return model, config


def generate_long_forecast(
    model,
    initial_context: torch.Tensor,
    horizon: int = 720,
) -> torch.Tensor:
    """Return forecast samples only, using the repository's 90-step block protocol."""
    model.eval()
    with torch.no_grad():
        return model.forecast_blockwise(
            initial_context,
            horizon=int(horizon),
            block_size=int(model.forecast_length),
        )


def evaluate_long_window(
    model,
    val_examples: np.ndarray,
    val_sequence_indices: np.ndarray,
    selected_indices: np.ndarray,
    output_dir: Path,
    t_in: int,
    target_pred_length: int,
    device: torch.device,
    seed: int,
    n_plot_examples: int,
) -> None:
    """Save the same full arrays and scored CSV layout as existing baselines."""
    target_length = t_in + target_pred_length
    max_context = t_in + 50
    all_predictions = []
    all_ground_truth = []

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
        forecast = generate_long_forecast(model, input_tensor, target_pred_length)
        full_prediction = torch.cat([input_tensor, forecast], dim=1)
        all_predictions.append(full_prediction.detach().cpu().numpy()[0])
        all_ground_truth.append(ground_truth)

    predictions = np.asarray(all_predictions, dtype=np.float32)
    ground_truth = np.asarray(all_ground_truth, dtype=np.float32)
    if predictions.shape != ground_truth.shape:
        raise ValueError(
            f"Prediction/ground-truth shape mismatch: {predictions.shape} vs "
            f"{ground_truth.shape}"
        )
    if not np.isfinite(predictions).all():
        raise ValueError("GRU_AR predictions contain NaN or Inf")

    config_name = f"{t_in}_{t_in + target_pred_length}"
    mode = "GRU_AR"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        output_dir / f"long_predictions_{config_name}_{mode}.npy",
        predictions,
    )
    np.save(output_dir / f"long_ground_truth_{config_name}.npy", ground_truth)

    scored_predictions = predictions[:, t_in:, :]
    scored_ground_truth = ground_truth[:, t_in:, :]
    region_names = [f"roi_{index}" for index in range(scored_predictions.shape[2])]
    save_sequences_to_neurobench_csv(
        scored_predictions,
        output_dir / f"long_predictions_scored_{mode}.csv",
        region_names=region_names,
    )
    save_sequences_to_neurobench_csv(
        scored_ground_truth,
        output_dir / "long_ground_truth_scored.csv",
        region_names=region_names,
    )
    plot_prediction_examples(
        predictions,
        ground_truth,
        output_dir,
        mode=mode,
        config_name=config_name,
        pred_start=t_in,
        seed=seed,
        n_examples=n_plot_examples,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--globals", type=Path, default=None)
    preliminary_args, _ = preliminary.parse_known_args(argv)
    full_globals = load_globals(preliminary_args.globals)
    defaults = full_globals.get("inference_gru_ar", {})
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--globals", type=Path, default=None)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints_GRU_AR_90_90_seed102") / "best_model.pt",
    )
    parser.add_argument(
        "--num-sequences", type=int, default=int(defaults.get("num_sequences", 200))
    )
    parser.add_argument(
        "--evaluation-seed",
        "--seed",
        dest="evaluation_seed",
        type=int,
        default=int(defaults.get("evaluation_seed", defaults.get("seed", 102))),
        help="Fixed seed for evaluation-row sampling",
    )
    parser.add_argument(
        "--long-pred-length",
        type=int,
        default=int(defaults.get("long_pred_length", 720)),
    )
    parser.add_argument(
        "--n-plot-examples",
        type=int,
        default=int(defaults.get("n_plot_examples", 1)),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(defaults.get("output_root", "evaluation_results")),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def _device_from_arg(choice: str) -> torch.device:
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
    device = _device_from_arg(args.device)
    checkpoint_path = resolve_repo_relative(root, args.checkpoint)
    output_root = resolve_repo_relative(root, args.output_root)

    model, config = create_inference_model(checkpoint_path, device)
    t_in = int(config["T_in"])
    t_out = int(config["T_out"])
    if t_in != 90 or t_out != 90:
        raise ValueError(
            f"The required blockwise protocol expects T_in=T_out=90, got {t_in}/{t_out}"
        )

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
        raise ValueError("Validation sequence metadata does not match validation examples")

    seed = int(args.evaluation_seed)
    split_seed = int(config.get("split_seed", config.get("random_seed", 101)))
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
    index_path = output_dir.parent / (
        f"selected_indices_N{count}_splitseed{split_seed}_evalseed{seed}.npy"
    )
    if index_path.exists():
        selected_indices = np.load(index_path).astype(np.int64, copy=False)
    else:
        selected_indices = np.random.choice(
            len(val_examples), size=count, replace=False
        ).astype(np.int64, copy=False)
        np.save(index_path, selected_indices)

    print("=" * 80)
    print("GRU_AR BLOCKWISE AUTOREGRESSIVE INFERENCE")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Split seed: {split_seed}")
    print(f"Evaluation seed: {seed}")
    print(f"Forecast: {args.long_pred_length} steps in {t_out}-step blocks")
    print(f"Output: {output_dir}")
    evaluate_long_window(
        model,
        val_examples,
        val_sequence_indices,
        selected_indices,
        output_dir,
        t_in,
        int(args.long_pred_length),
        device,
        seed,
        int(args.n_plot_examples),
    )


if __name__ == "__main__":
    main()
