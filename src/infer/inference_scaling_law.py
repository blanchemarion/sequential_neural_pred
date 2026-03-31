"""
Long-horizon inference for scaling-law checkpoints.

For each ``checkpoints_*_data*_ba*_Tin*_seed*_epoch*.pt`` file, loads the matching
``processed_val_config_{same_middle}.npy`` (normalized val split from training),
``processed_val_seq_indices_config_{same_middle}.npy`` (logical sequence id per val row,
required so long-horizon GT concatenates all subsequences of the same recording), and
``configs/config_{middle}.json`` (region names), runs long autoregressive rollouts, and
writes ``predictions/pred_{middle}_epoch{N}.npy`` — a single float32 array of shape
``(2, n_seq, T, V)`` where ``[0]`` is predictions and ``[1]`` is ground truth (**forecast
only**: ``LONG_PRED_LENGTH`` steps per sequence, **excluding** the initial ``T_in`` context).
Region order matches ``configs/config_{middle}.json`` ``region_names``.
Optional PNGs: ``--plot-examples``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Allow `python src/infer/inference_scaling_law.py` (no package context for relative imports).
_SRC_ROOT = Path(__file__).resolve().parent.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from helpers.scaling_law_globals import (
    load_scaling_law_globals,
    merge_inference_scaling_law_section,
    merge_paths_section,
    resolve_repo_relative,
)
from models.model_KV_cached import create_model_cached

_inf_defaults = merge_inference_scaling_law_section(load_scaling_law_globals())
NUM_SEQUENCES = int(_inf_defaults["num_sequences"])
LONG_PRED_LENGTH = int(_inf_defaults["long_pred_length"])
N_FORECAST_EXAMPLE_PLOTS = int(_inf_defaults["n_plot_examples"])


def scaling_law_project_root() -> Path:
    """Repository root (parent of ``src/``); scripts live under ``src/infer/`` or ``src/train/``."""
    return Path(__file__).resolve().parents[2]


def parse_scaling_checkpoint_stem(stem: str) -> tuple[str, int] | None:
    """
    From ``checkpoints_AR_KV_data25_ba2_Tin30_seed101_epoch1`` return
    (``data25_ba2_Tin30_seed101``, 1).
    """
    if "_epoch" not in stem:
        return None
    head, ep = stem.rsplit("_epoch", 1)
    if not ep.isdigit():
        return None
    m = re.search(r"(data\d+_ba\d+_Tin\d+_seed\d+)$", head)
    if not m:
        return None
    return m.group(1), int(ep)


def processed_val_npy_path(middle: str) -> Path:
    return scaling_law_project_root() / "data_processed" / f"processed_val_config_{middle}.npy"


def processed_val_seq_indices_npy_path(middle: str) -> Path:
    return (
        scaling_law_project_root()
        / "data_processed"
        / f"processed_val_seq_indices_config_{middle}.npy"
    )


def resolve_val_seq_indices_path(middle: str, ckpt_config: dict | None) -> Path:
    """Prefer canonical sidecar next to val; fall back to path stored in training checkpoint."""
    p = processed_val_seq_indices_npy_path(middle)
    if p.is_file():
        return p
    if ckpt_config:
        alt = ckpt_config.get("processed_val_seq_indices_path")
        if alt:
            alt_p = Path(alt)
            if alt_p.is_file():
                return alt_p
    raise FileNotFoundError(
        f"Missing val sequence-id array (re-run training with save_processed_split_examples): "
        f"expected {p}"
    )


def run_config_json_path(middle: str) -> Path:
    return scaling_law_project_root() / "configs" / f"config_{middle}.json"


def load_run_config(middle: str) -> dict:
    p = run_config_json_path(middle)
    if not p.is_file():
        raise FileNotFoundError(f"Run config not found: {p}")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def resize_pos_embedding_1d(old_pos: torch.Tensor, new_len: int) -> torch.Tensor:
    assert old_pos.ndim == 3 and old_pos.shape[0] == 1
    old_len = old_pos.shape[1]
    if new_len == old_len:
        return old_pos
    x = old_pos.permute(0, 2, 1)
    x = F.interpolate(x, size=new_len, mode="linear", align_corners=False)
    return x.permute(0, 2, 1)


def strip_compile_prefix(state_dict):
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        return {k[len("_orig_mod."):]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint_with_resized_pos(model, checkpoint, device="cpu"):
    sd = checkpoint["model_state_dict"]
    sd = strip_compile_prefix(sd)
    if hasattr(model, "pos_embedding") and "pos_embedding" in sd:
        old_pos = sd["pos_embedding"]
        new_pos = model.pos_embedding
        if old_pos.shape != new_pos.shape:
            print(f"[pos_embedding] resizing {tuple(old_pos.shape)} -> {tuple(new_pos.shape)}")
            sd["pos_embedding"] = resize_pos_embedding_1d(
                old_pos.to(new_pos.device, new_pos.dtype),
                new_pos.shape[1],
            )
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)
    model.eval()
    return checkpoint


def create_inference_model(checkpoint_path: Path, T_in: int, T_out: int, device):
    print(f"  Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    model = create_model_cached(
        n_vars=cfg["n_vars"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
        T_in=T_in,
        T_out=T_out,
        device=device,
    )
    load_checkpoint_with_resized_pos(model, checkpoint, device=device)
    model.eval()
    return model, cfg


def generate_long_sequence(model, initial_input, target_length, T_in, T_out, device):
    model.eval()
    current_sequence = initial_input.to(device)
    block_offset = 0
    with torch.no_grad():
        while current_sequence.shape[1] < target_length:
            if current_sequence.shape[1] >= T_in:
                input_chunk = current_sequence[:, -T_in:, :]
            else:
                pad_len = T_in - current_sequence.shape[1]
                pad = current_sequence[:, :1, :].repeat(1, pad_len, 1)
                input_chunk = torch.cat([pad, current_sequence], dim=1)
            pred = model.forward_autoregressive_kvcache(input_chunk, block_offset=block_offset)
            current_sequence = torch.cat([current_sequence, pred], dim=1)
    return current_sequence[:, :target_length, :]


def prepare_input_and_gt(val_examples, val_seq_indices, example_idx, T_in, target_length, MAX_CONTEXT):
    """
    val_examples: (N, n_vars, n_time) as saved by training.
    val_seq_indices: length N — logical sequence id for each val row (same id for all
    subsequences from one recording). Rows with the same id are concatenated in row order.
    """
    first_example = val_examples[example_idx].T
    seq_id = int(val_seq_indices[example_idx])

    if first_example.shape[0] >= MAX_CONTEXT:
        base_input = first_example[:MAX_CONTEXT, :]
    else:
        pad_len = MAX_CONTEXT - first_example.shape[0]
        base_input = np.concatenate(
            [first_example, np.repeat(first_example[-1:, :], pad_len, axis=0)],
            axis=0,
        )

    input_sequence = base_input[-T_in:, :]
    gt_parts = [input_sequence]
    first_example_future = first_example[MAX_CONTEXT:, :]
    if first_example_future.size > 0:
        gt_parts.append(first_example_future)

    same_sequence_mask = val_seq_indices == seq_id
    same_sequence_indices = np.where(same_sequence_mask)[0]
    sorted_seq_indices = sorted(same_sequence_indices.tolist())
    start_pos = sorted_seq_indices.index(int(example_idx))
    used_indices = sorted_seq_indices[start_pos + 1 :]

    for seq_example_idx in used_indices:
        seq_example = val_examples[seq_example_idx].T
        gt_parts.append(seq_example)

    gt_sequence = np.concatenate(gt_parts, axis=0)

    if gt_sequence.shape[0] >= target_length:
        gt_sequence = gt_sequence[:target_length, :]
    else:
        remaining = target_length - gt_sequence.shape[0]
        gt_sequence = np.concatenate(
            [gt_sequence, np.repeat(gt_sequence[-1:, :], remaining, axis=0)],
            axis=0,
        )

    input_tensor = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0)
    return input_tensor, gt_sequence


def collect_long_predictions(
    model,
    val_examples: np.ndarray,
    val_seq_indices: np.ndarray,
    selected_indices: np.ndarray,
    T_in: int,
    T_out: int,
    target_pred_length: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    MAX_CONTEXT = T_in + 50
    target_length = T_in + target_pred_length
    preds = []
    gts = []
    for k, example_idx in enumerate(selected_indices):
        print(f"    sequence {k + 1}/{len(selected_indices)} (val idx {example_idx})")
        input_tensor, gt_sequence = prepare_input_and_gt(
            val_examples,
            val_seq_indices,
            int(example_idx),
            T_in,
            target_length,
            MAX_CONTEXT,
        )
        input_tensor = input_tensor.to(device)
        long_pred = generate_long_sequence(
            model, input_tensor, target_length, T_in, T_out, device
        )
        preds.append(long_pred.detach().cpu().numpy()[0])
        gts.append(gt_sequence)
    P = np.asarray(preds, dtype=np.float32)
    G = np.asarray(gts, dtype=np.float32)
    min_n = min(P.shape[0], G.shape[0])
    min_t = min(P.shape[1], G.shape[1])
    min_v = min(P.shape[2], G.shape[2])
    return P[:min_n, :min_t, :min_v], G[:min_n, :min_t, :min_v]


def save_pred_gt_npy(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    region_names: list[str],
    out_path: Path,
) -> Path:
    """
    Write one float32 array ``(2, n_seq, T, V)``: stack[0] = pred, stack[1] = gt.
    Channel order matches ``region_names`` (not embedded in the file).
    """
    n_seq, T, V = predictions.shape
    if ground_truth.shape != predictions.shape:
        raise ValueError(f"Shape mismatch pred {predictions.shape} vs gt {ground_truth.shape}")
    if len(region_names) != V:
        raise ValueError(f"region_names length {len(region_names)} != V={V}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stacked = np.stack(
        [predictions.astype(np.float32, copy=False), ground_truth.astype(np.float32, copy=False)],
        axis=0,
    )
    np.save(out_path, stacked)
    return out_path


def plot_random_forecast_examples(
    pred: np.ndarray,
    gt: np.ndarray,
    region_names: list[str],
    output_base: Path,
    *,
    rng: np.random.Generator,
    n_examples: int = N_FORECAST_EXAMPLE_PLOTS,
) -> list[Path]:
    """
    Save PNGs for ``n_examples`` random sequences: per-sequence figure with one subplot per
    region (ground truth vs prediction over the forecast horizon).
    ``output_base`` is the prediction file path without ``.npy``, e.g. .../pred_foo_epoch1
    """
    n_seq, T, V = pred.shape
    if gt.shape != pred.shape or len(region_names) != V:
        raise ValueError("Shape mismatch in plot_random_forecast_examples")
    k = min(n_examples, n_seq)
    if k <= 0:
        return []

    pick = rng.choice(n_seq, size=k, replace=False)
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for j, s in enumerate(pick):
        fig, axes = plt.subplots(V, 1, figsize=(10, max(2.2, 1.8 * V)), sharex=True)
        if V == 1:
            axes = np.array([axes])
        t = np.arange(T)
        for r in range(V):
            ax = axes[r]
            ax.plot(t, gt[s, :, r], color="0.35", lw=1.2, label="Ground truth")
            ax.plot(t, pred[s, :, r], color="C3", lw=1.0, alpha=0.9, label="Prediction")
            label = region_names[r]
            if len(label) > 45:
                label = label[:42] + "…"
            ax.set_ylabel(label, fontsize=7)
            ax.grid(True, alpha=0.25)
            if r == 0:
                ax.legend(loc="upper right", fontsize=8)
        axes[-1].set_xlabel("Forecast time step")
        fig.suptitle(
            f"Val row {int(s)} — forecast length {T}",
            fontsize=11,
        )
        fig.tight_layout()
        out_file = output_base.parent / f"{output_base.name}_plot_example{j + 1}_row{s}.png"
        fig.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_file)

    print(f"  Saved {len(saved)} example plot(s) next to predictions.")
    for p in saved:
        print(f"    {p}")
    return saved


def run_one_checkpoint(
    ckpt_path: Path,
    *,
    device: torch.device,
    num_sequences: int,
    long_pred_length: int,
    predictions_dir: Path,
    n_plot_examples: int = N_FORECAST_EXAMPLE_PLOTS,
) -> Path | None:
    parsed = parse_scaling_checkpoint_stem(ckpt_path.stem)
    if parsed is None:
        print(f"[SKIP] Unrecognized checkpoint name: {ckpt_path.name}")
        return None
    middle, epoch = parsed

    val_path = processed_val_npy_path(middle)
    if not val_path.is_file():
        print(f"[SKIP] Missing saved val array: {val_path}")
        return None

    try:
        run_cfg = load_run_config(middle)
    except FileNotFoundError as e:
        print(f"[SKIP] {e}")
        return None

    region_names = run_cfg.get("region_names")
    if not region_names or not isinstance(region_names, list):
        print(f"[SKIP] config_{middle}.json: missing 'region_names' list")
        return None

    val_examples = np.load(val_path)
    if val_examples.ndim != 3:
        raise ValueError(f"Expected val (N,C,T), got {val_examples.shape}")

    print(f"\n{'='*60}")
    print(f"{ckpt_path.name}")
    print(f"  middle={middle}  epoch={epoch}")
    print(f"  val: {val_path} shape={val_examples.shape}")
    print(f"  config: {run_config_json_path(middle)}  regions={len(region_names)}")

    _ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    ck_cfg = _ck["config"]
    T_in = int(ck_cfg["T_in"])
    T_out = int(ck_cfg["T_out"])

    try:
        seq_path = resolve_val_seq_indices_path(middle, ck_cfg)
    except FileNotFoundError as e:
        print(f"[SKIP] {e}")
        del _ck
        return None

    val_seq_indices = np.load(seq_path).astype(np.int64, copy=False)
    if val_seq_indices.ndim != 1 or val_seq_indices.shape[0] != val_examples.shape[0]:
        raise ValueError(
            f"val_seq_indices shape {val_seq_indices.shape} incompatible with val N="
            f"{val_examples.shape[0]} (file {seq_path})"
        )

    unique_seqs = np.unique(val_seq_indices)
    n_logical = int(unique_seqs.size)
    n_val_rows = int(val_examples.shape[0])
    k = min(num_sequences, n_val_rows)
    rng = np.random.default_rng(int(run_cfg.get("random_seed", 101)))
    # Sample unique val subsequence rows (not unique logical sequence ids).
    # GT reconstruction still uses val_seq_indices + selected row start position.
    selected = rng.choice(n_val_rows, size=k, replace=False).astype(np.int64, copy=False)

    print(
        f"  val_seq_indices: {seq_path}  logical_sequences={n_logical}  "
        f"val_rows={n_val_rows}  sampled_subsequences={k}"
    )
    del _ck
    model, ckpt_cfg = create_inference_model(ckpt_path, T_in, T_out, device)

    if len(region_names) != int(ckpt_cfg.get("n_vars", len(region_names))):
        print(
            f"[WARN] region_names count {len(region_names)} vs checkpoint n_vars "
            f"{ckpt_cfg.get('n_vars')}"
        )

    pred_arr, gt_arr = collect_long_predictions(
        model,
        val_examples,
        val_seq_indices,
        selected,
        T_in,
        T_out,
        long_pred_length,
        device,
    )

    # Forecast only — drop initial T_in context (conditioning window).
    t_end = min(pred_arr.shape[1], T_in + long_pred_length)
    if t_end <= T_in:
        print(f"[SKIP] Traces shorter than T_in+1: T={pred_arr.shape[1]}, T_in={T_in}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None
    pred_arr = pred_arr[:, T_in:t_end, :]
    gt_arr = gt_arr[:, T_in:t_end, :]

    out_npy = predictions_dir / f"pred_{middle}_epoch{epoch}.npy"
    save_pred_gt_npy(pred_arr, gt_arr, region_names, out_npy)
    print(
        f"  Wrote {out_npy}  shape (2, n_seq, T, V) = "
        f"(2, {pred_arr.shape[0]}, {pred_arr.shape[1]}, {pred_arr.shape[2]})"
    )

    if n_plot_examples > 0:
        plot_rng = np.random.default_rng(
            int(run_cfg.get("random_seed", 101)) + 10_000 + epoch
        )
        plot_random_forecast_examples(
            pred_arr,
            gt_arr,
            region_names,
            out_npy.with_suffix(""),
            rng=plot_rng,
            n_examples=n_plot_examples,
        )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out_npy


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=None,
        help="Folder containing checkpoints_*_epoch*.pt (default: <repo>/checkpoints)",
    )
    parser.add_argument(
        "--scaling-law-globals",
        type=Path,
        default=None,
        help="Path to scaling_law_globals.json (default: <repo>/scaling_law_globals.json)",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=None,
        help="Output directory for .npy prediction files (default from globals paths.predictions)",
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=None,
        help=(
            f"Unique val subsequences to evaluate (default: "
            f"inference_scaling_law.num_sequences in globals, else {NUM_SEQUENCES})"
        ),
    )
    parser.add_argument(
        "--long-pred-length",
        type=int,
        default=None,
        help=f"Autoregressive steps after initial context (default from globals, else {LONG_PRED_LENGTH})",
    )
    parser.add_argument(
        "--plot-examples",
        type=int,
        default=None,
        help=f"Random val rows to plot (GT vs pred); 0 disables (default from globals, else {N_FORECAST_EXAMPLE_PLOTS})",
    )
    argv = argv if argv is not None else sys.argv[1:]
    args = parser.parse_args(argv)

    root = scaling_law_project_root()
    full = load_scaling_law_globals(args.scaling_law_globals)
    paths = merge_paths_section(full)
    inf = merge_inference_scaling_law_section(full)

    ckpt_dir = (
        Path(args.checkpoints_dir).resolve()
        if args.checkpoints_dir is not None
        else resolve_repo_relative(root, paths["checkpoints"])
    )
    pred_dir = (
        Path(args.predictions_dir).resolve()
        if args.predictions_dir is not None
        else resolve_repo_relative(root, paths["predictions"])
    )
    num_sequences = args.num_sequences if args.num_sequences is not None else inf["num_sequences"]
    long_pred_length = (
        args.long_pred_length if args.long_pred_length is not None else inf["long_pred_length"]
    )
    plot_examples = (
        args.plot_examples if args.plot_examples is not None else inf["n_plot_examples"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("SCALING-LAW INFERENCE (saved val + per-checkpoint .npy)")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Checkpoints dir: {ckpt_dir.resolve()}")
    print(f"Predictions dir: {pred_dir.resolve()}")
    print(f"num_sequences={num_sequences}  long_pred_length={long_pred_length}")

    if not ckpt_dir.is_dir():
        print(f"[ERROR] Not a directory: {ckpt_dir}")
        sys.exit(1)

    ckpts = sorted(ckpt_dir.glob("checkpoints_*_epoch*.pt"))
    if not ckpts:
        print(f"[ERROR] No checkpoints matching checkpoints_*_epoch*.pt in {ckpt_dir}")
        sys.exit(1)

    print(f"\nFound {len(ckpts)} checkpoint file(s).")
    ok = 0
    for p in ckpts:
        out = run_one_checkpoint(
            p,
            device=device,
            num_sequences=num_sequences,
            long_pred_length=long_pred_length,
            predictions_dir=pred_dir,
            n_plot_examples=plot_examples,
        )
        if out is not None:
            ok += 1

    print("\n" + "=" * 80)
    print(f"Done. Wrote {ok}/{len(ckpts)} prediction .npy file(s).")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
