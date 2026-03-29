"""
Functions for preprocessing the data for the model.
- Reshaping the data into examples
- Filtering out examples with NaN values
- Splitting the data into train, validation, and test sets
- Normalizing the data
- Creating dataloaders
- Verifying the data loading
"""

import torch
import numpy as np
from pathlib import Path
from typing import Tuple
from torch.utils.data import DataLoader


# ----------------------------
# Dataset
# ----------------------------
class TimeSeriesDataset:
    def __init__(self, data: np.ndarray, T_in: int, T_out: int):
        assert data.ndim == 3, f"Expected 3D array, got {data.ndim}D"
        assert T_in + T_out <= data.shape[2], \
            f"T_in + T_out ({T_in + T_out}) must be <= total time steps ({data.shape[2]})"

        self.T_in = T_in
        self.T_out = T_out

        # data: (N, C, T)
        inputs = data[:, :, :T_in]                    # (N, C, T_in)
        targets = data[:, :, T_in:T_in + T_out]       # (N, C, T_out)

        # to token format: (N, T, C)
        self.inputs = inputs.transpose(0, 2, 1)       # (N, T_in, C)
        self.targets = targets.transpose(0, 2, 1)     # (N, T_out, C)

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int):
        return torch.FloatTensor(self.inputs[idx]), torch.FloatTensor(self.targets[idx])


# ----------------------------
# I/O + reshaping
# ----------------------------
def load_data(data_path: str) -> np.ndarray:
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print(f"Loading data from {data_path}...")
    data = np.load(data_path)
    print(f"Loaded data shape: {data.shape}")
    return data


def reshape_to_examples(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_seq, n_sub, n_reg, n_time = data.shape
    n_examples = n_seq * n_sub

    examples = data.reshape(n_examples, n_reg, n_time)              # (N, C, T)
    sequence_indices = np.repeat(np.arange(n_seq), n_sub)           # (N,)

    print(f"Reshaped to {n_examples} examples")
    print(f"  Each example: {n_reg} regions × {n_time} time-steps")
    return examples, sequence_indices


def filter_examples_by_nan(examples: np.ndarray, sequence_indices: np.ndarray, T_in: int, T_out: int) -> tuple[np.ndarray, np.ndarray]:
    total = examples.shape[0]
    inputs = examples[:, :, :T_in]
    targets = examples[:, :, T_in:T_in + T_out]

    valid_inputs = np.isfinite(inputs).all(axis=(1, 2))
    valid_targets = np.isfinite(targets).all(axis=(1, 2))
    valid_mask = valid_inputs & valid_targets

    filtered_examples = examples[valid_mask]
    filtered_seq_indices = sequence_indices[valid_mask]

    removed = int((~valid_mask).sum())
    kept = int(valid_mask.sum())
    pct_removed = (removed / total * 100) if total > 0 else 0.0
    print(f"\nFiltered invalid examples: removed {removed} of {total} ({pct_removed:.1f}%).")
    print(f"Remaining valid examples: {kept}")

    return filtered_examples, filtered_seq_indices


def split_by_sequences(examples: np.ndarray, sequence_indices: np.ndarray, train_ratio: float = 0.8, random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(random_seed)

    unique_sequences = np.unique(sequence_indices)
    n_sequences = len(unique_sequences)

    shuffled_sequences = np.random.permutation(unique_sequences)

    n_train_sequences = int(n_sequences * train_ratio)
    train_sequence_ids = set(shuffled_sequences[:n_train_sequences])
    val_sequence_ids = set(shuffled_sequences[n_train_sequences:])

    train_mask = np.array([seq_id in train_sequence_ids for seq_id in sequence_indices])
    val_mask = ~train_mask

    train_examples = examples[train_mask]
    val_examples = examples[val_mask]
    train_seq_indices = sequence_indices[train_mask]
    val_seq_indices = sequence_indices[val_mask]

    print(f"\nSplit by sequences:")
    print(f"  Training sequences: {len(train_sequence_ids)} ({len(train_sequence_ids)/n_sequences*100:.1f}%)")
    print(f"  Validation sequences: {len(val_sequence_ids)} ({len(val_sequence_ids)/n_sequences*100:.1f}%)")
    print(f"  Training examples: {len(train_examples)}")
    print(f"  Validation examples: {len(val_examples)}")

    return train_examples, val_examples, train_seq_indices, val_seq_indices


def compute_train_stats_input_only(train_examples: np.ndarray, T_in: int, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-region mean/std using TRAIN ONLY and INPUT-ONLY timepoints.

    train_examples expected shape: (N, C, T_total)
    We use only [:T_in] along time axis.

    Returns:
      mean: (1, C, 1)
      std:  (1, C, 1)
    """
    if train_examples.ndim != 3:
        raise ValueError(f"Expected train_examples to be 3D (N,C,T), got {train_examples.shape}")

    if T_in <= 0 or T_in > train_examples.shape[2]:
        raise ValueError(f"T_in={T_in} must be in [1, T_total={train_examples.shape[2]}]")

    x_in = train_examples[:, :, :T_in]  # (N, C, T_in)

    mean = np.nanmean(x_in, axis=(0, 2), keepdims=True)  # (1, C, 1)
    std = np.nanstd(x_in, axis=(0, 2), keepdims=True)    # (1, C, 1)
    std = np.maximum(std, eps)

    return mean, std


def apply_normalization_nct(examples: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Apply z-score normalization for shape (N, C, T).
    mean/std: (1, C, 1)
    """
    if examples.ndim != 3:
        raise ValueError(f"Expected examples to be 3D (N,C,T), got {examples.shape}")
    return (examples - mean) / std


def normalize_after_split_input_only(
    train_examples: np.ndarray,
    val_examples: np.ndarray,
    T_in: int,
    test_examples: np.ndarray | None = None,
    eps: float = 1e-6,
    save_stats_path: str | None = None,
):
    """
    Compute stats on TRAIN only, INPUT-only timesteps, apply to train/val/(test).

    This guarantees:
    - no across-split leakage (val never influences stats)
    - no future leakage (target timesteps never influence stats)
    """
    mean, std = compute_train_stats_input_only(train_examples, T_in=T_in, eps=eps)

    train_norm = apply_normalization_nct(train_examples, mean, std)
    val_norm = apply_normalization_nct(val_examples, mean, std)
    test_norm = None if test_examples is None else apply_normalization_nct(test_examples, mean, std)

    if save_stats_path is not None:
        save_path = Path(save_stats_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        base_path = save_path.with_suffix("")
        mean_path = base_path.parent / f"{base_path.name}_mean.npy"
        std_path = base_path.parent / f"{base_path.name}_std.npy"
        meta_path = base_path.parent / f"{base_path.name}_meta.json"

        np.save(str(mean_path), mean.astype(np.float32))
        np.save(str(std_path), std.astype(np.float32))

        meta = {
            "layout": "NCT",
            "computed_on": "TRAIN_ONLY",
            "timepoints_used_for_stats": f"[0:{T_in})",
            "eps": float(eps),
        }
        meta_path.write_text(__import__("json").dumps(meta, indent=2))

        print("Saved normalization stats to:")
        print(f"  Mean: {mean_path}")
        print(f"  Std:  {std_path}")
        print(f"  Meta: {meta_path}")

    return train_norm, val_norm, test_norm, (mean, std, "NCT")


# ----------------------------
# Dataloaders + verification
# ----------------------------

def create_dataloaders(train_examples: np.ndarray, val_examples: np.ndarray, T_in: int, T_out: int, batch_size: int, num_workers: int, shuffle_train: bool = True):

    print(f"\nCreating DataLoaders:")
    print(f"  T_in (input length): {T_in}")
    print(f"  T_out (output length): {T_out}")
    print(f"  Batch size: {batch_size}")
    print(f"  num_workers: {num_workers}")

    train_dataset = TimeSeriesDataset(train_examples, T_in=T_in, T_out=T_out)
    val_dataset   = TimeSeriesDataset(val_examples,   T_in=T_in, T_out=T_out)

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=False,
    )

    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    return train_loader, val_loader



def verify_data_loading(train_loader, val_loader):
    print("\n" + "=" * 80)
    print("VERIFYING DATA LOADING")
    print("=" * 80)

    train_input, train_target = next(iter(train_loader))
    print(f"\nTraining batch:")
    print(f"  Input shape:  {train_input.shape}  (batch_size, T_in, n_regions)")
    print(f"  Target shape: {train_target.shape}  (batch_size, T_out, n_regions)")
    print(f"  Input dtype:  {train_input.dtype}")
    print(f"  Target dtype: {train_target.dtype}")

    val_input, val_target = next(iter(val_loader))
    print(f"\nValidation batch:")
    print(f"  Input shape:  {val_input.shape}  (batch_size, T_in, n_regions)")
    print(f"  Target shape: {val_target.shape}  (batch_size, T_out, n_regions)")

    print(f"\nSample input (first example, first time-step):")
    print(f"  {train_input[0, 0, :].numpy()}")

    print(f"\nSample target (first example, first time-step):")
    print(f"  {train_target[0, 0, :].numpy()}")

    print("\n[OK] Data loading verified!")


# ----------------------------
# Main (to run to test the preprocessing pipeline)
# ----------------------------
def main():
    print("=" * 80)
    print("DATA PREPARATION FOR MULTI-STEP TIME SERIES PREDICTION")
    print("=" * 80)

    data_path = "data/data100_ba2.npy"
    T_in = 90
    T_out = 90
    train_ratio = 0.8
    batch_size = 32
    random_seed = 42

    data_array = load_data(data_path)

    examples, sequence_indices = reshape_to_examples(data_array)

    examples, sequence_indices = filter_examples_by_nan(
        examples, sequence_indices, T_in=T_in, T_out=T_out
    )

    train_examples, val_examples, _, _ = split_by_sequences(
        examples, sequence_indices, train_ratio=train_ratio, random_seed=random_seed
    )

    train_examples, val_examples, _, norm = normalize_after_split_input_only(
        train_examples,
        val_examples,
        T_in=T_in,
        eps=1e-6,
        #save_stats_path="data/train_norm_stats.npy",
    )

    train_loader, val_loader = create_dataloaders(
        train_examples,
        val_examples,
        T_in=T_in,
        T_out=T_out,
        batch_size=batch_size,
        num_workers=8,
    )

    verify_data_loading(train_loader, val_loader)

    print("\n" + "=" * 80)
    print("DATA PREPARATION COMPLETE!")
    print("=" * 80)
    return train_loader, val_loader


if __name__ == "__main__":
    import gc
    try:
        train_loader, val_loader = main()
    finally:
        print("\n[INFO] Cleaning up memory...")
        try:
            del train_loader, val_loader
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("[INFO] GPU cache cleared")
