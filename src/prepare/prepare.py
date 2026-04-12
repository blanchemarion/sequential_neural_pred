"""
Prepare neural data for training.

Loads wide-format parquet (sequenceId, itemPosition, brain regions), optionally
explores/plots, and exports a 4D NumPy array compatible with
``src/train/train_all_regimes.py`` (default: ``data_processed/data25_ba2.npy``).

Defaults and paths come from ``scaling_law_globals.json`` at the repo root.
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_PREP_DIR = Path(__file__).resolve().parent
if str(_PREP_DIR) not in sys.path:
    sys.path.insert(0, str(_PREP_DIR))

from prepare_data import (
    brain_region_sample_rng,
    export_organized_data,
    list_brain_region_columns,
    load_parquet_data,
    organize_data_array,
    sample_brain_region_names,
    sample_sequence_ids,
    subset_df_by_sequences,
)

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from helpers.scaling_law_globals import (
    load_scaling_law_globals,
    merge_paths_section,
    merge_prepare_data_section,
    resolve_repo_relative,
    scaling_law_repo_root,
)

# Must match train_all_regimes.py: T_in=90, T_out=90, n_vars=2, data_path data_processed/data25_ba2.npy
TRAIN_T_IN = 90
TRAIN_T_OUT = 90
TRAIN_N_REGIONS = 16
TRAIN_SEQ_FRAC = 1.0
TRAIN_FILE_STEM = "data100_ba16"
# npy_partitions order in scaling_law_globals: 100 -> 0, 50 -> 1, 25 -> 2
TRAIN_SEQ_PARTITION_INDEX = 0
# Brain RNG stream index: 0=ba4, 1=ba8, 2=ba16 in prepare_data; use 3 for ba2-only export
TRAIN_BRAIN_PARTITION_INDEX = 2


def analyze_data_structure(df, metadata):
    """Analyze and print data structure information for wide-format data."""
    print("\n" + "="*80)
    print("DATA STRUCTURE ANALYSIS")
    print("="*80)
    
    # Identify region columns (all columns except sequenceId and itemPosition)
    id_cols = ['sequenceId', 'itemPosition']
    region_cols = [col for col in df.columns if col not in id_cols]
    
    # Basic information
    print(f"\n1. Dataset Shape:")
    print(f"   Rows: {df.shape[0]:,}")
    print(f"   Columns: {df.shape[1]}")
    print(f"   - ID columns: {len(id_cols)} ({', '.join(id_cols)})")
    print(f"   - Region columns: {len(region_cols)}")
    
    # Sequence information
    print(f"\n2. Sequence Information:")
    if 'sequenceId' in df.columns:
        n_sequences = df['sequenceId'].nunique()
        print(f"   Unique sequences: {n_sequences:,}")
        
        seq_lengths = df.groupby('sequenceId').size()
        print(f"   Sequence length statistics:")
        print(f"      Mean: {seq_lengths.mean():.2f}")
        print(f"      Min: {seq_lengths.min()}")
        print(f"      Max: {seq_lengths.max()}")
        print(f"      Std: {seq_lengths.std():.2f}")
    
    if 'itemPosition' in df.columns:
        print(f"   Item positions (time steps):")
        print(f"      Min: {df['itemPosition'].min()}")
        print(f"      Max: {df['itemPosition'].max()}")
        print(f"      Unique: {df['itemPosition'].nunique()}")
    
    # Region columns
    print(f"\n3. Brain Region Columns ({len(region_cols)} regions):")
    print(f"   First 5: {region_cols[:5]}")
    if len(region_cols) > 5:
        print(f"   Last 5: {region_cols[-5:]}")
    
    # Data types
    print(f"\n4. Data Types:")
    print(df.dtypes.value_counts())
    
    # Missing values
    print(f"\n5. Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("   [OK] No missing values found")
    else:
        print(f"   Total missing values: {missing.sum()}")
        missing_cols = missing[missing > 0]
        print(f"   Columns with missing values: {len(missing_cols)}")
        if len(missing_cols) <= 10:
            print(missing_cols)
        else:
            print(f"   Top 10 columns with most missing values:")
            print(missing_cols.sort_values(ascending=False).head(10))
    
    # Memory usage
    print(f"\n6. Memory Usage:")
    print(f"   Total memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Statistical summary for region columns
    print(f"\n7. Statistical Summary (Brain Region Values):")
    region_df = df[region_cols]
    print(f"   Overall mean: {region_df.mean().mean():.4f}")
    print(f"   Overall std: {region_df.std().mean():.4f}")
    print(f"   Overall min: {region_df.min().min():.4f}")
    print(f"   Overall max: {region_df.max().max():.4f}")
    
    print(f"\n   Per-region statistics (first 5 regions):")
    for col in region_cols[:5]:
        print(f"   {col}:")
        print(f"      Mean: {df[col].mean():.4f}, Std: {df[col].std():.4f}, "
              f"Min: {df[col].min():.4f}, Max: {df[col].max():.4f}")
    
    # Metadata information
    if metadata:
        print(f"\n8. Metadata Information:")
        if 'column_types' in metadata:
            print(f"   Column types defined: {len(metadata['column_types'])}")
        if 'selected_columns_statistics' in metadata:
            print(f"   Statistics available for: {len(metadata['selected_columns_statistics'])} columns")
    
    return region_cols


def create_plots(df, region_cols, output_dir='output'):
    """Create visualization plots for wide-format data."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    # 1. Distribution of sequence lengths
    if 'sequenceId' in df.columns:
        seq_lengths = df.groupby('sequenceId').size()
        plt.figure(figsize=(10, 6))
        plt.hist(seq_lengths, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Sequence Length (# time points)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Sequence Lengths', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'sequence_lengths_distribution.png', dpi=300, bbox_inches='tight')
        print(f"   [OK] Saved: sequence_lengths_distribution.png")
        plt.close()
    
    # 2. Sample time series plots for a few sequences and regions
    if 'sequenceId' in df.columns and 'itemPosition' in df.columns and len(region_cols) > 0:
        sample_seq_ids = sorted(df['sequenceId'].unique())[:3]
        sample_regions = region_cols[:min(3, len(region_cols))]
        
        fig, axes = plt.subplots(len(sample_regions), len(sample_seq_ids), 
                                figsize=(5*len(sample_seq_ids), 4*len(sample_regions)))
        
        if len(sample_regions) == 1 and len(sample_seq_ids) == 1:
            axes = np.array([[axes]])
        elif len(sample_regions) == 1:
            axes = axes.reshape(1, -1)
        elif len(sample_seq_ids) == 1:
            axes = axes.reshape(-1, 1)
        
        for reg_idx, region in enumerate(sample_regions):
            for seq_idx, seq_id in enumerate(sample_seq_ids):
                ax = axes[reg_idx, seq_idx]
                seq_data = df[df['sequenceId'] == seq_id].sort_values('itemPosition')
                ax.plot(seq_data['itemPosition'], seq_data[region], linewidth=2)
                ax.set_title(f'Seq {seq_id} - {region[:30]}...', fontsize=10)
                ax.set_xlabel('Time Step', fontsize=9)
                ax.set_ylabel('Value', fontsize=9)
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'sample_time_series.png', dpi=300, bbox_inches='tight')
        print(f"   [OK] Saved: sample_time_series.png")
        plt.close()
    
    # 3. Distribution of values across all regions
    plt.figure(figsize=(10, 6))
    sample_data = df[region_cols].sample(min(10000, len(df)), random_state=42).values.flatten()
    plt.hist(sample_data, bins=100, edgecolor='black', alpha=0.7)
    plt.xlabel('Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Brain Region Values (Sample)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'values_distribution.png', dpi=300, bbox_inches='tight')
    print(f"   [OK] Saved: values_distribution.png")
    plt.close()
    
    # 4. Correlation heatmap between regions (at a single timepoint)
    if len(region_cols) > 1:
        # Use the last timepoint for each sequence
        last_timepoints = df.groupby('sequenceId').tail(1)
        
        # Sample if too many sequences
        if len(last_timepoints) > 5000:
            last_timepoints = last_timepoints.sample(5000, random_state=42)
        
        corr_matrix = last_timepoints[region_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                    xticklabels=False, yticklabels=False)
        
        plt.title('Correlation Between Brain Regions (Last Timepoint)', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'region_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"   [OK] Saved: region_correlation_heatmap.png")
        plt.close()
    
    # 5. Mean value per region (boxplot or bar)
    if len(region_cols) <= 30:
        plt.figure(figsize=(14, 6))
        region_means = df[region_cols].mean().sort_values()
        plt.barh(range(len(region_means)), region_means.values, edgecolor='black', alpha=0.7)
        plt.yticks(range(len(region_means)), 
                  [r[:40] + '...' if len(r) > 40 else r for r in region_means.index],
                  fontsize=8)
        plt.xlabel('Mean Value', fontsize=12)
        plt.title('Mean Value per Brain Region', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(output_dir / 'region_means.png', dpi=300, bbox_inches='tight')
        print(f"   [OK] Saved: region_means.png")
        plt.close()
    
    print(f"\n   All plots saved to '{output_dir}' directory")


def denormalize_data(data_array, region_names, metadata, eps=1e-9):
    """
    Denormalize data array using statistics from metadata.
    x_raw = x_norm * std + mean
    
    Works with 4D array: (n_sequences, n_subsequences, n_regions, n_time)
    """
    if not metadata or 'selected_columns_statistics' not in metadata:
        print("\n   Warning: No statistics found in metadata, skipping denormalization")
        return data_array
    
    print("\n" + "="*80)
    print("DENORMALIZING DATA")
    print("="*80)
    
    stats = metadata['selected_columns_statistics']
    out = data_array.astype(np.float64, copy=True)
    
    print("   Denormalizing per region...")
    denormalized_count = 0
    for r, name in enumerate(region_names):
        if name not in stats:
            print(f"   Warning: Region '{name}' not found in JSON stats, skipping")
            continue
        
        std = float(stats[name]["std"])
        mean = float(stats[name]["mean"])
        # Denormalize across all sequences, subsequences, and time steps for this region
        out[:, :, r, :] = out[:, :, r, :] * (std + eps) + mean
        denormalized_count += 1
    
    print(f"   [OK] Denormalized {denormalized_count}/{len(region_names)} regions")
    return out


def export_to_csv(df, output_path='data/data_raw.csv'):
    """Export raw data to CSV file."""
    print("\n" + "="*80)
    print("EXPORTING RAW DATA TO CSV")
    print("="*80)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)
    
    print(f"   Exporting to {output_path}...")
    print(f"   This may take a while for large datasets...")
    
    df.to_csv(output_path, index=False)
    
    file_size = output_path.stat().st_size / (1024**2)
    print(f"   [OK] Export complete!")
    print(f"   File size: {file_size:.2f} MB")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")


def main():
    """Build ``data_processed/data25_ba2.npy`` for ``train_all_regimes.py`` defaults."""
    parser = argparse.ArgumentParser(
        description="Prepare parquet to NumPy for train_all_regimes.py (default: data25_ba2.npy)."
    )
    parser.add_argument(
        "--scaling-law-globals",
        type=Path,
        default=None,
        help="Path to scaling_law_globals.json (default: repo root)",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run structure analysis and save plots under output/",
    )
    parser.add_argument(
        "--export-denormalized",
        action="store_true",
        help=f"Also save {TRAIN_FILE_STEM}_denormalized.npy (uses parquet JSON stats)",
    )
    args = parser.parse_args()

    full = load_scaling_law_globals(args.scaling_law_globals)
    pcfg = merge_prepare_data_section(full)
    dir_paths = merge_paths_section(full)
    repo = scaling_law_repo_root()

    sequence_id_col = str(pcfg["sequence_id_col"])
    id_cols = tuple(str(x) for x in pcfg["id_cols"])
    parquet_filename = str(pcfg["parquet_filename"])
    metadata_filename = str(pcfg["metadata_json_filename"])
    subsequence_length = int(pcfg["subsequence_length"])
    only_full = bool(pcfg["only_full_subsequences"])
    seq_base_seed = int(pcfg["sequence_sample_base_seed"])
    brain_base_seed = int(pcfg["brain_region_sample_base_seed"])

    t_need = TRAIN_T_IN + TRAIN_T_OUT
    if subsequence_length < t_need:
        raise ValueError(
            f"prepare_data.subsequence_length ({subsequence_length}) must be >= "
            f"T_in + T_out from train_all_regimes ({t_need}). "
            f"Increase it in scaling_law_globals.json."
        )

    input_dir = resolve_repo_relative(repo, dir_paths["data_raw"])
    output_dir = resolve_repo_relative(repo, dir_paths["data_processed"])
    output_dir.mkdir(parents=True, exist_ok=True)

    df, metadata = load_parquet_data(
        input_dir,
        parquet_filename=parquet_filename,
        metadata_filename=metadata_filename,
    )

    if args.analyze:
        region_cols = analyze_data_structure(df, metadata)
        create_plots(df, region_cols, output_dir=resolve_repo_relative(repo, "output"))

    all_seq_ids = np.sort(df[sequence_id_col].unique())
    n_seq_total = len(all_seq_ids)
    rng = np.random.default_rng(seq_base_seed + TRAIN_SEQ_PARTITION_INDEX)
    chosen_ids = sample_sequence_ids(all_seq_ids, TRAIN_SEQ_FRAC, rng)
    df_part = subset_df_by_sequences(df, chosen_ids, id_col=sequence_id_col)

    all_region_cols = list_brain_region_columns(df, id_cols=id_cols)
    rng_regions = brain_region_sample_rng(
        TRAIN_SEQ_PARTITION_INDEX, TRAIN_BRAIN_PARTITION_INDEX, brain_base_seed
    )
    chosen_regions = sample_brain_region_names(all_region_cols, TRAIN_N_REGIONS, rng_regions)

    print("\n" + "=" * 80)
    print("TRAINING EXPORT (aligned with train_all_regimes.py)")
    print("=" * 80)
    print(f"   Sequences: {len(chosen_ids):,} / {n_seq_total:,} (fraction={TRAIN_SEQ_FRAC})")
    print(f"   Regions (n_vars={TRAIN_N_REGIONS}): {chosen_regions}")

    data_array, region_names, subsequence_info = organize_data_array(
        df_part,
        metadata,
        subsequence_length=subsequence_length,
        only_full_subsequences=only_full,
        region_cols=chosen_regions,
        id_cols=id_cols,
    )

    if data_array.shape[2] != TRAIN_N_REGIONS:
        raise ValueError(f"Expected {TRAIN_N_REGIONS} regions, got {data_array.shape[2]}")

    partition_info = {
        "sequence_partition": "25",
        "sequence_frac": TRAIN_SEQ_FRAC,
        "n_sequence_ids_total": n_seq_total,
        "n_sequence_ids_kept": int(len(chosen_ids)),
        "sequence_sample_seed": seq_base_seed + TRAIN_SEQ_PARTITION_INDEX,
        "brain_partition": "ba2",
        "n_brain_regions_in_array": len(region_names),
        "region_names": list(region_names),
        "train_T_in": TRAIN_T_IN,
        "train_T_out": TRAIN_T_OUT,
        "train_file_stem": TRAIN_FILE_STEM,
    }

    export_paths = export_organized_data(
        data_array,
        region_names,
        subsequence_info,
        output_dir=output_dir,
        file_stem=TRAIN_FILE_STEM,
        partition_info=partition_info,
    )

    npy_denorm_path = None
    if args.export_denormalized and metadata:
        data_array_denorm = denormalize_data(data_array, region_names, metadata)
        npy_denorm_path = output_dir / f"{TRAIN_FILE_STEM}_denormalized.npy"
        np.save(npy_denorm_path, data_array_denorm)
        print(f"   [OK] Denormalized: {npy_denorm_path}")

    print("\n" + "=" * 80)
    print("PREPARE COMPLETE")
    print("=" * 80)
    print("\nSummary:")
    print(f"  - Raw dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  - Organized array shape: {data_array.shape}  (n_seq, n_sub, n_regions, n_time)")
    print(f"  - Sequences in export: {subsequence_info['sequenceId'].nunique()}")
    print(f"  - Total subsequences: {len(subsequence_info)}")
    print(f"  - Time steps per subsequence (last axis): {data_array.shape[3]}")
    print(f"  - Brain regions (n_vars): {len(region_names)}")
    print("\nNext:")
    print(f"  python src/train/train_all_regimes.py   # expects {export_paths['full_array']}")
    print("\nExported:")
    print(f"  - {export_paths['full_array']}")
    print(f"  - {export_paths['metadata']}")
    if npy_denorm_path:
        print(f"  - {npy_denorm_path}")

if __name__ == "__main__":
    try:
        main()
    finally:
        # Clean up memory explicitly
        print("\n[INFO] Cleaning up memory...")
        plt.close('all')
        gc.collect()
        # Clear GPU memory if PyTorch is available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("[INFO] GPU cache cleared")
        except ImportError:
            pass