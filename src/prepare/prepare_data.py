"""
This script loads parquet data, explores its structure, creates visualizations,
and exports to npy format for model building.

Data Format: Wide format with sequenceId, itemPosition, and region columns
"""

import math
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import warnings
import seaborn as sns


PARQUET_FILENAME = "data-clean-all-2pct.parquet"

NPY_PARTITIONS = [
    {"name": "100", "sequence_frac": 1.0},
    {"name": "50", "sequence_frac": 0.5},
    {"name": "25", "sequence_frac": 0.25},
]

BRAIN_REGION_PARTITIONS = [
    {"name": "2", "brain_areas": 2},
    {"name": "4", "brain_areas": 4},
    {"name": "8", "brain_areas": 8},
    {"name": "16", "brain_areas": 16},
]

OUTPUT_DIR = "data_processed"
INPUT_DIR = "data_raw"
SEQUENCE_ID_COL = "sequenceId"
SEQUENCE_SAMPLE_BASE_SEED = 42

# Independent random region subset per (sequence partition × brain partition); reproducible via seed.
BRAIN_REGION_SAMPLE_BASE_SEED = 4242
SUBSEQUENCE_LENGTH = 360
ONLY_FULL_SUBSEQUENCES = True

ID_COLS_DEFAULT = ("sequenceId", "itemPosition")


def load_metadata(json_path):
    """Load metadata from JSON file."""
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    return metadata


def n_sequences_to_keep(n_total: int, sequence_frac: float) -> int:
    """How many sequence ids to sample; ceil(n * frac), at least 1 if n > 0 and frac < 1."""
    if n_total <= 0:
        return 0
    if sequence_frac >= 1.0:
        return n_total
    return min(n_total, max(1, math.ceil(n_total * sequence_frac)))


def sample_sequence_ids(all_ids: np.ndarray, sequence_frac: float, rng: np.random.Generator) -> np.ndarray:
    """
    Random subset of sequence ids (no replacement). all_ids should be sorted for reproducibility
    when the same seed is used with the same n_total.
    """
    n_total = len(all_ids)
    n_keep = n_sequences_to_keep(n_total, sequence_frac)
    if n_keep == 0:
        return np.array([], dtype=all_ids.dtype)
    if n_keep >= n_total:
        return all_ids.copy()
    idx = rng.choice(n_total, size=n_keep, replace=False)
    idx.sort()
    return all_ids[idx]


def subset_df_by_sequences(df: pd.DataFrame, sequence_ids: np.ndarray, id_col: str = SEQUENCE_ID_COL) -> pd.DataFrame:
    """All rows whose id_col is in sequence_ids (full sequences only)."""
    if len(sequence_ids) == 0:
        return df.iloc[0:0].copy()
    return df.loc[df[id_col].isin(sequence_ids)].copy()


def list_brain_region_columns(df: pd.DataFrame, id_cols=ID_COLS_DEFAULT) -> list[str]:
    """Sorted region column names (all non-id columns)."""
    id_set = set(id_cols)
    return sorted(c for c in df.columns if c not in id_set)


def brain_region_sample_rng(sequence_part_index: int, brain_part_index: int) -> np.random.Generator:
    """
    Deterministic RNG for picking region columns for one (sequence partition × brain partition).
    Each combination gets its own draw so easier/harder regions do not carry across data splits.
    """
    seed = (
        BRAIN_REGION_SAMPLE_BASE_SEED
        + sequence_part_index * 1_000_003
        + brain_part_index * 17_389
    )
    return np.random.default_rng(seed)


def brain_region_sample_seed_value(sequence_part_index: int, brain_part_index: int) -> int:
    """Integer seed matching brain_region_sample_rng (for metadata JSON)."""
    return int(
        BRAIN_REGION_SAMPLE_BASE_SEED
        + sequence_part_index * 1_000_003
        + brain_part_index * 17_389
    )


def sample_brain_region_names(
    all_region_cols: list[str],
    n_keep: int,
    rng: np.random.Generator,
) -> list[str]:
    """
    Random subset of region names without replacement; sorted by original column order.
    If n_keep >= len(all_region_cols), returns all columns (sorted).
    """
    n_total = len(all_region_cols)
    if n_total == 0:
        return []
    if n_keep <= 0:
        raise ValueError(f"n_keep must be positive, got {n_keep}")
    k = min(n_keep, n_total)
    if k == n_total:
        return list(all_region_cols)
    pick = rng.choice(n_total, size=k, replace=False)
    pick.sort()
    return [all_region_cols[i] for i in pick]


def load_parquet_data(data_dir="data", parquet_filename: str | None = None):
    """Load a single parquet file and its metadata."""
    data_dir = Path(data_dir)
    name = parquet_filename if parquet_filename is not None else PARQUET_FILENAME
    parquet_path = data_dir / name
    metadata_path = data_dir / "data-clean-all.json"

    if not parquet_path.exists():
        raise FileNotFoundError(f"{parquet_path} not found")
    
    print(f"Loading data from {parquet_path.name}...")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded dataset shape: {df.shape}")

    metadata = None
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        print(f"Warning: {metadata_path} not found, proceeding without metadata")

    return df, metadata


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


def organize_data_array(
    df,
    metadata,
    subsequence_length,
    only_full_subsequences,
    region_cols,
):
    """
    Organize wide-format parquet data into structured array format with subsequences.
    
    Each long sequence is split into subsequences of fixed length (default 91 timesteps).
    
    Input format: sequenceId, itemPosition, region1, region2, ...
    Output format: (n_sequences, n_subsequences, n_regions, n_time)
    
    Args:
        only_full_subsequences: If True, only keep full-length subsequences (no padding with NaN)
        region_cols: If set, only these columns (must exist on df) define the region axis, in order.
            If None, all non-id columns are used (sorted).
    
    Returns: (data_array, region_names, subsequence_info)
        - data_array: shape (n_sequences, n_subsequences, n_regions, n_time)
        - region_names: list of region column names
        - subsequence_info: DataFrame with metadata
    """
    print("\n" + "="*80)
    print("ORGANIZING DATA INTO ARRAY STRUCTURE WITH SUBSEQUENCES")
    print("="*80)
    
    # Identify columns
    id_cols = list(ID_COLS_DEFAULT)
    all_region = sorted([col for col in df.columns if col not in id_cols])
    if region_cols is None:
        region_cols = all_region
    else:
        region_cols = list(region_cols)
        missing = [c for c in region_cols if c not in df.columns]
        if missing:
            raise ValueError(f"region_cols not in dataframe: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        extra = set(region_cols) - set(all_region)
        if extra:
            raise ValueError(f"region_cols must be brain region columns only; unknown: {sorted(extra)[:5]}")
    
    print(f"   Found {len(region_cols)} brain regions")
    print(f"   Subsequence length: {subsequence_length} time steps")
    print(f"   Only full subsequences: {only_full_subsequences}")
    
    # Sort data for efficient processing
    df_sorted = df.sort_values(['sequenceId', 'itemPosition']).reset_index(drop=True)
    
    # First pass: determine subsequences per sequence
    print("   Analyzing sequence structure...")
    sequence_ids = sorted(df_sorted['sequenceId'].unique())
    
    seq_subseq_info = {}  # seq_id -> list of (sub_idx, start, end, length, is_full)
    
    for seq_id in sequence_ids:
        seq_data = df_sorted[df_sorted['sequenceId'] == seq_id].reset_index(drop=True)
        seq_length = len(seq_data)
        
        # Calculate number of full subsequences
        n_full_subseq = seq_length // subsequence_length
        
        subseq_list = []
        for sub_idx in range(n_full_subseq):
            start_idx = sub_idx * subsequence_length
            end_idx = start_idx + subsequence_length
            subseq_list.append((sub_idx, start_idx, end_idx, subsequence_length, True))
        
        # Add partial subsequence if not using only_full_subsequences
        if not only_full_subsequences:
            remainder = seq_length % subsequence_length
            if remainder > 0:
                start_idx = n_full_subseq * subsequence_length
                end_idx = seq_length
                subseq_list.append((n_full_subseq, start_idx, end_idx, remainder, False))
        
        seq_subseq_info[seq_id] = subseq_list
    
    # Filter out sequences with no valid subsequences
    valid_sequences = [(seq_id, info) for seq_id, info in seq_subseq_info.items() if len(info) > 0]
    
    if len(valid_sequences) == 0:
        raise ValueError("No valid subsequences found! Try using only_full_subsequences=False")
    
    n_sequences = len(valid_sequences)
    max_subsequences = max(len(info) for _, info in valid_sequences)
    n_regions = len(region_cols)
    
    print(f"   Valid sequences: {n_sequences}")
    print(f"   Dimensions: {n_sequences} sequences × {max_subsequences} max subsequences × {n_regions} regions × {subsequence_length} time steps")
    
    # Preallocate 4D array with NaN
    print("   Preallocating 4D array...")
    data_array = np.full((n_sequences, max_subsequences, n_regions, subsequence_length), 
                         np.nan, dtype=np.float32)
    
    # Fill array
    print("   Filling array...")
    subsequence_info_list = []
    
    for seq_idx, (seq_id, subseq_list) in enumerate(valid_sequences):
        seq_data = df_sorted[df_sorted['sequenceId'] == seq_id].reset_index(drop=True)
        
        for sub_idx, start_idx, end_idx, actual_length, is_full in subseq_list:
            subseq_data = seq_data.iloc[start_idx:end_idx]
            
            # Extract region values as array: (actual_length, n_regions)
            region_values = subseq_data[region_cols].values
            
            # Fill into 4D array: data_array[seq, sub, region, time]
            # Need to transpose: (actual_length, n_regions) -> (n_regions, actual_length)
            data_array[seq_idx, sub_idx, :, :actual_length] = region_values.T
            
            # Store metadata
            subsequence_info_list.append({
                'sequenceId': seq_id,
                'subsequenceId': sub_idx,
                'start_pos': int(subseq_data['itemPosition'].iloc[0]),
                'end_pos': int(subseq_data['itemPosition'].iloc[-1]),
                'length': actual_length,
                'is_full': is_full
            })
        
        if (seq_idx + 1) % 100 == 0:
            print(f"      Processed {seq_idx + 1}/{n_sequences} sequences...")
    
    subsequence_info = pd.DataFrame(subsequence_info_list)
    
    print(f"   [OK] Array shape: {data_array.shape}")
    print(f"        Format: (n_sequences={n_sequences}, n_subsequences={max_subsequences}, n_regions={n_regions}, n_time={subsequence_length})")
    
    # Check for missing values
    n_missing = np.isnan(data_array).sum()
    if n_missing > 0:
        pct_missing = 100 * n_missing / data_array.size
        print(f"   Note: Array contains {n_missing:,} NaN values ({pct_missing:.2f}% of total)")
        if only_full_subsequences:
            print(f"         NaNs are only in padding slots for sequences with fewer subsequences")
        else:
            print(f"         NaNs are in padding slots and partial subsequences")
    
    # Print statistics about subsequences per sequence
    subseq_per_seq = subsequence_info.groupby('sequenceId')['subsequenceId'].max() + 1
    print(f"\n   Subsequences per sequence statistics:")
    print(f"      Mean: {subseq_per_seq.mean():.2f}")
    print(f"      Min: {subseq_per_seq.min()}")
    print(f"      Max: {subseq_per_seq.max()}")
    
    if only_full_subsequences:
        n_full = (subsequence_info['is_full'] == True).sum()
        print(f"      Full subsequences: {n_full} (100%)")
    else:
        n_full = (subsequence_info['is_full'] == True).sum()
        n_partial = (subsequence_info['is_full'] == False).sum()
        print(f"      Full subsequences: {n_full} ({100*n_full/len(subsequence_info):.1f}%)")
        print(f"      Partial subsequences: {n_partial} ({100*n_partial/len(subsequence_info):.1f}%)")
    
    return data_array, region_cols, subsequence_info



def export_organized_data(
    data_array,
    region_names,
    subsequence_info,
    output_dir,
    file_stem,
    partition_info=None,
):
    """
    Export organized data in multiple formats:
    1. NumPy array (.npy)
    2. Subsequence info CSV
    3. Collapsed array (last timepoint of each subsequence)
    
    Input: 4D array (n_sequences, n_subsequences, n_regions, n_time)
    file_stem: outputs {file_stem}.npy and {file_stem}_metadata.json
    partition_info: optional dict merged into metadata JSON (e.g. sequence_frac, partition name)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print("EXPORTING ORGANIZED DATA")
    print("="*80)
    
    n_seq, n_sub, n_reg, n_time = data_array.shape
    
    # 1. Save full numpy array
    npy_path = output_dir / f"{file_stem}.npy"
    print(f"   Saving full array to {npy_path}...")
    np.save(npy_path, data_array)
    file_size = npy_path.stat().st_size / (1024**2)
    print(f"   [OK] Saved: {file_size:.2f} MB")

    
    # 3. Save metadata about the array
    metadata_path = output_dir / f"{file_stem}_metadata.json"
    metadata_dict = {
        'shape': list(data_array.shape),
        'dimensions': {
            'sequences': n_seq,
            'subsequences': n_sub,
            'regions': n_reg,
            'timepoints': n_time
        },
        'region_names': region_names,
        'n_sequences': n_seq,
        'max_subsequences_per_sequence': n_sub
    }
    if partition_info:
        metadata_dict['partition'] = partition_info
    with open(metadata_path, 'w') as f:
        json.dump(metadata_dict, f, indent=2)
    print(f"   [OK] Saved metadata: {metadata_path}")
    
    # 4. Save last timepoint array (per subsequence)
    print("   Extracting last timepoints from each subsequence...")
    # Shape: (n_seq, n_sub, n_reg)
    last_timepoint_array = np.full((n_seq, n_sub, n_reg), np.nan, dtype=np.float32)
    
    for seq_idx in range(n_seq):
        for sub_idx in range(n_sub):
            # Find last non-NaN timepoint for this subsequence
            for time_idx in range(n_time - 1, -1, -1):
                if not np.all(np.isnan(data_array[seq_idx, sub_idx, :, time_idx])):
                    last_timepoint_array[seq_idx, sub_idx, :] = data_array[seq_idx, sub_idx, :, time_idx]
                    break

    
    return {
        'full_array': npy_path,
        'metadata': metadata_path
    }


def main():
    """Load parquet once, then build and export one .npy per configured partition."""
    print("="*80)
    print("DATA PREPARATION")
    print("="*80)

    output_dir = Path(OUTPUT_DIR)
    input_dir = Path(INPUT_DIR)
    df, metadata = load_parquet_data(input_dir)

    if SEQUENCE_ID_COL not in df.columns:
        raise KeyError(f"Column {SEQUENCE_ID_COL!r} required for sequence-level partitions")

    all_seq_ids = np.sort(df[SEQUENCE_ID_COL].unique())
    n_seq_total = len(all_seq_ids)
    print(f"\n   Total distinct {SEQUENCE_ID_COL}: {n_seq_total:,}")

    # Analyze structure
    # region_cols = analyze_data_structure(df, metadata)

    # Create plots
    # create_plots(df, region_cols, output_dir='output')

    all_export_paths = []

    all_region_cols = list_brain_region_columns(df)
    n_regions_available = len(all_region_cols)
    if n_regions_available == 0:
        raise ValueError("No brain region columns in parquet")

    for part_index, part in enumerate(NPY_PARTITIONS):
        name = part["name"]
        sequence_frac = float(part["sequence_frac"])
        if sequence_frac <= 0:
            raise ValueError(f"Partition {name!r}: sequence_frac must be > 0, got {sequence_frac}")

        rng = np.random.default_rng(SEQUENCE_SAMPLE_BASE_SEED + part_index)
        chosen_ids = sample_sequence_ids(all_seq_ids, sequence_frac, rng)
        n_keep = len(chosen_ids)
        df_part = subset_df_by_sequences(df, chosen_ids)

        print("\n" + "="*80)
        print(f"SEQUENCE PARTITION: {name}  (sequence_frac={sequence_frac})")
        print("="*80)
        print(f"   Sequences kept: {n_keep:,} / {n_seq_total:,}")
        print(f"   Rows in subset: {len(df_part):,}")

        for b_index, bpart in enumerate(BRAIN_REGION_PARTITIONS):
            bname = bpart["name"]
            n_areas = int(bpart["brain_areas"])
            if n_areas <= 0:
                raise ValueError(f"BRAIN_REGION_PARTITIONS entry {bname!r}: brain_areas must be > 0")

            rng_regions = brain_region_sample_rng(part_index, b_index)
            chosen_regions = sample_brain_region_names(all_region_cols, n_areas, rng_regions)
            brain_seed = brain_region_sample_seed_value(part_index, b_index)

            print("\n" + "-" * 60)
            print(
                f"  Brain partition: {bname} regions (requested={n_areas}, kept={len(chosen_regions)}) "
                f"[independent draw for seq={name} × ba{bname}; seed={brain_seed}]"
            )
            print(f"   {chosen_regions}")

            data_array, region_names, subsequence_info = organize_data_array(
                df_part,
                metadata,
                subsequence_length=SUBSEQUENCE_LENGTH,
                only_full_subsequences=ONLY_FULL_SUBSEQUENCES,
                region_cols=chosen_regions,
            )

            file_stem = f"data{name}_ba{bname}"
            partition_info = {
                "sequence_partition": name,
                "sequence_frac": sequence_frac,
                "n_sequence_ids_total": n_seq_total,
                "n_sequence_ids_kept": n_keep,
                "sequence_sample_seed": SEQUENCE_SAMPLE_BASE_SEED + part_index,
                "brain_partition": bname,
                "brain_areas_requested": n_areas,
                "n_brain_regions_in_array": len(chosen_regions),
                "n_brain_regions_available": n_regions_available,
                "region_names": chosen_regions,
                "brain_sample_seed": brain_seed,
            }
            export_paths = export_organized_data(
                data_array,
                region_names,
                subsequence_info,
                output_dir=output_dir,
                file_stem=file_stem,
                partition_info=partition_info,
            )
            all_export_paths.append((f"{name}_{bname}", export_paths))

            del data_array, subsequence_info
            gc.collect()

        del df_part
        gc.collect()

    print("\n" + "="*80)
    print("EXPLORATION COMPLETE!")
    print("="*80)
    print("\nSummary (full parquet):")
    print(f"  - Raw dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  - Distinct sequences: {n_seq_total:,}")
    print(f"  - Brain region columns in parquet: {len(list_brain_region_columns(df))}")
    print("\nExported (sequence × brain):")
    for name, paths in all_export_paths:
        print(f"  - {name}: {paths['full_array']}")
        print(f"            {paths['metadata']}")

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