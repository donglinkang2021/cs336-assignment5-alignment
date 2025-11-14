"""
Visualize batch size benchmark results as heatmaps.
"""

import json
import glob
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

def load_jsonl_files(directory):
    """Load all JSONL files from directory and organize by sequence length."""
    data_by_seq_len = {}
    
    jsonl_files = sorted(glob.glob(f"{directory}/*.jsonl"))
    
    for file_path in jsonl_files:
        with open(file_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                seq_len = entry['sequence_length']
                
                if seq_len not in data_by_seq_len:
                    data_by_seq_len[seq_len] = []
                
                data_by_seq_len[seq_len].append(entry)
    
    return data_by_seq_len

def create_config_label(entry):
    """Create a short label for each configuration."""
    opt = "MemEffAdamW" if entry['optimizer'] == "MemoryEfficientAdamW" else "AdamW"
    gc = "GC" if entry['gradient_checkpointing'] else "noGC"
    fa = "FA" if entry['flash_attention'] else "noFA"
    return f"{opt}\n{gc}+{fa}"

def plot_heatmaps(data_by_seq_len, output_path="benchmark_heatmap.png"):
    """Create dual heatmaps for batch size and memory usage."""
    
    # Sort sequence lengths
    seq_lengths = sorted(data_by_seq_len.keys())
    
    # Get configurations from first sequence length
    first_seq = seq_lengths[0]
    configs = data_by_seq_len[first_seq]
    
    # Sort configurations for consistent ordering
    configs = sorted(configs, key=lambda x: (
        x['optimizer'],
        not x['gradient_checkpointing'],
        not x['flash_attention']
    ))
    
    config_labels = [create_config_label(c) for c in configs]
    
    # Prepare data matrices
    batch_size_matrix = []
    memory_matrix = []
    
    for seq_len in seq_lengths:
        batch_sizes = []
        memories = []
        
        # Sort entries same way as configs
        entries = sorted(data_by_seq_len[seq_len], key=lambda x: (
            x['optimizer'],
            not x['gradient_checkpointing'],
            not x['flash_attention']
        ))
        
        for entry in entries:
            batch_sizes.append(entry['max_batch_size'])
            memories.append(entry['peak_memory_gb'])
        
        batch_size_matrix.append(batch_sizes)
        memory_matrix.append(memories)
    
    batch_size_matrix = np.array(batch_size_matrix)
    memory_matrix = np.array(memory_matrix)
    
    # Get model name
    model_name = configs[0]['model_short_name']
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    
    # Plot 1: Max Batch Size Heatmap
    sns.heatmap(
        batch_size_matrix,
        annot=True,
        fmt='d',
        cmap='viridis',
        cbar_kws={'label': 'Max Batch Size'},
        xticklabels=config_labels,
        yticklabels=[f'{sl}' for sl in seq_lengths],
        ax=ax1,
        linewidths=0.5,
        linecolor='white',
        annot_kws={'fontsize': 14, 'weight': 'bold'}
    )
    ax1.set_title(f'{model_name}\nMaximum Batch Size by Configuration', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Sequence Length', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=0)
    
    # Plot 2: Peak Memory Heatmap
    sns.heatmap(
        memory_matrix,
        annot=True,
        fmt='.1f',
        cmap='viridis',
        cbar_kws={'label': 'Peak Memory (GB)'},
        xticklabels=config_labels,
        yticklabels=[f'{sl}' for sl in seq_lengths],
        ax=ax2,
        linewidths=0.5,
        linecolor='white',
        annot_kws={'fontsize': 14, 'weight': 'bold'}
    )
    ax2.set_title(f'{model_name}\nPeak Memory Usage by Configuration (GB)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Sequence Length', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to: {output_path}")
    
    # Also show the figure
    plt.show()
    
    return fig

def plot_combined_heatmap(data_by_seq_len, output_path="benchmark_combined_heatmap.png"):
    """Create a single heatmap showing both batch size and memory."""
    
    # Sort sequence lengths
    seq_lengths = sorted(data_by_seq_len.keys())
    
    # Get configurations from first sequence length
    first_seq = seq_lengths[0]
    configs = data_by_seq_len[first_seq]
    
    # Sort configurations for consistent ordering
    configs = sorted(configs, key=lambda x: (
        x['optimizer'],
        not x['gradient_checkpointing'],
        not x['flash_attention']
    ))
    
    config_labels = [create_config_label(c) for c in configs]
    
    # Prepare data matrices
    batch_size_matrix = []
    memory_matrix = []
    
    for seq_len in seq_lengths:
        batch_sizes = []
        memories = []
        
        # Sort entries same way as configs
        entries = sorted(data_by_seq_len[seq_len], key=lambda x: (
            x['optimizer'],
            not x['gradient_checkpointing'],
            not x['flash_attention']
        ))
        
        for entry in entries:
            batch_sizes.append(entry['max_batch_size'])
            memories.append(entry['peak_memory_gb'])
        
        batch_size_matrix.append(batch_sizes)
        memory_matrix.append(memories)
    
    batch_size_matrix = np.array(batch_size_matrix)
    memory_matrix = np.array(memory_matrix)
    
    # Get model name
    model_name = configs[0]['model_short_name']
    
    # Create figure with single plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Create combined annotations (batch size + memory)
    combined_annot = np.empty_like(batch_size_matrix, dtype=object)
    for i in range(batch_size_matrix.shape[0]):
        for j in range(batch_size_matrix.shape[1]):
            combined_annot[i, j] = f"BS: {batch_size_matrix[i, j]}\n{memory_matrix[i, j]:.1f}GB"
    
    # Plot heatmap with batch size as the primary metric
    sns.heatmap(
        batch_size_matrix,
        annot=combined_annot,
        fmt='',
        cmap='viridis',
        cbar_kws={'label': 'Max Batch Size'},
        xticklabels=config_labels,
        yticklabels=[f'{sl}' for sl in seq_lengths],
        ax=ax,
        linewidths=1,
        linecolor='white',
        annot_kws={'fontsize': 14, 'weight': 'bold'}
    )
    
    ax.set_title(f'{model_name} - Batch Size Benchmark\n(Batch Size & Peak Memory per Configuration)', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xlabel('Configuration', fontsize=13, fontweight='bold')
    ax.set_ylabel('Sequence Length', fontsize=13, fontweight='bold')
    ax.tick_params(axis='x', rotation=0, labelsize=10)
    ax.tick_params(axis='y', rotation=0, labelsize=11)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Combined heatmap saved to: {output_path}")
    
    # Also show the figure
    plt.show()
    
    return fig

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize batch size benchmark results")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="batch_size_benchmark",
        help="Directory containing JSONL files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_heatmap.png",
        help="Output image file path"
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Create a single combined heatmap instead of two separate ones"
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data_dir}...")
    data_by_seq_len = load_jsonl_files(args.data_dir)
    
    print(f"Found data for sequence lengths: {sorted(data_by_seq_len.keys())}")
    
    # Create visualization
    if args.combined:
        plot_combined_heatmap(data_by_seq_len, args.output)
    else:
        plot_heatmaps(data_by_seq_len, args.output)
    
    print("Done!")
