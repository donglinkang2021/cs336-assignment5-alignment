#!/usr/bin/env python3
"""
Visualize evaluation results for different temperature and top_p combinations.
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse


def load_summary(result_dir: Path) -> Dict:
    """Load summary.json from a result directory."""
    summary_path = result_dir / "summary.json"
    if not summary_path.exists():
        return None
    
    with open(summary_path, 'r') as f:
        return json.load(f)


def collect_results(base_dir: Path, model_prefix: str) -> Dict:
    """
    Collect all results from the evaluation directory.
    
    Returns:
        Dict mapping (temperature, top_p) -> {dataset: accuracy}
    """
    results = {}
    
    # Find all model result directories
    for model_dir in base_dir.glob(f"{model_prefix}-temperature*"):
        if not model_dir.is_dir():
            continue
        
        # Extract temperature and top_p from directory name
        dir_name = model_dir.name
        try:
            # Parse: Qwen2.5-Math-1.5B-temperature0.6-topp0.95
            parts = dir_name.split('-temperature')
            if len(parts) != 2:
                continue
            
            temp_topp = parts[1].split('-topp')
            if len(temp_topp) != 2:
                continue
            
            temperature = float(temp_topp[0])
            top_p = float(temp_topp[1])
            
            config_key = (temperature, top_p)
            results[config_key] = {}
            
            # Load results for each dataset
            datasets = ['competition_math', 'gsm8k', 'math500']
            for dataset in datasets:
                summary = load_summary(model_dir / dataset)
                if summary and 'aggregate_metrics' in summary:
                    accuracy = summary['aggregate_metrics'].get('answer_reward', 0.0)
                    results[config_key][dataset] = accuracy
        
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse directory {dir_name}: {e}")
            continue
    
    return results


def plot_grouped_bar_chart(results: Dict, output_path: Path, title: str):
    """
    Create a grouped bar chart comparing different configurations.
    
    Each group represents a (temperature, top_p) combination.
    Each bar in a group represents a different dataset.
    """
    if not results:
        print("No results to plot!")
        return
    
    datasets = ['competition_math', 'gsm8k', 'math500']
    dataset_labels = ['Competition Math', 'GSM8K', 'MATH500']
    
    # Sort configurations by temperature, then top_p
    configs = sorted(results.keys())
    
    # Prepare data
    x = np.arange(len(configs))
    width = 0.25  # Width of each bar
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot bars for each dataset
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    for i, (dataset, label, color) in enumerate(zip(datasets, dataset_labels, colors)):
        accuracies = [results[config].get(dataset, 0.0) for config in configs]
        offset = width * (i - 1)
        bars = ax.bar(x + offset, accuracies, width, label=label, color=color, alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=8)
    
    # Customize plot
    ax.set_xlabel('Configuration (Temperature, Top-p)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (Answer Reward)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'T={t:.1f}\np={p:.2f}' for t, p in configs], fontsize=9)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max([max(results[c].values()) for c in configs if results[c]]) * 1.15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()


def plot_comparison_chart(results_by_model: Dict[str, Dict[Tuple[float, float], Dict[str, float]]],
                          output_path: Path,
                          title: str):
    """
    Plot comparison of multiple models (e.g., base vs instruct) on the same grouped chart.

    results_by_model: mapping from model label -> results dict (same format as collect_results output)
    """
    if not results_by_model:
        print("No results to plot!")
        return

    datasets = ['competition_math', 'gsm8k', 'math500']
    dataset_labels = ['Competition Math', 'GSM8K', 'MATH500']

    # All configs across all models (sorted)
    all_configs = sorted({c for r in results_by_model.values() for c in r.keys()})
    if not all_configs:
        print("No configurations found in results!")
        return

    model_labels = list(results_by_model.keys())
    n_models = len(model_labels)
    n_datasets = len(datasets)
    n_configs = len(all_configs)

    # Bar width for overlapping bars - datasets are spaced, but models overlap
    group_width = 0.8
    bar_width = group_width / n_datasets
    
    x = np.arange(n_configs)

    fig, ax = plt.subplots(figsize=(14, 6))

    # Use original color scheme for all models
    original_colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    palettes = {}
    for model_label in model_labels:
        palettes[model_label] = original_colors

    # Draw overlapping bars: instruct first (background, transparent), then base (foreground, opaque)
    # Sort model_labels so instruct is drawn first (background)
    sorted_models = sorted(model_labels, key=lambda x: 1 if "Instruct" not in x else 0)
    
    for cfg_idx, cfg in enumerate(all_configs):
        for d_idx, dataset in enumerate(datasets):
            # Position for this dataset group
            offset = (d_idx - (n_datasets - 1) / 2) * bar_width
            xpos = x[cfg_idx] + offset
            
            for m_idx, model_label in enumerate(sorted_models):
                accuracy = results_by_model.get(model_label, {}).get(cfg, {}).get(dataset, 0.0)
                color = palettes.get(model_label, palettes[model_labels[0]])[d_idx]
                
                # Instruct model: background, more transparent (alpha=0.5)
                # Base model: foreground, opaque (alpha=0.95), on top
                is_base = "Instruct" not in model_label
                alpha_val = 0.95 if is_base else 0.5
                zorder = 2 if is_base else 1
                
                # Only add label for the first config
                label = None
                if cfg_idx == 0:
                    model_short = "Base" if is_base else "Instruct"
                    label = f"{dataset_labels[d_idx]} ({model_short})"
                
                bar = ax.bar(xpos, accuracy, bar_width, 
                           color=color, alpha=alpha_val, label=label, zorder=zorder,
                           edgecolor='white', linewidth=0.5)
                
                # Add value labels on top of all bars with uniform formatting
                if accuracy > 0:
                    ax.text(xpos, accuracy, f'{accuracy:.3f}', 
                           ha='center', va='bottom', fontsize=8, fontweight='bold')

    # X ticks and labels
    ax.set_xticks(x)
    ax.set_xticklabels([f'T={t:.1f}\np={p:.2f}' for t, p in all_configs], fontsize=9)
    ax.set_xlabel('Configuration (Temperature, Top-p)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (Answer Reward)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Build a compact legend: show one legend entry per model+dataset (we added only for cfg_idx==0)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # ax.legend(handles, labels, bbox_to_anchor=(1.02, 1), loc='upper right', fontsize=9)
        ax.legend(handles, labels, loc='upper right', fontsize=9)

    # y limit based on maximum observed accuracy
    max_val = 0.0
    for r in results_by_model.values():
        for v in r.values():
            if v:
                max_val = max(max_val, max(v.values()))
    if max_val <= 0:
        ax.set_ylim(0, 1.0)
    else:
        ax.set_ylim(0, max_val * 1.15)

    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {output_path}")
    plt.close()

def plot_single(
    base_dir: str = "eval_results1",
    model_prefix: str = "Qwen/Qwen2.5-Math-1.5B",
    output_dir: str = "images",
    title: str = "Base Model: Temperature and Top-p Comparison"
):    
    base_dir = Path(base_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Collect results
    print(f"Collecting results from {base_dir}...")
    results = collect_results(base_dir, model_prefix)
    
    if not results:
        print("No results found!")
        return
    
    print(f"Found {len(results)} configurations:")
    for config in sorted(results.keys()):
        print(f"  Temperature={config[0]:.1f}, Top-p={config[1]:.2f}: {results[config]}")
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Grouped bar chart
    model_name = model_prefix.split('/')[-1]
    output_path = output_dir / f'{model_name}_comparison.png'
    plot_grouped_bar_chart(results, output_path, title)
    
    print("\nVisualization complete!")

def plot_multi(
    base_dir:str = 'eval_results1',
    base_prefix:str = 'Qwen/Qwen2.5-Math-1.5B',
    instr_prefix:str = 'Qwen/Qwen2.5-Math-1.5B-Instruct',
    title:str = 'Base vs Instruct: Temperature/Top-p Comparison'
):
    base_dir = Path(base_dir)
    base_results = collect_results(base_dir, base_prefix)
    instr_results = collect_results(base_dir, instr_prefix)
    results_by_model = {
        base_prefix: base_results,
        instr_prefix: instr_results,
    }
    plot_comparison_chart(
        results_by_model, 
        Path('images/base_vs_instruct_comparison.png'),
        title=title
    )

if __name__ == '__main__':
    # plot_single(
    #     model_prefix="Qwen/Qwen2.5-Math-1.5B",
    #     title="Base Model: Temperature and Top-p Comparison"
    # )
    # plot_single(
    #     model_prefix="Qwen/Qwen2.5-Math-1.5B-Instruct",
    #     title="Instruct Model: Temperature and Top-p Comparison"
    # )
    plot_multi(
        base_prefix='Qwen/Qwen2.5-Math-1.5B',
        instr_prefix='Qwen/Qwen2.5-Math-1.5B-Instruct',
        title='Base vs Instruct: Temperature/Top-p Comparison'
    )
    

# python scripts/visualize_results.py