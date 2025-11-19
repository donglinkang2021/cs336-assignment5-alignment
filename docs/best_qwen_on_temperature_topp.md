# 评测一下不同 Temperature/Top-p 对基础模型生成效果的影响

<div align="center">
    <img src="../images/base_vs_instruct_comparison.png" width=800>
    </br>
    <caption>测试了五组 Temperature/Top-p 在 Qwen2.5-Math-1.5B 及 Qwen2.5-Math-1.5B-Instruct 上的表现 </caption>
</div>

- [visualize_results.py](#visualize_resultspy) 将 [eval_t_topp.sh](#eval_t_toppsh) 和 [eval_t_topp1.sh](#eval_t_topp1sh) 的结果可视化出来，结果如上图所示；
- [print_results_table.py](../scripts/print_results_table.py) 直接将结果文件的 summary.json 中的 aggregate_metrics 打印出来，如下表所示

结论：

- 奖励方面：
    *   在 Answer Reward（答案准确性）方面，`Qwen2.5-Math-1.5B-Instruct` 模型在所有超参数设置和数据集上的表现均远超基础模型 `Qwen2.5-Math-1.5B`。【可以说明Instruct的训练在答案准确性还是提升很多的】
    *   在 Format Reward（格式正确性）方面，基础模型 `Qwen2.5-Math-1.5B` 的得分远高于指令微调模型。【指令微调后的模型遗忘了原来模型指令遵循的的能力，及时把temperature调高也无济于事】
- 超参数方面：
    *   对于 Answer Reward，较低的 `temperature`（如 0.0 或 0.6）通常能带来更好的结果。当 `temperature` 升高到 1.0 时，两个模型的答案准确性都出现了明显下降。
    *   对于 Format Reward，`temperature` 对基础模型的影响较大，`temperature=0.0` 时得分最高；而对指令微调模型的影响则不明显，其得分普遍很低。

对比[Qwen2.5](https://arxiv.org/pdf/2409.12122)中的实际结果：

原论文是：
- `Qwen2.5-Math-1.5B` 
    - gsm8k(8-shot)=76.8
    - MATH(4-shot)=49.8
- `Qwen2.5-Math-1.5B-Instruct`(CoT)
    - gsm8k(zero-shot,pass@1)=84.8
    - MATH(zero-shot,pass@1)=75.8

我们所测得的实际最好结果是：
- `Qwen2.5-Math-1.5B`
    - gsm8k(zero-shot,pass@1)=57.9 (T=0,top_p=1/T=0.6,top_p=0.95)
    - MATH(zero-shot,pass@1)=55.2 (T=0,top_p=1)
- `Qwen2.5-Math-1.5B-Instruct`
    - gsm8k(zero-shot,pass@1)=74.5 (T=0,top_p=1)
    - MATH(zero-shot,pass@1)=75.0 (T=0.6,top_p=0.95/1.0)


如果考虑由于prompt_template带来的误差，我们用的不是CoT的模版("system": "Please reason step by step, and put your final answer within \\boxed{}.") ，并且考虑 few-shot learning 带来的性能提升估计的话，这里的评测结果还是可以对的上原文的评测结果的；

启发:
1. 后续打算就使用 temperature=0.6, top_p=0.95 作为初始的 rollout settings 去 reasoning了；
2. 肯定还是用指令微调之前的 Base 模型来进行训练，之后训练完模型之后就可以直接在现在的基础上看对比 base model 怎么样，对比 instruct model 怎么样，现在相当于把自己的benchmark已经写的差不多了，后续打算也测试一下deepseek的蒸馏模型效果如何；

## Appendix

### Answer Reward by Model Config and Dataset

| Model Config                                       |   competition_math |    gsm8k |   math500 |
|:---------------------------------------------------|-------------------:|---------:|----------:|
| Qwen2.5-Math-1.5B-Instruct-temperature0.0-topp1.0  |              0.736 | 0.745262 |     0.682 |
| Qwen2.5-Math-1.5B-Instruct-temperature0.6-topp0.95 |              0.75  | 0.741471 |     0.668 |
| Qwen2.5-Math-1.5B-Instruct-temperature0.6-topp1.0  |              0.75  | 0.731615 |     0.686 |
| Qwen2.5-Math-1.5B-Instruct-temperature1.0-topp0.95 |              0.674 | 0.595906 |     0.606 |
| Qwen2.5-Math-1.5B-Instruct-temperature1.0-topp1.0  |              0.58  | 0.514784 |     0.5   |
| Qwen2.5-Math-1.5B-temperature0.0-topp1.0           |              0.552 | 0.579227 |     0.5   |
| Qwen2.5-Math-1.5B-temperature0.6-topp0.95          |              0.472 | 0.579227 |     0.494 |
| Qwen2.5-Math-1.5B-temperature0.6-topp1.0           |              0.456 | 0.575436 |     0.43  |
| Qwen2.5-Math-1.5B-temperature1.0-topp0.95          |              0.364 | 0.470053 |     0.35  |
| Qwen2.5-Math-1.5B-temperature1.0-topp1.0           |              0.304 | 0.351024 |     0.292 |


### Format Reward by Model Config and Dataset

| Model Config                                       |   competition_math |      gsm8k |   math500 |
|:---------------------------------------------------|-------------------:|-----------:|----------:|
| Qwen2.5-Math-1.5B-Instruct-temperature0.0-topp1.0  |              0     | 0          |     0.004 |
| Qwen2.5-Math-1.5B-Instruct-temperature0.6-topp0.95 |              0.004 | 0.00227445 |     0.002 |
| Qwen2.5-Math-1.5B-Instruct-temperature0.6-topp1.0  |              0.002 | 0.00530705 |     0.006 |
| Qwen2.5-Math-1.5B-Instruct-temperature1.0-topp0.95 |              0.004 | 0.0060652  |     0.006 |
| Qwen2.5-Math-1.5B-Instruct-temperature1.0-topp1.0  |              0.002 | 0.0015163  |     0     |
| Qwen2.5-Math-1.5B-temperature0.0-topp1.0           |              0.488 | 0.780895   |     0.506 |
| Qwen2.5-Math-1.5B-temperature0.6-topp0.95          |              0.402 | 0.576194   |     0.444 |
| Qwen2.5-Math-1.5B-temperature0.6-topp1.0           |              0.424 | 0.544352   |     0.392 |
| Qwen2.5-Math-1.5B-temperature1.0-topp0.95          |              0.312 | 0.334344   |     0.27  |
| Qwen2.5-Math-1.5B-temperature1.0-topp1.0           |              0.21  | 0.278999   |     0.214 |


### eval_t_topp.sh

```bash
#!/bin/bash
# Add error handling
set -e  # Exit immediately on error
set -u  # Error on unset variables
set -o pipefail  # Fail a pipeline if any command fails

export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=3

# Use more descriptive variable names and add read-only protection
readonly MODEL_PATH="Qwen/Qwen2.5-Math-1.5B"

# Define temperature and top_p combinations to test
# Format: "temperature:top_p"
COMBINATIONS=(
    "0.0:1.0"
    "0.6:0.95"
    "0.6:1.0"
    "1.0:0.95"
    "1.0:1.0"
)

echo "Starting grid search evaluation..."
echo "Total combinations to test: ${#COMBINATIONS[@]}"
echo "-------------------------------------------"

# Loop through each combination
for combo in "${COMBINATIONS[@]}"; do
    # Split the combination into temperature and top_p
    IFS=':' read -r temp topp <<< "$combo"
    
    SAVE_PATH="Qwen/Qwen2.5-Math-1.5B-temperature${temp}-topp${topp}"
    
    echo ""
    echo "==================================================="
    echo "Testing: temperature=${temp}, top_p=${topp}"
    echo "==================================================="
    
    # Run evaluation
    uv run cs336_alignment/evaluate_math.py \
        model.model_name_or_path="$MODEL_PATH" \
        generation.temperature="$temp" \
        generation.top_p="$topp" \
        datasets=[competition_math,gsm8k,math500] \
        output_dir="eval_results1/${SAVE_PATH}/"
    
    echo "Completed: temperature=${temp}, top_p=${topp}"
done

echo ""
echo "==================================================="
echo "All evaluations completed successfully!"
echo "==================================================="
```

### eval_t_topp1.sh

```bash
#!/bin/bash
# Add error handling
set -e  # Exit immediately on error
set -u  # Error on unset variables
set -o pipefail  # Fail a pipeline if any command fails

export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=2

# Use more descriptive variable names and add read-only protection
readonly MODEL_PATH="Qwen/Qwen2.5-Math-1.5B-Instruct"

# Define temperature and top_p combinations to test
# Format: "temperature:top_p"
COMBINATIONS=(
    "0.0:1.0"
    "0.6:0.95"
    "0.6:1.0"
    "1.0:0.95"
    "1.0:1.0"
)

echo "Starting grid search evaluation..."
echo "Total combinations to test: ${#COMBINATIONS[@]}"
echo "-------------------------------------------"

# Loop through each combination
for combo in "${COMBINATIONS[@]}"; do
    # Split the combination into temperature and top_p
    IFS=':' read -r temp topp <<< "$combo"
    
    SAVE_PATH="Qwen/Qwen2.5-Math-1.5B-Instruct-temperature${temp}-topp${topp}"
    
    echo ""
    echo "==================================================="
    echo "Testing: temperature=${temp}, top_p=${topp}"
    echo "==================================================="
    
    # Run evaluation
    uv run cs336_alignment/evaluate_math.py \
        model.model_name_or_path="$MODEL_PATH" \
        generation.temperature="$temp" \
        generation.top_p="$topp" \
        datasets=[competition_math,gsm8k,math500] \
        output_dir="eval_results1/${SAVE_PATH}/"
    
    echo "Completed: temperature=${temp}, top_p=${topp}"
done

echo ""
echo "==================================================="
echo "All evaluations completed successfully!"
echo "==================================================="
```

### visualize_results.py

```python
#!/usr/bin/env python3
"""
Visualize evaluation results for different temperature and top_p combinations.
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


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
```