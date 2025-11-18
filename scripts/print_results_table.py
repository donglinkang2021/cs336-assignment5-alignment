from pathlib import Path
import json
import pandas as pd

base_dir = Path("eval_results1")

# Collect data from all summary.json files
data = []

# Recursively find all summary.json files
for summary_file in base_dir.rglob("summary.json"):
    with open(summary_file, 'r') as f:
        summary_data = json.load(f)
    
    # Extract path information
    parts = summary_file.relative_to(base_dir).parts
    model_type = parts[0] if len(parts) > 0 else "unknown"
    model_config = parts[1] if len(parts) > 1 else "unknown"
    dataset = parts[2] if len(parts) > 2 else "unknown"
    
    # Extract aggregate_metrics
    metrics = summary_data.get("aggregate_metrics", {})
    answer_reward = metrics.get("answer_reward", None)
    format_reward = metrics.get("format_reward", None)
    reward = metrics.get("reward", None)
    
    data.append({
        "Model Type": model_type,
        "Model Config": model_config,
        "Dataset": dataset,
        "Answer Reward": answer_reward,
        "Format Reward": format_reward,
        "Reward": reward,
        "File Path": str(summary_file.relative_to(base_dir.parent))
    })

# Create DataFrame
df = pd.DataFrame(data)

# Sort by Model Config and Dataset
df = df.sort_values(by=["Model Config", "Dataset"])

# Save as CSV
output_csv = base_dir / "metrics_summary.csv"
df.to_csv(output_csv, index=False)
print(f"Table saved to: {output_csv}\n")

# Display full table in Markdown format
print("# Summary of Answer Reward and Format Reward\n")
print(df.to_markdown(index=False))
print("\n")

# Create pivot tables for easier comparison
print("# Pivot Table - Answer Reward by Model Config and Dataset\n")
pivot_answer = df.pivot_table(
    values='Answer Reward', 
    index='Model Config', 
    columns='Dataset', 
    aggfunc='first'
)
print(pivot_answer.to_markdown())
print("\n")

print("# Pivot Table - Format Reward by Model Config and Dataset\n")
pivot_format = df.pivot_table(
    values='Format Reward', 
    index='Model Config', 
    columns='Dataset', 
    aggfunc='first'
)
print(pivot_format.to_markdown())
print("\n")

print("# Pivot Table - Total Reward by Model Config and Dataset\n")
pivot_reward = df.pivot_table(
    values='Reward', 
    index='Model Config', 
    columns='Dataset', 
    aggfunc='first'
)
print(pivot_reward.to_markdown())
print("\n")
