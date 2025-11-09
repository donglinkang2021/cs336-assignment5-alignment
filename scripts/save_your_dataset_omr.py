from datasets import load_dataset, DatasetDict
from pathlib import Path

data_args = {'path': 'nvidia/OpenMathReasoning', 'split': 'cot'}

print(f"Loading dataset from {data_args['path']}...")
raw_dataset = load_dataset(**data_args)
print(raw_dataset)
print(f"Total samples: {len(raw_dataset)}")

# Convert pass_rate_72b_tir to float and sort
def add_pass_rate_float(example):
    try:
        example['pass_rate_float'] = float(example['pass_rate_72b_tir'])
    except (ValueError, TypeError):
        # Handle 'n/a' or other non-numeric values by setting to -1 (will be filtered out)
        example['pass_rate_float'] = -1.0
    return example

# Add float version of pass_rate for sorting
dataset_with_float = raw_dataset.map(add_pass_rate_float)

# Filter out records with invalid pass_rate (n/a)
valid_dataset = dataset_with_float.filter(lambda x: x['pass_rate_float'] >= 0)
print(f"Valid samples (after filtering n/a): {len(valid_dataset)}")

# Sort by pass_rate_float in descending order and select top 12500
sorted_dataset = valid_dataset.sort('pass_rate_float', reverse=True)
top_dataset = sorted_dataset.select(range(12500))

# Remove the temporary float column
top_dataset = top_dataset.remove_columns(['pass_rate_float'])

# Split into train (12000) and validation (500)
split_dataset = top_dataset.train_test_split(test_size=500, seed=42)

# Create save directory
save_dir = Path("data/OMR12k")
save_dir.mkdir(parents=True, exist_ok=True)

# Save as jsonl format
split_dataset['train'].to_json(str(save_dir / 'train.jsonl'))
split_dataset['test'].to_json(str(save_dir / 'validation.jsonl'))

print(f"Dataset saved to {save_dir}")
print(f"Train samples: {len(split_dataset['train'])}")
print(f"Validation samples: {len(split_dataset['test'])}")
print(f"Sample pass_rate range: {split_dataset['train'][0]['pass_rate_72b_tir']} (highest) to {split_dataset['train'][-1]['pass_rate_72b_tir']}")

# Optional: Upload to Hugging Face
dataset_dict = DatasetDict({
    'train': split_dataset['train'],
    'validation': split_dataset['test']
})
# dataset_dict.push_to_hub("your-username/OMR12k")