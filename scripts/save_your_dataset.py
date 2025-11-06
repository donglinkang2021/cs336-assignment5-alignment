from datasets import load_dataset, DatasetDict
from pathlib import Path

# ...existing code...
data_args = {'path': 'qwedsacf/competition_math', 'split': 'train'}

print(f"Loading dataset from {data_args['path']}...")
raw_dataset = load_dataset(**data_args)
print(raw_dataset)
print(raw_dataset[0])

# Split dataset into train and validation sets (12000 train, 500 validation)
split_dataset = raw_dataset.train_test_split(test_size=500, seed=42)

# Create save directory
save_dir = Path("data/Math")
save_dir.mkdir(parents=True, exist_ok=True)

# Save as jsonl format (recommended for uploading to Hugging Face)
split_dataset['train'].to_json(str(save_dir / 'train.jsonl'))
split_dataset['test'].to_json(str(save_dir / 'validation.jsonl'))

print(f"Dataset saved to {save_dir}")
print(f"Train samples: {len(split_dataset['train'])}")
print(f"Validation samples: {len(split_dataset['test'])}")

# Upload to Hugging Face (requires login: huggingface-cli login)
dataset_dict = DatasetDict({
    'train': split_dataset['train'],
    'validation': split_dataset['test']
})
# dataset_dict.push_to_hub("your-username/your-dataset-name")
