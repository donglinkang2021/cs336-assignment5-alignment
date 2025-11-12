from datasets import load_dataset, DatasetDict
from pathlib import Path

# Load the existing OMR12k dataset
data_dir = Path("data/OMR12k")
print(f"Loading dataset from {data_dir}...")

# Load train and validation splits
train_dataset = load_dataset('json', data_files=str(data_dir / 'train.jsonl'), split='train')
validation_dataset = load_dataset('json', data_files=str(data_dir / 'validation.jsonl'), split='train')

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(validation_dataset)}")

# Format the generated_solution by wrapping content after </think> with <answer> tags
def format_solution(example):
    solution = example['generated_solution']
    # Find the </think> tag
    think_end = solution.find('</think>')
    
    if think_end != -1:
        # Split at </think> tag
        before_think = solution[:think_end + len('</think>')]
        after_think = solution[think_end + len('</think>'):]
        
        # Wrap the content after </think> with <answer> tags
        formatted_solution = before_think + f"\n<answer>\n{after_think.strip()}\n</answer>"
        example['generated_solution'] = formatted_solution
    
    return example

# Apply formatting to both splits
print("Formatting train dataset...")
train_formatted = train_dataset.map(format_solution)
print("Formatting validation dataset...")
validation_formatted = validation_dataset.map(format_solution)

# Create DatasetDict
formatted_dataset = DatasetDict({
    'train': train_formatted,
    'validation': validation_formatted
})

# Create save directory
save_dir = Path("data/OMR12k-formated")
save_dir.mkdir(parents=True, exist_ok=True)

# Save as jsonl format
print(f"Saving formatted dataset to {save_dir}...")
formatted_dataset['train'].to_json(str(save_dir / 'train.jsonl'))
formatted_dataset['validation'].to_json(str(save_dir / 'validation.jsonl'))

print(f"Dataset saved to {save_dir}")
print(f"Train samples: {len(formatted_dataset['train'])}")
print(f"Validation samples: {len(formatted_dataset['validation'])}")

# Print a sample to verify formatting
print("\nSample formatted solution (first 500 chars):")
print(formatted_dataset['train'][0]['generated_solution'])

# Optional: Upload to Hugging Face
# formatted_dataset.push_to_hub("your-username/OMR12k-formated")
