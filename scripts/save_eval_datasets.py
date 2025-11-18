from datasets import load_dataset
from pathlib import Path
import json

# Define data sources
data_paths = [
    {'path': 'AI-MO/aimo-validation-amc', 'split': 'train', 'save_name': 'amc'},
    {'path': 'AI-MO/aimo-validation-aime', 'split': 'train', 'save_name': 'aime'},
    {'path': 'opencompass/AIME2025', 'name': 'AIME2025-I', 'split': 'test', 'save_name': 'aime25_i'},
    {'path': 'opencompass/AIME2025', 'name': 'AIME2025-II', 'split': 'test', 'save_name': 'aime25_ii'},
    {'path': 'qwedsacf/competition_math', 'split': 'train', 'save_name': 'competition_math'},
    {'path': 'openai/gsm8k', 'name': 'main', 'split': 'test', 'save_name': 'gsm8k'},
    {'path': 'HuggingFaceH4/MATH-500', 'split': 'test', 'save_name': 'math500'},
]

# Create save directory
save_dir = Path("data/eval")
save_dir.mkdir(parents=True, exist_ok=True)

# Load and save datasets
all_aime_data = []

for data_args in data_paths:
    save_name = data_args.pop('save_name')
    print(f"Loading dataset from {data_args['path']}...")
    raw_dataset = load_dataset(**data_args)
    print(raw_dataset)
    
    if save_name == 'amc':
        # Save all AMC data with only question and answer
        output_path = save_dir / 'amc' / 'test.jsonl'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to keep only question and answer
        with open(output_path, 'w') as f:
            for item in raw_dataset:
                cleaned_item = {
                    'question': item.get('problem', ''),
                    'answer': item.get('answer', '')
                }
                f.write(json.dumps(cleaned_item) + '\n')
        print(f"Saved {len(raw_dataset)} samples to {output_path}")
    
    elif save_name == 'aime':
        # Store AIME data for later processing, extract year from URL
        for item in raw_dataset:
            url = item.get('url', '')
            # Extract year from URL (e.g., "2022", "2023", etc.)
            year = None
            if '2022' in url:
                year = 'aime22'
            elif '2023' in url:
                year = 'aime23'
            elif '2024' in url:
                year = 'aime24'
            elif '2025' in url:
                year = 'aime25'
            
            all_aime_data.append({
                'question': item.get('problem', ''),
                'answer': item.get('answer', ''),
                'year': year
            })
    
    elif save_name in ['aime25_i', 'aime25_ii']:
        # Store AIME 2025 data
        for item in raw_dataset:
            all_aime_data.append({
                'question': item.get('question', ''),
                'answer': item.get('answer', ''),
                'year': 'aime25'
            })
    
    elif save_name == 'competition_math':
        # Split into test sets of 500 and 5000 samples
        # First create a test split with seed for reproducibility
        
        # Shuffle and take samples
        shuffled = raw_dataset.shuffle(seed=42)
        
        # Save 500 samples
        samples_500 = shuffled.select(range(500))
        output_path = save_dir / 'competition_math_500' / 'test.jsonl'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for item in samples_500:
                cleaned_item = {
                    'question': item.get('problem', ''),
                    'answer': item.get('solution', '')  # Using solution as answer
                }
                f.write(json.dumps(cleaned_item) + '\n')
        print(f"Saved {len(samples_500)} samples to {output_path}")
        
        # Save 5000 samples
        samples_5k = shuffled.select(range(5000))
        output_path = save_dir / 'competition_math_5k' / 'test.jsonl'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for item in samples_5k:
                cleaned_item = {
                    'question': item.get('problem', ''),
                    'answer': item.get('solution', '')
                }
                f.write(json.dumps(cleaned_item) + '\n')
        print(f"Saved {len(samples_5k)} samples to {output_path}")
    
    elif save_name == 'gsm8k':
        # Save all GSM8K test data
        output_path = save_dir / 'gsm8k' / 'test.jsonl'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for item in raw_dataset:
                cleaned_item = {
                    'question': item.get('question', ''),
                    'answer': item.get('answer', '')
                }
                f.write(json.dumps(cleaned_item) + '\n')
        print(f"Saved {len(raw_dataset)} samples to {output_path}")
    
    elif save_name == 'math500':
        # Save all MATH-500 test data
        output_path = save_dir / 'math500' / 'test.jsonl'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for item in raw_dataset:
                cleaned_item = {
                    'question': item.get('problem', ''),
                    'answer': item.get('answer', '')
                }
                f.write(json.dumps(cleaned_item) + '\n')
        print(f"Saved {len(raw_dataset)} samples to {output_path}")

print(f"\nTotal AIME samples collected: {len(all_aime_data)}")

# Save all AIME data combined
output_path = save_dir / 'aime' / 'test.jsonl'
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, 'w') as f:
    for item in all_aime_data:
        f.write(json.dumps(item) + '\n')

print(f"Saved {len(all_aime_data)} samples to {output_path}")

# Split AIME data by year
aime_by_year = {
    'aime22': [],
    'aime23': [],
    'aime24': [],
    'aime25': []
}

for item in all_aime_data:
    year = item.get('year')
    if year and year in aime_by_year:
        aime_by_year[year].append({
            'question': item['question'],
            'answer': item['answer']
        })

# Save each year's data (limit to 30 samples each)
for year, data in aime_by_year.items():
    # Take first 30 samples
    data_to_save = data[:30]
    
    if len(data_to_save) > 0:
        output_path = save_dir / year / 'test.jsonl'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as jsonl
        with open(output_path, 'w') as f:
            for item in data_to_save:
                f.write(json.dumps(item) + '\n')
        
        print(f"Saved {len(data_to_save)} samples to {output_path}")
    else:
        print(f"Warning: No data found for {year}")

print("\nDataset processing complete!")
print(f"Files saved in: {save_dir}")
