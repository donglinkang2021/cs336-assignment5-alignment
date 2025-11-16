from datasets import load_dataset, DatasetDict
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm
from cs336_alignment.utils import load_prompt_template

# Configuration
MODEL_NAME = "models/Qwen2.5-Math-1.5B"
PROMPT_TEMPLATE = load_prompt_template("r1_zero")
NUM_PROC = 8

def format_solution(solution: str) -> str:
    """Format the generated_solution by wrapping content after </think> with <answer> tags."""
    # Find the </think> tag
    think_end = solution.find('</think>')
    
    if think_end != -1:
        # Split at </think> tag
        before_think = solution[:think_end + len('</think>')]
        after_think = solution[think_end + len('</think>'):]
        
        # Wrap the content after </think> with <answer> tags
        formatted_solution = before_think + f"\n<answer>\n{after_think.strip()}\n</answer>"
        return formatted_solution
    
    return solution

def main():
    # Load tokenizer
    print(f"Loading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Load dataset
    data_args = {'path': 'nvidia/OpenMathReasoning', 'split': 'cot'}
    print(f"Loading dataset from {data_args['path']}...")
    raw_dataset = load_dataset(**data_args)
    print(f"Total samples: {len(raw_dataset)}")
    
    # Add float version of pass_rate for sorting
    def add_pass_rate_float(example):
        try:
            example['pass_rate_float'] = float(example['pass_rate_72b_tir'])
        except (ValueError, TypeError):
            example['pass_rate_float'] = -1.0
        return example
    
    print("Adding pass_rate_float field...")
    dataset_with_float = raw_dataset.map(add_pass_rate_float, num_proc=NUM_PROC)
    
    # Filter out records with invalid pass_rate (n/a)
    print("Filtering out invalid pass_rate values...")
    valid_dataset = dataset_with_float.filter(lambda x: x['pass_rate_float'] >= 0, num_proc=NUM_PROC)
    print(f"Valid samples (after filtering n/a): {len(valid_dataset)}")
    
    # Sort by pass_rate_float in descending order
    print("Sorting by pass_rate...")
    sorted_dataset = valid_dataset.sort('pass_rate_float', reverse=True)
    
    # Format solutions and calculate token lengths using batch processing
    print("\nFormatting solutions and calculating token lengths...")
    
    def process_and_tokenize(examples):
        """Process examples in batch: format solutions and calculate token lengths."""
        prompts = examples['problem']
        solutions = examples['generated_solution']
        
        # Format solutions
        formatted_solutions = []
        for solution in solutions:
            formatted_solution = format_solution(solution.lstrip("<think>\n"))
            formatted_solutions.append(formatted_solution)
        
        # Prepare texts for tokenization
        texts = [
            PROMPT_TEMPLATE.format(question=prompt) + response
            for prompt, response in zip(prompts, formatted_solutions)
        ]
        
        # Batch tokenize
        tokenized = tokenizer(texts, add_special_tokens=True)
        
        # Calculate token lengths
        token_lengths = [len(ids) for ids in tokenized['input_ids']]
        
        return {
            'formatted_solution': formatted_solutions,
            'token_length': token_lengths
        }
    
    # Apply batch processing
    processed_dataset = sorted_dataset.map(
        process_and_tokenize,
        batched=True,
        batch_size=1000,
        num_proc=NUM_PROC,
        desc="Processing and tokenizing"
    )
    
    # Filter by token length and select top samples
    print("\nFiltering by token length...")
    
    # Filter for 2048
    dataset_2048 = processed_dataset.filter(lambda x: x['token_length'] < 2048, num_proc=NUM_PROC)
    print(f"Found {len(dataset_2048)} samples with token length < 2048")
    
    # Filter for 4096
    dataset_4096 = processed_dataset.filter(lambda x: x['token_length'] < 4096, num_proc=NUM_PROC)
    print(f"Found {len(dataset_4096)} samples with token length < 4096")
    
    # Update generated_solution with formatted version
    def update_solution(example):
        example['generated_solution'] = example['formatted_solution']
        return example
    
    print(f"\nCollected {min(len(dataset_2048), 12500)} samples for OMR12k-2048")
    print(f"Collected {min(len(dataset_4096), 12500)} samples for OMR12k-4096")
    
    # Process OMR12k-2048
    if len(dataset_2048) >= 12500:
        print("\nProcessing OMR12k-2048...")
        # Take exactly 12500 samples
        dataset_2048_selected = dataset_2048.select(range(12500))
        
        # Update generated_solution with formatted version and remove temp columns
        dataset_2048_selected = dataset_2048_selected.map(update_solution, num_proc=NUM_PROC)
        dataset_2048_selected = dataset_2048_selected.remove_columns(['pass_rate_float', 'formatted_solution', 'token_length'])
        
        # Split into train (12000) and validation (500)
        split_dataset_2048 = dataset_2048_selected.train_test_split(test_size=500, seed=42)
        
        # Create save directory
        save_dir_2048 = Path("data/OMR12k-2048")
        save_dir_2048.mkdir(parents=True, exist_ok=True)
        
        # Save as jsonl format
        print("Saving OMR12k-2048...")
        split_dataset_2048['train'].to_json(str(save_dir_2048 / 'train.jsonl'))
        split_dataset_2048['test'].to_json(str(save_dir_2048 / 'validation.jsonl'))
        
        print(f"Dataset saved to {save_dir_2048}")
        print(f"Train samples: {len(split_dataset_2048['train'])}")
        print(f"Validation samples: {len(split_dataset_2048['test'])}")
    else:
        print(f"\nWarning: Only found {len(dataset_2048)} samples with token length < 2048")
        print("Not enough samples to create OMR12k-2048")
    
    # Process OMR12k-4096
    if len(dataset_4096) >= 12500:
        print("\nProcessing OMR12k-4096...")
        # Take exactly 12500 samples
        dataset_4096_selected = dataset_4096.select(range(12500))
        
        # Update generated_solution with formatted version and remove temp columns
        dataset_4096_selected = dataset_4096_selected.map(update_solution, num_proc=NUM_PROC)
        dataset_4096_selected = dataset_4096_selected.remove_columns(['pass_rate_float', 'formatted_solution', 'token_length'])
        
        # Split into train (12000) and validation (500)
        split_dataset_4096 = dataset_4096_selected.train_test_split(test_size=500, seed=42)
        
        # Create save directory
        save_dir_4096 = Path("data/OMR12k-4096")
        save_dir_4096.mkdir(parents=True, exist_ok=True)
        
        # Save as jsonl format
        print("Saving OMR12k-4096...")
        split_dataset_4096['train'].to_json(str(save_dir_4096 / 'train.jsonl'))
        split_dataset_4096['test'].to_json(str(save_dir_4096 / 'validation.jsonl'))
        
        print(f"Dataset saved to {save_dir_4096}")
        print(f"Train samples: {len(split_dataset_4096['train'])}")
        print(f"Validation samples: {len(split_dataset_4096['test'])}")
    else:
        print(f"\nWarning: Only found {len(dataset_4096)} samples with token length < 4096")
        print("Not enough samples to create OMR12k-4096")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
