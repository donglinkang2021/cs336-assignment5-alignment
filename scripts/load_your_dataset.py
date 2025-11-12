from datasets import load_dataset
from pathlib import Path

def load_prompt_template(prompt_name: str) -> str:
    """Load a prompt template from the prompts directory."""
    prompt_path = Path("cs336_alignment") / "prompts" / f"{prompt_name}.prompt"
    with open(prompt_path) as f:
        return f.read()

template = load_prompt_template("r1_zero")

data_dir = "data/OMR12k-formated"

dataset = load_dataset('json', data_files={
    'train': f'{data_dir}/train.jsonl',
    'validation': f'{data_dir}/validation.jsonl'
})

question = dataset['train'][0]['problem']
solution = dataset['train'][0]['generated_solution']

print(dataset['train'][0].keys())
print(template.format(question=question))
print(solution)