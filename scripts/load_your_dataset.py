from datasets import load_dataset

dataset = load_dataset('json', data_files={
    'train': 'data/Math/train.jsonl',
    'validation': 'data/Math/validation.jsonl'
})

print(dataset)