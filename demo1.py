from datasets import load_dataset

# ...existing code...
data_paths = [
    {'path': 'qwedsacf/competition_math', 'split': 'train'},
]

for data_args in data_paths:
    print(f"Loading dataset from {data_args['path']}...")
    raw_dataset = load_dataset(**data_args)
    print(raw_dataset)
    print(raw_dataset[0])