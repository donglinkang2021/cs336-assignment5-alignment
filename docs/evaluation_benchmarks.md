# Custom Benchmarks

- [save_eval_datasets.py](../scripts/save_eval_datasets.py) 从现有的数据集中搭建了自己的 benchmarks，方便后续评测;

执行下面的程序可以简单加载查看一下评测数据集；

```python
from datasets import load_dataset

# ...existing code...
data_paths = [
    # eval datasets
    # {'path': 'AI-MO/aimo-validation-amc', 'split': 'train'},
    # {'path': 'AI-MO/aimo-validation-aime', 'split': 'train'},
    # {'path': 'opencompass/AIME2025', 'name': 'AIME2025-I', 'split': 'test'},
    # {'path': 'opencompass/AIME2025', 'name': 'AIME2025-II', 'split': 'test'},
    # {'path': 'qwedsacf/competition_math', 'split': 'train'},
    # {'path': 'openai/gsm8k', 'name': 'main', 'split': 'test'},
    # {'path': 'HuggingFaceH4/MATH-500', 'split': 'test'}
    {
        'path': "json",
        'data_files': "data/eval/competition_math_500/test.jsonl",
        'split': "train"
    },
    {
        'path': "json",
        'data_files': "data/eval/competition_math_5k/test.jsonl",
        'split': "train"
    },
    {
        'path': "json",
        'data_files': "data/eval/aime/test.jsonl",
        'split': "train"
    },
    {
        'path': "json",
        'data_files': "data/eval/aime22/test.jsonl",
        'split': "train"
    },
    {
        'path': "json",
        'data_files': "data/eval/aime23/test.jsonl",
        'split': "train"
    },
    {
        'path': "json",
        'data_files': "data/eval/aime24/test.jsonl",
        'split': "train"
    },
    {
        'path': "json",
        'data_files': "data/eval/aime25/test.jsonl",
        'split': "train"
    },
    {
        'path': "json",
        'data_files': "data/eval/amc/test.jsonl",
        'split': "train"
    },
    {
        'path': "json",
        'data_files': "data/eval/math500/test.jsonl",
        'split': "train"
    },
    {
        'path': "json",
        'data_files': "data/eval/gsm8k/test.jsonl",
        'split': "train"
    },
]

for data_args in data_paths:
    print(f"Loading dataset from {data_args['data_files']}...")
    raw_dataset = load_dataset(**data_args)
    print(raw_dataset)
    print(raw_dataset[0])

# python demo.py
```