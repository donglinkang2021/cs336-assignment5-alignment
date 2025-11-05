# 1 Assignment Overview

We provide—as an entirely optional supplement to the required course materials—an assignment on training language models to follow instructions and aligning language models to pairwise preference judgments.

### What you will implement

1.  Zero-shot prompting baselines for a variety of evaluation datasets.
2.  Supervised fine-tuning, given demonstration data with instruction-response pairs.
3.  Direct preference optimization (DPO) for learning from pairwise preference data.

### What you will run

1.  Measure Llama 3.1 zero-shot prompting performance (our baseline).
2.  Instruction fine-tune Llama 3.1.
3.  Fine-tune Llama 3.1 on pairwise preference data.

### What the code looks like

All the assignment code as well as this writeup are available on GitHub at: [github.com/stanford-cs336/assignment5-alignment](https://github.com/stanford-cs336/assignment5-alignment)

Please `git clone` the repository. If there are any updates, we will notify you and you can `git pull` to get the latest.

1.  `cs336_alignment/*`: This is where you’ll write your code for assignment 5. Note that there’s no code in here, so you should be able to do whatever you want from scratch.
2.  `cs336_alignment/prompts/*`: For your convenience, we’ve provided text files with the zero-shot system prompt and the Alpaca instruction-tuning prompt, to minimize possible errors caused by copying-and-pasting prompts from the PDF to your code.
3.  `tests/*.py`: This contains all the tests that you must pass. Specifically, for this supplemental assignment, you will be using the tests in `tests/test_data.py`, `tests/test_dpo.py`, `tests/test_metrics.py`, and `tests/test_sft.py`. These tests invoke the hooks defined in `tests/adapters.py`. You’ll implement the adapters to connect your code to the tests. Writing more tests and/or modifying the test code can be helpful for debugging your code, but your implementation is expected to pass the original provided test suite.
4.  `data/*`: This folder contains the benchmark datasets that we’ll be using to evaluate our models: MMLU, GSM8K, AlpacaEval, and SimpleSafetyTests.
5.  `scripts/alpaca_eval_vllm_llama3_3_70b_fn/`: This file contains an evaluation config for AlpacaEval that uses Llama 3.3 70B Instruct to judge generated responses against a reference.
6.  `README.md`: This file contains some basic instructions on setting up your environment.

### What you can use

As in the main assignment, we expect you to build these components from scratch. You may use tools like `vLLM` to generate text from language models, and use Huggingface Transformers to load the Llama 3.* models and tokenizers (refer to the main assignment handout for a walkthrough on these tools). Again, you may not use any of the training utilities (e.g., the `Trainer` class).
