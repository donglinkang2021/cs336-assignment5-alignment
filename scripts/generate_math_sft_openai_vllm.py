import os
import json
from pathlib import Path
import argparse
import asyncio
from tqdm.asyncio import tqdm as async_tqdm

from datasets import load_dataset
from openai import AsyncOpenAI

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


API_URL = os.environ.get("API_URL", "http://0.0.0.0:8000")
API_KEY = os.environ.get("OPENAI_API_KEY", os.environ.get("API_KEY", "EMPTY"))


def create_async_client():
    """Create AsyncOpenAI client"""
    client = AsyncOpenAI(
        base_url=f"{API_URL}/v1",
        api_key=API_KEY,
    )
    return client


async def get_models_id(client):
    """Get available model IDs"""
    list_completion = await client.models.list()
    return [model.id async for model in list_completion]


async def generate_for_dataset(client, model_name, max_samples=None, resume=True, concurrency=16):
    """
    Generate solutions for competition_math dataset using async OpenAI client
    
    Args:
        client: AsyncOpenAI client instance
        model_name: Name of the model to use
        max_samples: Maximum number of samples to process (None for all)
        resume: Whether to resume from existing output file
        concurrency: Number of concurrent requests
    """
    # Load the train split of the competition_math dataset
    print("Loading dataset qwedsacf/competition_math (train split)...")
    ds = load_dataset("qwedsacf/competition_math", split="train")
    if max_samples:
        max_samples = min(max_samples, len(ds))
        ds = ds.select(range(max_samples))

    out_dir = Path("data/math-r1-cot")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "sft.jsonl"
    out_file_filter = out_dir / "sft_filter.jsonl"

    template = "{question} Please reason step by step, and put your final answer within \\boxed{{}}."

    # Load already processed samples if resuming
    processed_indices = set()
    if resume and out_file.exists():
        print(f"Resume mode: loading existing data from {out_file}...")
        with open(out_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                try:
                    record = json.loads(line.strip())
                    question = record.get("question", record.get("problem", ""))
                    processed_indices.add(hash(question))
                except Exception as e:
                    print(f"Warning: Failed to parse line {line_num}: {e}")
        print(f"Found {len(processed_indices)} already processed samples. Skipping them...")

    # Prepare samples to process
    samples_to_run = []
    for sample in ds:
        question = sample.get("problem", sample.get("question", ""))
        if hash(question) in processed_indices:
            continue
        samples_to_run.append(sample)

    print(f"Will process {len(samples_to_run)} samples with concurrency={concurrency}")

    # Statistics
    written = 0
    written_filter = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    file_lock = asyncio.Lock()

    async def process_one(sample):
        """Process a single sample"""
        nonlocal written, written_filter, total_prompt_tokens, total_completion_tokens, total_tokens
        
        question = sample.get("problem", sample.get("question", ""))
        ground_truth = sample.get("solution", sample.get("answer", ""))
        prompt = template.format(question=question)

        # Make async request with retry
        for attempt in range(2):
            try:
                resp = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=12288,
                    temperature=0.6,
                    top_p=0.95,
                    extra_body={
                        'repetition_penalty': 1.05,
                    },
                )
                break
            except Exception as e:
                if attempt == 0:
                    print(f"\nRequest failed: {e}, retrying after 5s...")
                    await asyncio.sleep(5)
                    continue
                else:
                    print(f"\nRetry failed: {e}, skipping sample")
                    return

        # Extract generated text
        try:
            gen_text = resp.choices[0].message.content
        except Exception:
            gen_text = str(resp)

        # Extract token usage
        token_usage = {}
        try:
            if hasattr(resp, 'usage') and resp.usage:
                token_usage = {
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens,
                    "total_tokens": resp.usage.total_tokens,
                }
        except Exception:
            pass

        # Calculate metrics
        metrics = r1_zero_reward_fn(gen_text, ground_truth, False)

        # Compose output record
        out_record = dict(sample)
        out_record.update({
            "question": question,
            "answer": ground_truth,
            "generation_solution": gen_text,
            "metrics": metrics,
            "token_usage": token_usage,
        })

        # Write to file with lock
        async with file_lock:
            with open(out_file, "a", encoding="utf-8") as f_all:
                f_all.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                f_all.flush()
            
            if metrics.get("answer_reward", 0.0) == 1.0:
                with open(out_file_filter, "a", encoding="utf-8") as f_filt:
                    f_filt.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                    f_filt.flush()

        # Update statistics
        written += 1
        if token_usage:
            total_prompt_tokens += token_usage.get("prompt_tokens", 0)
            total_completion_tokens += token_usage.get("completion_tokens", 0)
            total_tokens += token_usage.get("total_tokens", 0)
        if metrics.get("answer_reward", 0.0) == 1.0:
            written_filter += 1

    # Use semaphore to limit concurrency
    semaphore = asyncio.Semaphore(concurrency)

    async def sem_task(sample):
        async with semaphore:
            await process_one(sample)

    # Process all samples with progress bar
    tasks = [sem_task(s) for s in samples_to_run]
    await async_tqdm.gather(*tasks, desc="Generating")

    # Print statistics
    skipped = len(processed_indices)
    print(f"\nSkipped {skipped} already processed samples")
    print(f"Wrote {written} new records to {out_file}")
    print(f"Wrote {written_filter} new filtered (answer_reward=1) records to {out_file_filter}")
    print(f"Total records in {out_file}: {len(processed_indices) + written}")
    print(f"\nToken Usage Statistics:")
    print(f"  Total prompt tokens: {total_prompt_tokens:,}")
    print(f"  Total completion tokens: {total_completion_tokens:,}")
    print(f"  Total tokens: {total_tokens:,}")
    if written > 0:
        print(f"  Average tokens per sample: {total_tokens / written:.1f}")


async def main_async(args):
    """Main async function"""

    client = create_async_client()
    
    # Optionally list available models
    print("Available models:")
    models = await get_models_id(client)
    for model in models:
        print(f"  - {model}")

    await generate_for_dataset(
        client, models[0],
        max_samples=args.max_samples,
        resume=not args.no_resume,
        concurrency=args.concurrency
    )


def main():
    parser = argparse.ArgumentParser(description="Generate math solutions using AsyncOpenAI client")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples to process")
    parser.add_argument("--no-resume", action="store_true", help="Start from scratch instead of resuming")
    parser.add_argument("--concurrency", type=int, default=16, help="Number of concurrent requests (default: 16)")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

# Usage examples:
# uv run scripts/generate_math_sft_openai_vllm.py
# uv run scripts/generate_math_sft_openai_vllm.py --concurrency 20
# uv run scripts/generate_math_sft_openai_vllm.py --max-samples 100 --concurrency 10
