import os
import time
import json
from pathlib import Path
import argparse
from tqdm import tqdm

from datasets import load_dataset
from openai import OpenAI

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


API_URL = os.environ.get("API_URL", "http://ds-r1-0528-671b-64k-ep.api.sii.edu.cn")
MODEL_NAME = os.environ.get("MODEL_NAME", "deepseek-r1-0528-ep")
API_KEY = os.environ.get("OPENAI_API_KEY", os.environ.get("API_KEY", "EMPTY"))


def create_client():
    client = OpenAI(
        base_url=f"{API_URL}/v1",
        api_key=API_KEY,
    )
    return client


def generate_for_dataset(client, model_name, max_samples=None, sleep_between=0.0, resume=True):
    # Load the train split of the competition_math dataset
    print("Loading dataset qwedsacf/competition_math (train split)...")
    ds = load_dataset("qwedsacf/competition_math", split="train")
    if max_samples:
        max_samples = min(max_samples, len(ds))
        ds = ds.select(range(max_samples))

    out_dir = Path("Math")
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
                    # Use the index from the dataset to track processed samples
                    # We'll store a hash of the question to identify duplicates
                    question = record.get("question", record.get("problem", ""))
                    processed_indices.add(hash(question))
                except Exception as e:
                    print(f"Warning: Failed to parse line {line_num}: {e}")
        print(f"Found {len(processed_indices)} already processed samples. Skipping them...")

    written = 0
    written_filter = 0
    skipped = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    # Open files in append mode to allow resuming after interruption
    with open(out_file, "a", encoding="utf-8") as fout_all, open(out_file_filter, "a", encoding="utf-8") as fout_filter:
        for sample in tqdm(ds, desc="Generating"):
            question = sample.get("problem", sample.get("question", ""))
            ground_truth = sample.get("solution", sample.get("answer", ""))

            # Skip if already processed
            question_hash = hash(question)
            if question_hash in processed_indices:
                skipped += 1
                continue

            prompt = template.format(question=question)

            # Build request
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=32768,
                    temperature=0.6,
                    top_p=0.95,
                    extra_body={
                        'repetition_penalty': 1.05,
                    },
                    stream=False,
                )
            except Exception as e:
                print(f"\nRequest failed: {e}, retrying after 5s...")
                time.sleep(5)
                try:
                    resp = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=32768,
                        temperature=0.6,
                        top_p=0.95,
                        extra_body={
                            'repetition_penalty': 1.05,
                        },
                        stream=False,
                    )
                except Exception as e2:
                    print(f"\nRetry failed: {e2}, skipping sample")
                    continue

            # Extract text from response (handle different response shapes)
            try:
                # Newer SDK: choices[0].message.content
                gen_text = resp.choices[0].message.content
            except Exception:
                try:
                    gen_text = resp.choices[0].text
                except Exception:
                    gen_text = str(resp)

            # Extract token usage information
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

            metrics = r1_zero_reward_fn(gen_text, ground_truth, False)

            # Compose output record: include original fields, plus mapped question/answer, generation and metrics
            out_record = dict(sample)
            out_record.update({
                "question": question,
                "answer": ground_truth,
                "generation_solution": gen_text,
                "metrics": metrics,
                "token_usage": token_usage,
            })

            # Write immediately and flush to disk
            fout_all.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            fout_all.flush()  # Force write to disk
            written += 1

            # Accumulate token statistics
            if token_usage:
                total_prompt_tokens += token_usage.get("prompt_tokens", 0)
                total_completion_tokens += token_usage.get("completion_tokens", 0)
                total_tokens += token_usage.get("total_tokens", 0)

            if metrics.get("answer_reward", 0.0) == 1.0:
                fout_filter.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                fout_filter.flush()  # Force write to disk
                written_filter += 1

            if sleep_between > 0:
                time.sleep(sleep_between)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples to process")
    parser.add_argument("--api-url", type=str, default=None, help="API base URL (overrides env)")
    parser.add_argument("--model", type=str, default=None, help="Model name (overrides env)")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between requests")
    parser.add_argument("--no-resume", action="store_true", help="Start from scratch instead of resuming")
    args = parser.parse_args()

    global API_URL, MODEL_NAME
    if args.api_url:
        API_URL = args.api_url
    if args.model:
        MODEL_NAME = args.model

    client = create_client()
    generate_for_dataset(
        client, 
        MODEL_NAME, 
        max_samples=args.max_samples, 
        sleep_between=args.sleep,
        resume=not args.no_resume
    )


if __name__ == "__main__":
    main()

# uv run scripts/generate_competition_math_sft.py