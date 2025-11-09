"""Simple test script for log_generations function."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from cs336_alignment.sft_utils import log_generations

# Simple mock reward function for testing
def mock_reward_fn(response, ground_truth):
    """Simple reward function that checks if ground truth is in response."""
    is_correct = ground_truth.lower() in response.lower()
    return {
        "reward": 1.0 if is_correct else 0.0,
        "format_reward": 1.0,  # Always give format reward for testing
        "answer_reward": 1.0 if is_correct else 0.0,
    }

def main():
    print("Loading model and tokenizer...")
    # Use a small model for testing
    model_name = "models/Qwen2.5-Math-1.5B"  # Small model for quick testing
    # model_name = "models/Llama-3.1-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Move model to appropriate device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    print(f"Model loaded on {device}")
    
    # Create some test prompts
    prompts = [
        "What is 2 + 2? Answer: ",
        "The capital of France is ",
        "Complete this: Hello, ",
    ]
    
    ground_truths = [
        "4",
        "Paris",
        "world",
    ]
    
    print("\nGenerating and logging responses...")
    
    # Test the log_generations function
    results = log_generations(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        ground_truths=ground_truths,
        reward_fn=mock_reward_fn,
        generation_kwargs={
            'max_new_tokens': 200,
            'do_sample': False,  # Use greedy decoding for reproducibility
            'pad_token_id': tokenizer.pad_token_id,
        },
        max_prompts=3,
    )
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    # Print aggregate statistics
    print("\nAggregate Statistics:")
    print(f"  Average Response Length: {results['avg_response_length/total']:.2f}")
    print(f"  Average Response Length (Correct): {results['avg_response_length/correct']:.2f}")
    print(f"  Average Response Length (Incorrect): {results['avg_response_length/incorrect']:.2f}")
    print(f"  Average Token Entropy: {results['avg_token_entropy']:.4f}")
    print(f"  Average Reward: {results['avg_reward']:.2f}")
    print(f"  Average Format Reward: {results['avg_format_reward']:.2f}")
    print(f"  Average Answer Reward: {results['avg_answer_reward']:.2f}")
    
    # Print individual examples
    print("\nIndividual Examples:")
    for i, example in enumerate(results['examples'], 1):
        print(f"\n--- Example {i} ---")
        print(f"Prompt: {example['prompt']}")
        print(f"Response: {example['response']}")
        print(f"Ground Truth: {example['ground_truth']}")
        print(f"Reward: {example['reward']}")
        print(f"Format Reward: {example['format_reward']}")
        print(f"Answer Reward: {example['answer_reward']}")
        print(f"Avg Token Entropy: {example['avg_token_entropy']:.4f}")
        print(f"Response Length: {example['response_length']}")
    
    print("\n" + "="*80)
    print("Test completed successfully!")
    print("="*80)

if __name__ == "__main__":
    main()
