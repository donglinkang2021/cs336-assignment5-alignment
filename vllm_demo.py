from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm_sync import VLLMClient

def generate_text(client:VLLMClient, tokenizer: AutoTokenizer):
    """Generate text using the VLLM client."""
    # Generate completions
    generation_kwargs = dict(
        temperature=1.0,
        top_p=1.0,
        max_tokens=32,
        # stop=["</answer>"],
        # include_stop_str_in_output=True,
    )
    output = client.generate(
        ["Hello, AI!", "Tell me a joke"], 
        n=4, generation_kwargs = generation_kwargs
    )
    # print("Responses:", responses)  # noqa
    print("====================== Generated Text ====================")
    for completion_ids in output["completion_ids"]:
        # completion_ids is list[int] of token IDs
        text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        print("Generated text:", text)

def main():
    client = VLLMClient()
    client.init_communicator()

    # Update model weights
    model0 = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B", dtype="bfloat16"
    ).to("cuda:0")
    tokenizer0 = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
    model1 = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Math-1.5B", dtype="bfloat16"
    ).to("cuda:0")
    tokenizer1 = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
    model2 = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Math-1.5B-Instruct", dtype="bfloat16"
    ).to("cuda:0")
    tokenizer2 = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct")

    client.update_model_params(model0)
    generate_text(client, tokenizer0)

    client.update_model_params(model1)
    generate_text(client, tokenizer1)
    
    client.update_model_params(model2)
    generate_text(client, tokenizer2)

if __name__ == "__main__":
    main()

# HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python vllm_demo.py