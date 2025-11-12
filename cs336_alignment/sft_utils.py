import re
import torch
from transformers import PreTrainedTokenizerBase, PreTrainedModel, GenerationConfig
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

# uv run pytest -k test_tokenize_prompt_and_output
def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
        "input_ids": torch.Tensor of shape (B, L):
            the tokenized prompt and output strings, with the final token sliced off.
        "labels": torch.Tensor of shape (B, L):
            shifted input_ids (i.e., the input_ids without the first token).
        "response_mask": torch.Tensor of shape (B, L):
            a mask on the response tokens in `labels`.

    Note:
        - B=batch_size, L=max(prompt_and_output_lens) - 1
    """
    batch_size = len(prompt_strs)
    assert len(output_strs) == batch_size, "prompt_strs and output_strs must have the same length"
    
    # Tokenize prompts and outputs separately
    prompt_tokenized = tokenizer(
        prompt_strs,
        add_special_tokens=True,
        padding=False,
        truncation=False,
    )
    output_tokenized = tokenizer(
        output_strs,
        add_special_tokens=False,  # Don't add special tokens to output
        padding=False,
        truncation=False,
    )
    
    # Concatenate prompt and output token IDs
    prompt_and_output_ids = []
    prompt_lens = []
    output_lens = []
    
    for i in range(batch_size):
        prompt_ids = prompt_tokenized["input_ids"][i]
        output_ids = output_tokenized["input_ids"][i]
        
        # Concatenate prompt and output
        combined_ids = prompt_ids + output_ids
        prompt_and_output_ids.append(combined_ids)
        
        prompt_lens.append(len(prompt_ids))
        output_lens.append(len(output_ids))
    
    # Find max length
    max_len = max(len(ids) for ids in prompt_and_output_ids)
    
    # Pad sequences and create masks
    input_ids_list = []
    labels_list = []
    response_mask_list = []
    
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    for i in range(batch_size):
        seq = prompt_and_output_ids[i]
        seq_len = len(seq)
        prompt_len = prompt_lens[i]
        
        # Pad the sequence to max_len
        padded_seq = seq + [pad_token_id] * (max_len - seq_len)
        
        # input_ids: all tokens except the last one
        input_ids_list.append(padded_seq[:-1])
        # labels: all tokens except the first one (shifted by 1)
        labels_list.append(padded_seq[1:])
        # response_mask: 1 for response tokens, 0 for prompt and padding
        # Response starts at position prompt_len (in the original sequence)
        # After slicing (removing first token), response starts at prompt_len - 1
        response_mask_list.append(
            [0] * (prompt_len - 1) 
            + [1] * (seq_len - prompt_len) 
            + [0] * (max_len - seq_len)
        )
    
    # Convert to tensors
    return {
        "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
        "labels": torch.tensor(labels_list, dtype=torch.long),
        "response_mask": torch.tensor(response_mask_list, dtype=torch.long),
    }

# uv run pytest -k test_compute_entropy
def compute_entropy_naive(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy over the vocabulary dimension.

    Args:
        logits: torch.Tensor of shape `(batch_size, sequence_length, vocab_size)`
            containing unnormalized logits.

    Returns:
        entropy: torch.Tensor of shape `(batch_size, sequence_length)`. 
            The entropy for each next-token prediction.
    """
    probs = logits.softmax(dim=-1) # (B, T, V)
    entropy = logits.logsumexp(dim=-1) - (probs * logits).sum(dim=-1) # (B, T)
    return entropy

def compute_entropy(logits:torch.Tensor, chunk_size:int=128) -> torch.Tensor:
    """Memory-efficient implementation of `compute_entropy`."""
    num_chunks = (logits.shape[1] + chunk_size - 1) // chunk_size
    entropy_chunks = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, logits.shape[1])
        chunk_logits = logits[:, start_idx:end_idx, :]
        # Use the numerically stable method for torch.bfloat16, do not use logsumexp
        chunk_probs = chunk_logits.softmax(dim=-1)
        chunk_log_probs = chunk_logits.log_softmax(dim=-1)
        chunk_entropy = -(chunk_probs * chunk_log_probs).sum(dim=-1)
        entropy_chunks.append(chunk_entropy)
    return torch.cat(entropy_chunks, dim=1)

def selective_log_softmax_naive(logits:torch.Tensor, index:torch.Tensor) -> torch.Tensor:
    """Naive implementation of the common `selective_log_softmax` operation"""
    return logits.log_softmax(dim=-1).gather(
        dim=-1, index=index.unsqueeze(-1)
    ).squeeze(-1)

def selective_log_softmax(logits:torch.Tensor, index:torch.Tensor, chunk_size:int=128) -> torch.Tensor:
    """Memory-efficient implementation of the common `selective_log_softmax` operation"""
    num_chunks = (logits.shape[1] + chunk_size - 1) // chunk_size
    log_probs_chunks = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, logits.shape[1])
        chunk_logits = logits[:, start_idx:end_idx, :]
        chunk_log_probs = chunk_logits.log_softmax(dim=-1)
        chunk_log_probs = chunk_log_probs.gather(dim=-1, index=index[:, start_idx:end_idx].unsqueeze(-1)).squeeze(-1)
        log_probs_chunks.append(chunk_log_probs)
    return torch.cat(log_probs_chunks, dim=1)

# uv run pytest -k test_get_response_log_probs
def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """Get the conditional log-probs of the response given the prompt, and optionally the entropy of the next token predictions.

    Args:
        model: PreTrainedModel, HuggingFace model used for scoring, 
            placed on the correct device and in inference mode if gradients should not be computed.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            concatenated prompt + response tokens as produced by your tokenization method.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            labels as produced by your tokenization method.
        return_token_entropy: bool, whether to return the entropy of the
            next token predictions.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length):
                the conditional log-probs of the response given the prompt.
                Note that we have not masked out the token indices corresponding
                to the prompt or padding; that is done in the train loop.
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length):
                the entropy of the next token predictions. As with the log-probs,
                we have not masked out the token indices corresponding to the prompt
                or padding; that is done in the train loop.
    """
    logits:torch.Tensor = model(input_ids).logits  # (B, T, V)
    log_probs = selective_log_softmax(logits, labels)
    output = {"log_probs": log_probs}
    if return_token_entropy:
        output["token_entropy"] = compute_entropy(logits)  # (B, T)
    return output

# uv run pytest -k test_masked_normalize
def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int = -1,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to sum and normalize.
        mask: torch.Tensor, the mask. We only consider elements
            with mask value 1.
        dim: int | None, the dimension to sum along before
            normalization. If None, sum over all dimensions.
        normalize_constant: float, the constant to divide by
            for normalization.

    Returns:
        torch.Tensor, the normalized sum, where masked elements
            (mask=0) don't contribute to the sum.
    """
    return (tensor * mask).sum(dim=dim) / normalize_constant

# uv run pytest -k test_sft_microbatch_train_step
def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.
    
    Args:
        policy_log_probs: A tensor of shape `(batch_size, sequence_length)` containing
            the per-token log-probabilities from the SFT policy being trained.
        response_mask: A tensor of shape `(batch_size, sequence_length)` with `1` for
            response tokens and `0` for prompt/padding tokens. This mask is used to
            select the tokens for which the loss should be computed.
        gradient_accumulation_steps: The number of microbatches to process before
            performing an optimizer step. The loss is divided by this number to
            ensure that the accumulated gradients are correctly scaled.
        normalize_constant: An optional constant to divide the sum of log-probabilities by.
            Defaults to 1.0, which means no extra normalization.
    Returns:
        A tuple containing:
            - loss: A scalar tensor representing the microbatch loss, scaled for
              gradient accumulation. This value is suitable for logging.
            - metadata: A dictionary containing metadata and statistics from the loss computation, which can also be used for logging.
    """
    loss = - masked_normalize(policy_log_probs, response_mask, normalize_constant=normalize_constant).mean() / gradient_accumulation_steps
    loss.backward()
    return loss.detach(), {}

def get_completion_mask(completion_ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    """Get a mask for the completion tokens up to and including the first EOS token."""
    device = completion_ids.device
    is_eos = completion_ids == eos_token_id  # (batch_size, completion_length)
    # Find first EOS position for each sequence, default to sequence length if no EOS
    eos_positions = is_eos.int().argmax(dim=1)  # (batch_size,)
    # If no EOS token exists, argmax returns 0, so we need to check
    has_eos = is_eos.any(dim=1)  # (batch_size,)
    eos_positions = torch.where(has_eos, eos_positions, completion_ids.size(1))
    # Create mask: 1 for tokens up to and including first EOS, 0 after
    completion_mask = torch.arange(completion_ids.size(1), device=device).unsqueeze(0) <= eos_positions.unsqueeze(1)
    return completion_mask.int()

def get_per_token_logits(
    model: PreTrainedModel,
    prompt_completion_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    logits_to_keep: int,
    return_token_entropy: bool = True,
) -> torch.Tensor:
    # prompt_completion_ids: (B, prompt_length + completion_length)
    logits = model(
        prompt_completion_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1
    ).logits  # (B, L, V), L=completion_length + 1,
    logits = logits[:, :-1, :] # (B, completion_length, V)
    output = {"logits": logits}
    if return_token_entropy:
        output["token_entropy"] = compute_entropy(logits)  # (B, T)
    return output

# log_generations
def log_generations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    ground_truths: list[str],
    reward_fn: callable,
    generation_kwargs: dict | None = None,
    max_prompts: int | None = None,
) -> dict[str, any]:
    """Log generations from the model on given prompts.
    
    Args:
        model: PreTrainedModel, the model to generate from.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.
        prompts: list[str], the prompts to generate from.
        ground_truths: list[str], the ground truth answers.
        reward_fn: callable, a function that takes (response, ground_truth) and
            returns a dict with keys "reward", "format_reward", "answer_reward".
    
    Returns:
        dict[str, any]: A dictionary containing logging information with keys:
            - "examples": list of dicts, each containing:
                - "prompt": str
                - "response": str
                - "ground_truth": str
                - "reward": float
                - "format_reward": float
                - "answer_reward": float
                - "avg_token_entropy": float
                - "response_length": int
            - "avg_response_length/total": float
            - "avg_response_length/correct": float
            - "avg_response_length/incorrect": float
            - "avg_token_entropy": float
            - "avg_reward": float
            - "avg_format_reward": float
            - "avg_answer_reward": float
    """
    device = model.device
    if generation_kwargs is None:
        generation_kwargs = {
            'max_new_tokens': 1024,
            'do_sample': True,
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            'temperature': 1.0,
            'top_p': 1.0,
        }
    generation_config = GenerationConfig(**generation_kwargs)
    
    # Limit number of prompts if specified
    if max_prompts is not None and max_prompts < len(prompts):
        prompts = prompts[:max_prompts]
        ground_truths = ground_truths[:max_prompts]
    
    # Set model to eval mode
    model.eval()
    
    # Store results
    examples = []
    
    with torch.no_grad():

        # 1. prompts -> tokenizer -> prompt_ids, prompt_mask
        # Tokenize all prompts in batch
        prompt_inputs = tokenizer(
            prompts, return_tensors="pt",
            padding=True, padding_side="left",
            add_special_tokens=True,
        )
        prompt_ids = prompt_inputs["input_ids"].to(device)
        prompt_mask = prompt_inputs["attention_mask"].to(device)
        
        # 2. prompt_ids, prompt_mask -> model.generate -> prompt_completion_ids
        # Generate responses in batch
        prompt_completion_ids = model.generate(
            prompt_ids, attention_mask=prompt_mask,
            generation_config = generation_config,
        )
        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # 3. completion_ids -> completion_mask
        completion_mask = get_completion_mask(
            completion_ids, tokenizer.eos_token_id
        )  # (B, completion_length)

        # 4. prompt_ids -> prompt_text, completion_ids -> completion_text
        prompts_text = tokenizer.batch_decode(
            prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        prompts_text = [
            re.sub(rf"^({re.escape(tokenizer.pad_token)})+", "", text) for text in prompts_text
        ]
        completions_text = tokenizer.batch_decode(
            completion_ids, skip_special_tokens=True
        )

        # 5. prompt_completion_ids, attention_mask -> model -> logits -> entropy
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        output = get_per_token_logits(model, prompt_completion_ids, attention_mask, logits_to_keep, return_token_entropy=True)
        entropy = output["token_entropy"]  # (B, completion_length)

        # 6. get rewards and log everything
        for i, (prompt, completion, ground_truth) in enumerate(zip(prompts_text, completions_text, ground_truths)):
            # Apply completion mask to get valid tokens only
            valid_length = completion_mask[i].sum().item()
            response_entropy = entropy[i, :valid_length]
            avg_entropy = response_entropy.mean().item()

            # Compute reward
            reward_info = reward_fn(completion, ground_truth)

            # Store example info
            example_info = {
                "prompt": prompt,
                "response": completion,
                "ground_truth": ground_truth,
                "reward": reward_info["reward"],
                "format_reward": reward_info["format_reward"],
                "answer_reward": reward_info["answer_reward"],
                "avg_token_entropy": avg_entropy,
                "response_length": valid_length,
            }
            examples.append(example_info)

    def _mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    return {
        "examples": examples,
        "avg_token_entropy": _mean([ex["avg_token_entropy"] for ex in examples]),
        "avg_reward": _mean([ex["reward"] for ex in examples]),
        "avg_format_reward": _mean([ex["format_reward"] for ex in examples]),
        "avg_answer_reward": _mean([ex["answer_reward"] for ex in examples]),
        "avg_response_length/total": _mean([ex["response_length"] for ex in examples]),
        "avg_response_length/correct": _mean(
            [ex["response_length"] for ex in examples if ex["answer_reward"] > 0]),
        "avg_response_length/incorrect": _mean(
            [ex["response_length"] for ex in examples if ex["answer_reward"] <= 0]),
    }
