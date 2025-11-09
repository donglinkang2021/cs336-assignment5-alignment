
import torch
from transformers import PreTrainedTokenizerBase

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

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
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
