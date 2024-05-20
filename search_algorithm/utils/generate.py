from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.generation import TopKLogitsWarper, TopPLogitsWarper

def top_k_top_p_filtering(
    logits: torch.FloatTensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.FloatTensor:
    
    if top_k > 0:
        logits = TopKLogitsWarper(top_k=top_k, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep)(
            None, logits
        )

    if 0 <= top_p <= 1.0:
        logits = TopPLogitsWarper(top_p=top_p, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep)(
            None, logits
        )

    return logits

def generate(
    model: nn.Module, queries: List[torch.LongTensor], txt_len: int = 20, top_k: int = 0, top_p: float = 1.0,
    temperature: int = 0.3
) -> torch.LongTensor:
    """Sample text from language model."""
    input_ids = queries
    for _i in range(txt_len):
        # Get Logits
        outputs = model(input_ids)
        next_token_logits = outputs[0][:, -1, :]
        
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature
            
        next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        # Sample
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
        
    return input_ids[:, -txt_len:]
