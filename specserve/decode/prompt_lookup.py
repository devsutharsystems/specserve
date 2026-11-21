"""Draft by copying a continuation of the current suffix from token history."""

from collections.abc import Callable

import torch

from .greedy import DecodeResult, _decode


def find_proposal(
    history: list[int], *, max_matching_ngram_size: int, prompt_lookup_num_tokens: int
) -> list[int]:
    """Use the matching rule pinned in the Day 2b plan."""
    if max_matching_ngram_size <= 0 or prompt_lookup_num_tokens <= 0:
        raise ValueError("Prompt-lookup settings must be positive")

    for size in range(min(max_matching_ngram_size, len(history)), 0, -1):
        suffix_start = len(history) - size
        suffix = history[suffix_start:]
        latest_copy_start = min(suffix_start - 1, len(history) - prompt_lookup_num_tokens)
        for start in range(latest_copy_start - size + 1):
            copy_start = start + size
            if history[start:copy_start] == suffix:
                return history[copy_start:copy_start + prompt_lookup_num_tokens]
    return []


def prompt_lookup_decode(
    model,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_token_id: int | None,
    max_matching_ngram_size: int,
    prompt_lookup_num_tokens: int,
    on_prefill: Callable[[], None] | None = None,
) -> DecodeResult:
    if max_matching_ngram_size <= 0 or prompt_lookup_num_tokens <= 0:
        raise ValueError("Prompt-lookup settings must be positive")

    def propose(history: list[int]) -> list[int]:
        return find_proposal(
            history,
            max_matching_ngram_size=max_matching_ngram_size,
            prompt_lookup_num_tokens=prompt_lookup_num_tokens,
        )

    return _decode(
        model, input_ids, max_new_tokens=max_new_tokens, eos_token_id=eos_token_id,
        on_prefill=on_prefill, propose=propose,
    )
