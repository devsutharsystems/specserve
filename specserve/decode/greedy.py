"""Batch-one greedy decoding and shared target verification."""

from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DecodeIteration:
    proposed: int
    accepted: int
    mismatch: bool
    stopped: bool
    tokens_emitted: int
    target_passes: int


@dataclass
class DecodeResult:
    token_ids: list[int]
    iterations: list[DecodeIteration]
    stopped_on_eos: bool


def greedy_decode(
    model,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_token_id: int | None,
    on_prefill: Callable[[], None] | None = None,
) -> DecodeResult:
    return _decode(
        model, input_ids, max_new_tokens=max_new_tokens, eos_token_id=eos_token_id,
        on_prefill=on_prefill, propose=None,
    )


@torch.inference_mode()
def _decode(
    model,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_token_id: int | None,
    on_prefill: Callable[[], None] | None,
    propose: Callable[[list[int]], list[int]] | None,
) -> DecodeResult:
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError("Expected one nonempty, unpadded sequence of input IDs")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")

    generated: list[int] = []
    rows: list[DecodeIteration] = []
    if max_new_tokens == 0:
        return DecodeResult(generated, rows, False)

    history = input_ids[0].tolist() if propose is not None else []
    prefill = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
    cache = prefill.past_key_values
    generated.append(prefill.logits[0, -1].argmax().item())
    if on_prefill is not None:
        on_prefill()

    while len(generated) < max_new_tokens and generated[-1] != eos_token_id:
        remaining = max_new_tokens - len(generated)
        proposal = [] if propose is None else propose(history + generated)[:remaining]

        # The last emitted token is pending. Its logits verify the first guess.
        # Each later position predicts the token after that position.
        pending_and_guesses = input_ids.new_tensor([[generated[-1], *proposal]])
        verified = model(input_ids=pending_and_guesses, past_key_values=cache, use_cache=True)
        cache = verified.past_key_values
        predictions = verified.logits[0].argmax(dim=-1).tolist()

        accepted = 0
        mismatch = False
        emitted = []
        for guess, prediction in zip(proposal, predictions):
            if guess != prediction:
                mismatch = True
                emitted.append(prediction)
                break
            accepted += 1
            emitted.append(guess)
            if guess == eos_token_id:
                break

        if not mismatch and accepted == len(proposal) and accepted < remaining:
            if not emitted or emitted[-1] != eos_token_id:
                emitted.append(predictions[accepted])

        # Only the pending token and accepted guesses belong in the cache.
        # A correction or bonus becomes the pending token of the next step.
        discarded = len(proposal) - accepted
        if discarded:
            cache.crop(-discarded)

        generated.extend(emitted)
        rows.append(DecodeIteration(
            proposed=len(proposal), accepted=accepted, mismatch=mismatch,
            stopped=len(generated) == max_new_tokens or generated[-1] == eos_token_id,
            tokens_emitted=len(emitted), target_passes=1,
        ))

    return DecodeResult(generated, rows, generated[-1] == eos_token_id)
