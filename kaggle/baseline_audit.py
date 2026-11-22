"""Ordered exact-match gates and untimed fp16 divergence replay."""

import math
from collections import defaultdict

import torch

from kaggle.baseline_artifacts import digest, json_bytes, stop_reason
from specserve.decode.prompt_lookup import find_proposal


def check_row(stage, method, prompt_id, *, status, repeat=None, **details):
    return {
        "stage": stage, "method": method, "prompt_id": prompt_id,
        "repeat": repeat, "status": status, "details": details,
    }


def first_difference(reference, output):
    for position, (expected, actual) in enumerate(zip(reference, output)):
        if expected != actual:
            return position
    if len(reference) != len(output):
        return min(len(reference), len(output))
    return None


def _token_at(tokens, position):
    return tokens[position] if position < len(tokens) else None


def fp32_gate(stages, hf, decode, audit, persist):
    for stage, prompts in stages:
        references = {}
        for method in ("greedy", "prompt_lookup"):
            for prompt in prompts:
                result = decode(method, prompt)
                if method == "greedy":
                    expected = hf(prompt)
                    references[prompt.id] = result.token_ids
                else:
                    expected = references[prompt.id]
                actual = result.token_ids
                position = first_difference(expected, actual)
                details = {"reference_length": len(expected), "output_length": len(actual)}
                if position is not None:
                    details.update(
                        position=position, reference_token_id=_token_at(expected, position),
                        output_token_id=_token_at(actual, position),
                    )
                audit.append(check_row(
                    stage, method, prompt.id, status="pass" if position is None else "fail",
                    **details,
                ))
                if position is not None:
                    persist()
                    raise RuntimeError(f"{stage}: {method} exact-match failed on {prompt.id}")
        persist()


def _top_two(logits):
    if logits.numel() < 2 or not torch.isfinite(logits).all().item():
        return None
    # Choose the first ID the same way the decoder does, even for exact ties.
    top_id = logits.argmax().item()
    other = logits.clone()
    other[top_id] = -float("inf")
    second_id = other.argmax().item()
    values = [float(logits[top_id].item()), float(logits[second_id].item())]
    return {"ids": [top_id, second_id], "logits": values, "gap": values[0] - values[1]}


def _locate_forward(result, position):
    if position == 0:
        return 0, 0
    emitted = 1  # prefill emitted token at index zero
    for call, row in enumerate(result.iterations, start=1):
        if position < emitted + row.tokens_emitted:
            return call, position - emitted
        emitted += row.tokens_emitted
    # An unexpectedly short output may have a next logit in its last pass.
    if result.iterations:
        return len(result.iterations), result.iterations[-1].tokens_emitted
    return None


class ReplayTarget:
    """Observe a single prediction while replaying the complete original path."""

    def __init__(self, target, result, position, prompt_ids, compared_ids):
        self.target = target
        self.location = _locate_forward(result, position)
        self.prompt_ids = prompt_ids
        self.position = position
        self.shared_prefix = result.token_ids[:position]
        self.compared_ids = compared_ids
        self.call = 0
        self.query_lengths = []
        self.seen_tokens = []
        self.evidence = None
        self.last_cache = None

    def __call__(self, *, input_ids, past_key_values=None, **kwargs):
        cache_length = 0 if past_key_values is None else int(past_key_values.get_seq_length())
        cached_tokens = self.seen_tokens[:cache_length]
        forward_tokens = input_ids[0].tolist()
        output = self.target(input_ids=input_ids, past_key_values=past_key_values, **kwargs)
        self.last_cache = output.past_key_values
        self.seen_tokens = cached_tokens + forward_tokens
        self.query_lengths.append(len(forward_tokens))
        if self.location is not None and self.call == self.location[0]:
            row = self.location[1]
            if row < output.logits.shape[1]:
                logits = output.logits[0, row]
                top_two = _top_two(logits)
                # Prefill returns one logit row for the entire prompt.
                prefix_length = cache_length + len(forward_tokens) - output.logits.shape[1] + row + 1
                self.evidence = {
                    "forward_index": self.call, "forward_shape": list(input_ids.shape),
                    "query_lengths_through_forward": self.query_lengths.copy(),
                    "cache_length_before_forward": cache_length, "logit_row": row,
                    "prefix_matches": (
                        prefix_length == len(self.prompt_ids) + self.position
                        and self.seen_tokens[:prefix_length] == self.prompt_ids + self.shared_prefix
                    ),
                    "top_two": top_two,
                    "compared_logits": (
                        {str(token): float(logits[token].item()) for token in self.compared_ids
                         if token is not None} if top_two is not None else None
                    ),
                    "source": "original_shape_replay",
                }
        self.call += 1
        return output


def replay(target, decode, method, prompt, result, position, eos, settings, compared_ids):
    observer = ReplayTarget(target, result, position, prompt.token_ids, compared_ids)
    replayed = decode(method, prompt, target=observer)
    if replayed != result:
        return {"reproduced": False, "prediction": None}
    evidence = observer.evidence
    if evidence is None and result.token_ids and position == len(result.token_ids):
        # Probe an invalid early stop using the actual replay cache after its
        # final crop. This extra pass is diagnostic, never an original prediction.
        if stop_reason(result.token_ids, eos, prompt.budget) == "unexpected_stop":
            cache = observer.last_cache
            history = prompt.token_ids + result.token_ids
            if cache is not None and cache.get_seq_length() == len(history) - 1:
                proposal = []
                if method == "prompt_lookup":
                    proposal = find_proposal(
                        history, max_matching_ngram_size=settings["max_matching_ngram_size"],
                        prompt_lookup_num_tokens=settings["prompt_lookup_num_tokens"],
                    )[:prompt.budget - position]
                ids = torch.tensor([[history[-1], *proposal]], device=prompt.tensor.device)
                observer.location = (observer.call, 0)
                with torch.inference_mode():
                    observer(input_ids=ids, past_key_values=cache, use_cache=True)
                evidence = observer.evidence
                if evidence:
                    evidence["source"] = "early_stop_continuation_probe"
    return {"reproduced": True, "prediction": evidence}


def _stop(result, prompt, eos):
    return {
        "length": len(result.token_ids), "eos_token_id": eos, "budget": prompt.budget,
        "stopped_on_eos": result.stopped_on_eos,
        "reason": stop_reason(result.token_ids, eos, prompt.budget),
        "last_token_id": result.token_ids[-1] if result.token_ids else None,
        "last_iteration_stopped": result.iterations[-1].stopped if result.iterations else None,
    }


def _near_tie_candidate(reference, output, reference_id, output_id):
    if not reference or not output or reference_id is None or output_id is None:
        return False
    if any(
        path["source"] != "original_shape_replay" or not path["prefix_matches"]
        or path["top_two"] is None
        for path in (reference, output)
    ):
        return False
    return (
        reference["query_lengths_through_forward"] != output["query_lengths_through_forward"]
        and reference["top_two"]["ids"] == [reference_id, output_id]
        and output["top_two"]["ids"] == [output_id, reference_id]
        and all(math.isfinite(path["top_two"]["gap"]) for path in (reference, output))
    )


def fp16_comparison(stage, method, prompt, reference, result, *, target, decode, eos,
                    settings, reviews, identity, repeat=None):
    position = first_difference(reference.token_ids, result.token_ids)
    if position is None:
        reference_stop = _stop(reference, prompt, eos)
        output_stop = _stop(result, prompt, eos)
        if (
            reference_stop["reason"] != "unexpected_stop"
            and reference_stop["stopped_on_eos"] == output_stop["stopped_on_eos"]
            == (output_stop["reason"] == "eos")
        ):
            return check_row(stage, method, prompt.id, repeat=repeat, status="pass",
                             output_length=len(result.token_ids))
        return check_row(
            stage, method, prompt.id, repeat=repeat, status="fail", explained=False,
            reference_stop=reference_stop, output_stop=output_stop,
            diagnosis="Matching IDs have inconsistent or invalid stopping decisions.",
        )
    ref_id = _token_at(reference.token_ids, position)
    out_id = _token_at(result.token_ids, position)
    compared_ids = (ref_id, out_id)
    reference_replay = replay(
        target, decode, "greedy", prompt, reference, position, eos, settings, compared_ids,
    )
    output_replay = replay(target, decode, method, prompt, result, position, eos, settings, compared_ids)
    evidence = {
        "position": position, "reference_token_id": ref_id, "output_token_id": out_id,
        "reference_stop": _stop(reference, prompt, eos), "output_stop": _stop(result, prompt, eos),
        "reference_replay": reference_replay, "output_replay": output_replay,
        "outputs_sha256": digest(json_bytes([reference.token_ids, result.token_ids])),
    }
    case_id = digest(json_bytes([identity, prompt.id, method, evidence]))
    candidate = _near_tie_candidate(
        reference_replay["prediction"], output_replay["prediction"], ref_id, out_id,
    )
    review = reviews.get(case_id)
    reviewed = bool(candidate and review and review["verdict"] == "near_tie")
    if not reference_replay["reproduced"] or not output_replay["reproduced"]:
        diagnosis = "A replay did not reproduce the original output and iteration trace."
    elif ref_id is None or out_id is None:
        diagnosis = "Length-only divergence at a shared prefix. Inspect the stopping decisions."
    elif candidate and review:
        diagnosis = review["diagnosis"]
    elif candidate:
        diagnosis = "The original forward shapes reproduced the flip. Review both logit gaps."
    else:
        diagnosis = "Replay did not establish a flip between the same two tokens at different shapes."
    return check_row(
        stage, method, prompt.id, repeat=repeat, status="reviewed" if reviewed else "fail",
        **evidence, case_id=case_id, near_tie_reviewable=candidate, diagnosis=diagnosis, review=review,
    )


def first_five_per_category(prompts):
    counts = defaultdict(int)
    subset = []
    for prompt in prompts:
        counts[prompt.category] += 1
        if counts[prompt.category] <= 5:
            subset.append(prompt)
    return subset
