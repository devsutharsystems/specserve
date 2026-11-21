import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache

from specserve.decode.greedy import DecodeIteration, greedy_decode
from specserve.decode.prompt_lookup import find_proposal, prompt_lookup_decode
from specserve.decode.prompting import TARGET_MODEL_ID, TARGET_REVISION, load_tokenizer, prompt_token_ids

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "bench/prompts.jsonl"
SEED = 20261118


def frozen_prompts():
    return [json.loads(line) for line in PROMPTS.read_text().splitlines()]


def test_frozen_sha256():
    assert hashlib.sha256(PROMPTS.read_bytes()).hexdigest() == (
        "f8d753325a1cc4e52d809e311e47f8182474c6c1504f4c29cd977d1f91289c36"
    )


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer()


def test_pinned_chat_template(tokenizer):
    assert tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 2 + 2?"}],
        tokenize=False,
        add_generation_prompt=True,
    ) == "<|user|>\nWhat is 2 + 2?</s>\n<|assistant|>\n"
    assert prompt_token_ids(tokenizer, "What is 2 + 2?")[:5] == [1, 529, 29989, 1792, 29989]


def test_all_frozen_token_counts(tokenizer):
    records = frozen_prompts()
    assert len(records) == 100
    for record in records:
        ids = prompt_token_ids(tokenizer, record["prompt"])
        assert len(ids) == record["prompt_tokens"], record["id"]
        assert len(ids) <= 896 and len(ids) + record["max_new_tokens"] <= 1024


@pytest.fixture
def tiny_llama():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        config = LlamaConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            pad_token_id=0, bos_token_id=1, eos_token_id=None,
        )
        return LlamaForCausalLM(config).float().eval()


@pytest.mark.parametrize("stop_at_eos", [True, False], ids=["eos", "budget"])
def test_greedy_matches_hf(tiny_llama, stop_at_eos):
    ids = torch.tensor([[1, 5, 7, 9]])
    budget = 6

    def hf_tokens(eos):
        return tiny_llama.generate(
            ids, do_sample=False, max_new_tokens=budget, eos_token_id=eos,
        )[0, ids.shape[1]:].tolist()

    trace = hf_tokens(None)
    assert len(trace) == budget
    eos = trace[0] if stop_at_eos else next(
        token for token in range(tiny_llama.config.vocab_size) if token not in trace
    )
    result = greedy_decode(tiny_llama, ids, max_new_tokens=budget, eos_token_id=eos)
    assert result.token_ids == hf_tokens(eos) == (trace[:1] if stop_at_eos else trace)
    assert result.stopped_on_eos is stop_at_eos
    if stop_at_eos:
        assert result.iterations == []
    else:
        assert result.iterations == [
            DecodeIteration(0, 0, False, step == budget - 1, 1, 1)
            for step in range(1, budget)
        ]


@pytest.mark.parametrize(("history", "expected"), [
    ([1, 2, 3, 4, 1, 2], [3, 4]),
    ([1, 2, 3, 4], []),
    ([9, 8, 1, 2, 3, 1], [2, 3]),
    ([1, 2, 3, 4, 1, 2, 5, 6, 1, 2], [3, 4]),
    ([2, 9, 9, 1, 2, 3, 4, 1, 2], [3, 4]),
    ([1, 1], []),
    ([1, 2, 1, 2, 1], [2, 1]),
])
def test_matcher(history, expected):
    assert find_proposal(
        history, max_matching_ngram_size=3, prompt_lookup_num_tokens=2
    ) == expected


@torch.inference_mode()
def test_prompt_lookup_real_hf_cache(tiny_llama):
    ids = torch.tensor([[8] * 16])
    calls = []
    positions = []

    def record_positions(_module, _args, kwargs):
        positions.append(kwargs["position_ids"].flatten().tolist())

    def traced_target(*, input_ids, past_key_values=None, **kwargs):
        retained = []
        if past_key_values is not None:
            assert isinstance(past_key_values, DynamicCache)
            retained = [(layer.keys.clone(), layer.values.clone()) for layer in past_key_values.layers]
        output = tiny_llama(input_ids=input_ids, past_key_values=past_key_values, **kwargs)
        assert isinstance(output.past_key_values, DynamicCache)
        calls.append((input_ids[0].tolist(), retained, output.logits[0, 0].argmax().item()))
        return output

    with tiny_llama.model.rotary_emb.register_forward_pre_hook(record_positions, with_kwargs=True):
        lookup = prompt_lookup_decode(
            traced_target, ids, max_new_tokens=6, eos_token_id=None,
            max_matching_ngram_size=3, prompt_lookup_num_tokens=8,
        )
    greedy = greedy_decode(tiny_llama, ids, max_new_tokens=6, eos_token_id=None)
    assert lookup.token_ids == greedy.token_ids
    rejection = next(index for index, row in enumerate(lookup.iterations) if row.mismatch)
    row = lookup.iterations[rejection]
    assert row.proposed > row.accepted and not row.stopped

    retained_len = ids.shape[1] + sum(item.tokens_emitted for item in lookup.iterations[:rejection])
    retained_len += 1 + row.accepted
    verify_call = rejection + 1  # Call zero is prefill.
    next_call = verify_call + 1
    next_inputs, retained, next_token = calls[next_call]
    correction = lookup.token_ids[retained_len - ids.shape[1]]
    assert next_inputs[0] == correction
    verified_positions = positions[verify_call][:1 + row.accepted]
    assert positions[next_call] == list(range(retained_len, retained_len + len(next_inputs)))

    prefix = ids[0].tolist() + greedy.token_ids[:retained_len - ids.shape[1]]
    with tiny_llama.model.rotary_emb.register_forward_pre_hook(record_positions, with_kwargs=True):
        fresh = tiny_llama(input_ids=torch.tensor([prefix]), use_cache=True)
    assert verified_positions == positions[-1][-(1 + row.accepted):]
    assert fresh.past_key_values.get_seq_length() == retained_len
    assert len(retained) == len(fresh.past_key_values.layers)
    for (keys, values), layer in zip(retained, fresh.past_key_values.layers):
        assert keys.shape[-2] == values.shape[-2] == retained_len
        torch.testing.assert_close(keys, layer.keys, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(values, layer.values, atol=1e-6, rtol=1e-5)

    next_position = positions[next_call][0]
    positions.clear()
    with tiny_llama.model.rotary_emb.register_forward_pre_hook(record_positions, with_kwargs=True):
        fresh_next = tiny_llama(input_ids=torch.tensor([prefix + [correction]]), use_cache=False)
    assert positions[0][-1] == next_position == retained_len
    assert next_token == fresh_next.logits[0, -1].argmax().item()


@pytest.mark.weights
@torch.inference_mode()
def test_tinyllama_fp32_exact_match():
    if not torch.cuda.is_available():
        pytest.skip("real TinyLlama weights are reserved for the Kaggle GPU")
    settings = yaml.safe_load((ROOT / "configs/bench.yaml").read_text())
    tokenizer = load_tokenizer()
    target = LlamaForCausalLM.from_pretrained(
        TARGET_MODEL_ID, revision=TARGET_REVISION, dtype=torch.float32
    ).to("cuda").eval()
    eos = tokenizer.eos_token_id
    assert target.generation_config.eos_token_id == eos
    assert target.generation_config.forced_eos_token_id is None
    assert target.generation_config.min_length == 0
    records = frozen_prompts()
    first_20 = [record for record in records if int(record["id"].rsplit("-", 1)[1]) <= 5]
    assert len(first_20) == 20 and len(records) == 100
    # Any subset failure stops the test before checking the full benchmark.
    for stage, stage_records in (("first-20", first_20), ("full", records)):
        for record in stage_records:
            ids = torch.tensor([prompt_token_ids(tokenizer, record["prompt"])], device="cuda")
            budget = record["max_new_tokens"]
            hf = target.generate(
                ids, do_sample=False, max_new_tokens=budget
            )[0, ids.shape[1]:].tolist()
            greedy = greedy_decode(target, ids, max_new_tokens=budget, eos_token_id=eos)
            lookup = prompt_lookup_decode(
                target, ids, max_new_tokens=budget, eos_token_id=eos,
                max_matching_ngram_size=settings["max_matching_ngram_size"],
                prompt_lookup_num_tokens=settings["prompt_lookup_num_tokens"],
            )
            assert lookup.token_ids == greedy.token_ids == hf, (stage, record["id"])


class ScriptedCache:
    def __init__(self):
        self.tokens = []

    def crop(self, tokens_to_remove):
        assert tokens_to_remove < 0
        del self.tokens[tokens_to_remove:]


class ScriptedTarget:
    """Predict a fixed next ID at each absolute position, recording cache inputs."""

    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []

    def __call__(self, *, input_ids, use_cache, past_key_values=None, logits_to_keep=0):
        assert use_cache
        cache = ScriptedCache() if past_key_values is None else past_key_values
        tokens = input_ids[0].tolist()
        start = len(cache.tokens)
        self.calls.append((list(cache.tokens), list(range(start, start + len(tokens))), tokens))
        cache.tokens.extend(tokens)
        next_ids = [self.outputs.get(position, 15) for position in self.calls[-1][1]]
        if logits_to_keep:
            next_ids = next_ids[-logits_to_keep:]
        logits = torch.nn.functional.one_hot(torch.tensor([next_ids]), num_classes=16)
        return SimpleNamespace(logits=logits, past_key_values=cache)


@pytest.mark.parametrize(("continuation", "eos", "budget", "expected", "rows", "calls"), [
    pytest.param(
        [7, 8, 9], None, 3, [7, 8, 9],
        [(0, 0, False, False, 1, 1), (0, 0, False, True, 1, 1)],
        [[7], [8]], id="no-match",
    ),
    pytest.param(
        [2, 7, 8], None, 3, [2, 7, 8],
        [(2, 0, True, False, 1, 1), (0, 0, False, True, 1, 1)],
        [[2, 3, 4], [7]], id="first-rejected",
    ),
    pytest.param(
        [2, 3, 7, 8], None, 4, [2, 3, 7, 8],
        [(2, 1, True, False, 2, 1), (0, 0, False, True, 1, 1)],
        [[2, 3, 4], [7]], id="partially-accepted",
    ),
    pytest.param(
        [2, 3, 4, 7, 8], None, 5, [2, 3, 4, 7, 8],
        [(2, 2, False, False, 3, 1), (0, 0, False, True, 1, 1)],
        [[2, 3, 4], [7]], id="all-accepted-bonus",
    ),
    pytest.param([2], 2, 4, [2], [], [], id="prefill-eos"),
    pytest.param([2], None, 1, [2], [], [], id="prefill-budget"),
    pytest.param(
        [2, 3], 3, 4, [2, 3],
        [(2, 1, False, True, 1, 1)], [[2, 3, 4]], id="accepted-eos",
    ),
    pytest.param(
        [2, 7], 7, 4, [2, 7],
        [(2, 0, True, True, 1, 1)], [[2, 3, 4]], id="correction-eos",
    ),
    pytest.param(
        [2, 3, 4, 7], 7, 5, [2, 3, 4, 7],
        [(2, 2, False, True, 3, 1)], [[2, 3, 4]], id="bonus-eos",
    ),
    pytest.param(
        [2, 3], None, 2, [2, 3],
        [(1, 1, False, True, 1, 1)], [[2, 3]], id="clipped-budget",
    ),
])
def test_scripted_lookup(continuation, eos, budget, expected, rows, calls):
    prompt = [1, 2, 3, 4, 1]
    ids = torch.tensor([prompt])
    outputs = dict(enumerate(continuation, start=len(prompt) - 1))
    target = ScriptedTarget(outputs)
    lookup = prompt_lookup_decode(
        target, ids, max_new_tokens=budget, eos_token_id=eos,
        max_matching_ngram_size=3, prompt_lookup_num_tokens=2,
    )
    greedy = greedy_decode(ScriptedTarget(outputs), ids, max_new_tokens=budget, eos_token_id=eos)
    assert lookup.token_ids == greedy.token_ids == expected
    assert lookup.stopped_on_eos == greedy.stopped_on_eos == (expected[-1] == eos)
    assert lookup.iterations == [DecodeIteration(*row) for row in rows]
    assert target.calls[0] == ([], list(range(len(prompt))), prompt)
    assert [tokens for _, _, tokens in target.calls[1:]] == calls

    emitted_before = 1
    for (cached, positions, tokens), row in zip(target.calls[1:], lookup.iterations):
        assert cached == prompt + expected[:emitted_before - 1]
        assert positions == list(range(len(cached), len(cached) + len(tokens)))
        assert tokens[0] == expected[emitted_before - 1]
        fresh_prefix = torch.tensor([cached + tokens[:1]])
        fresh = ScriptedTarget(outputs)(input_ids=fresh_prefix, use_cache=True, logits_to_keep=1)
        assert fresh.logits[0, -1].argmax().item() == expected[emitted_before]
        emitted_before += row.tokens_emitted
