from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from specserve.model.config import DraftConfig
from specserve.model.draft import DraftModel
from specserve.model.kv_cache import LayerKV


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs/draft_a.yaml"
SEED = 20261118
RTOL = 1.3e-6
ATOL = 1e-5


@pytest.fixture(scope="module")
def draft_config():
    return DraftConfig.from_yaml(CONFIG_PATH)


def seeded_model(config):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        return DraftModel(config).float().eval()


@pytest.fixture
def model(draft_config):
    return seeded_model(replace(
        draft_config, vocab_size=32, d_model=16, n_layer=2,
        n_head=4, hidden_dim=32, context_length=16,
    ))


def assert_logit_agreement(actual, expected, label):
    assert actual.shape == expected.shape
    assert actual.device.type == expected.device.type == "cpu"
    assert actual.dtype == expected.dtype == torch.float32
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    print(f"max_abs_logit_diff[{label}]={float((actual - expected).abs().max())}")
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    assert torch.equal(actual.argmax(dim=-1), expected.argmax(dim=-1))


def assert_cache_layout(model, cache, length):
    assert cache.length == length
    assert len(cache.layers) == model.config.n_layer
    for pair in cache.layers:
        for tensor in pair:
            assert tensor.shape == (1, model.config.n_head, length, model.config.head_dim)
            assert tensor.device.type == "cpu"
            assert tensor.dtype == torch.float32
            assert torch.isfinite(tensor).all()
            assert not tensor.requires_grad


@pytest.mark.parametrize("chunks", [(8,), (3, 1, 1, 1, 1, 1), (3, 2, 3), (1, 3, 4)])
@torch.no_grad()
def test_cached_logits_match_full_sequence_at_different_chunk_boundaries(model, chunks):
    ids = torch.tensor([[2, 7, 1, 18, 5, 11, 4, 23]])
    expected = model(ids)
    cache = model.new_cache()
    assert cache.length == 0 and cache.layers == ()
    start = 0
    for size in chunks:
        end = start + size
        prior = cache.layers
        actual = model.forward_cached(ids[:, start:end], cache)
        assert actual.shape == (1, size, model.config.vocab_size)
        assert_logit_agreement(actual, expected[:, start:end], f"toy {chunks} rows {start}:{end}")
        assert_cache_layout(model, cache, end)
        for before, after in zip(prior, cache.layers):
            for before_tensor, after_tensor in zip(before, after):
                assert torch.equal(before_tensor, after_tensor[..., :start, :])
        start = end
    assert start == ids.shape[1]


@torch.no_grad()
def test_cache_stores_rotated_keys_and_projected_values(model):
    ids = torch.tensor([[2, 7, 1, 18]])
    cache = model.new_cache()
    model.forward_cached(ids[:, :2], cache)
    model.forward_cached(ids[:, 2:], cache)

    block = model.blocks[0]
    hidden = block.attn_norm(model.token_embedding(ids))
    shape = (1, ids.shape[1], model.config.n_head, model.config.head_dim)
    keys = block.attention.k_proj(hidden).reshape(shape).transpose(1, 2)
    values = block.attention.v_proj(hidden).reshape(shape).transpose(1, 2)
    rotated = block.attention.rope(keys, torch.arange(ids.shape[1]))
    assert not torch.allclose(keys, rotated)
    torch.testing.assert_close(cache.layers[0].keys, rotated, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(cache.layers[0].values, values, rtol=RTOL, atol=ATOL)


@torch.no_grad()
def assert_greedy_continuation(model, ids, cache, logits, steps, label):
    for step in range(steps):
        expected = model(ids)[:, -1:, :]
        assert_logit_agreement(logits, expected, f"{label} greedy step {step}")
        next_id = logits.argmax(dim=-1)
        ids = torch.cat((ids, next_id), dim=1)
        logits = model.forward_cached(next_id, cache)
        assert cache.length == ids.shape[1]
    assert_logit_agreement(logits, model(ids)[:, -1:, :], f"{label} final greedy row")


def test_cached_greedy_matches_full_sequence(model):
    prefix = torch.tensor([[2, 7, 1, 18]])
    cache = model.new_cache()
    logits = model.forward_cached(prefix, cache)
    assert_greedy_continuation(model, prefix, cache, logits[:, -1:, :], 5, "prefill")


def test_cached_chunk_future_token_cannot_change_earlier_logits(model):
    prefix = torch.tensor([[2, 7, 1]])
    first, second = model.new_cache(), model.new_cache()
    model.forward_cached(prefix, first)
    model.forward_cached(prefix, second)
    chunk = torch.tensor([[18, 5, 11, 4]])
    changed = chunk.clone()
    changed[0, 2] = 23
    original = model.forward_cached(chunk, first)
    perturbed = model.forward_cached(changed, second)
    torch.testing.assert_close(original[:, :2], perturbed[:, :2], rtol=RTOL, atol=ATOL)
    assert float((original[:, 2:] - perturbed[:, 2:]).abs().max()) > ATOL


@torch.no_grad()
def test_crop_keeps_prefix_and_regenerates_without_rejected_suffix(model):
    prefix = torch.tensor([[2, 7, 1, 18]])
    rejected = torch.tensor([[5, 11, 4]])
    replacement = torch.tensor([[6, 14, 12]])
    cache = model.new_cache()
    prefilling = model.forward_cached(prefix, cache)
    prefix_layers = cache.layers
    model.forward_cached(rejected, cache)
    cache.crop(prefix.shape[1])
    assert_cache_layout(model, cache, prefix.shape[1])
    for cropped, original in zip(cache.layers, prefix_layers):
        assert torch.equal(cropped.keys, original.keys)
        assert torch.equal(cropped.values, original.values)

    actual = model.forward_cached(replacement, cache)
    ids = torch.cat((prefix, replacement), dim=1)
    assert_logit_agreement(actual, model(ids)[:, -replacement.shape[1]:], "replacement after crop")
    assert_greedy_continuation(model, ids, cache, actual[:, -1:, :], 3, "after crop")

    cache.crop(0)
    assert cache.length == 0 and cache.layers == ()
    torch.testing.assert_close(model.forward_cached(prefix, cache), prefilling, rtol=0, atol=0)


@torch.no_grad()
def test_context_limit_and_invalid_inputs_keep_cache_intact(model):
    length = model.config.context_length
    ids = torch.arange(length).unsqueeze(0)
    cache = model.new_cache()
    model.forward_cached(ids[:, :-2], cache)
    logits = model.forward_cached(ids[:, -2:], cache)
    assert_cache_layout(model, cache, length)
    assert_logit_agreement(logits, model(ids)[:, -2:], "end of toy context")
    before = cache.layers
    for tokens, error in ((ids[0], "shape"), (ids.expand(2, -1), "shape"),
                          (ids[:, :0], "nonempty"), (ids[:, :1], "context")):
        with pytest.raises(ValueError, match=error):
            model.forward_cached(tokens, cache)
        assert cache.layers is before
    for invalid_length in (-1, length + 1, True, 1.0):
        with pytest.raises(ValueError, match="Crop length"):
            cache.crop(invalid_length)
        assert cache.layers is before
    cache.crop(length)
    assert cache.layers is before


def test_cache_is_request_local_and_belongs_to_its_model(model):
    first, second = model.new_cache(), model.new_cache()
    ids = torch.tensor([[2, 7, 1]])
    model.forward_cached(ids, first)
    before = first.layers
    assert second.length == 0
    model.forward_cached(ids[:, :1], second)
    assert second.length == 1 and first.layers is before
    with pytest.raises(ValueError, match="created by this draft model"):
        seeded_model(model.config).forward_cached(ids, first)
    assert first.layers is before


def test_incompatible_model_dtype_leaves_cache_intact(model):
    cache = model.new_cache()
    ids = torch.tensor([[2, 7, 1]])
    model.forward_cached(ids, cache)
    before = cache.layers
    with pytest.raises(ValueError, match="device and dtype must match"):
        model.double().forward_cached(ids, cache)
    assert cache.layers is before


@pytest.mark.parametrize("failure", ["missing layer", "length", "shape", "device", "dtype"])
def test_cache_rejects_inconsistent_layer_updates(model, failure):
    cache = model.new_cache()
    model.forward_cached(torch.tensor([[2, 7, 1]]), cache)
    before = cache.layers
    keys, values = before[-1]
    length = cache.length
    if failure == "missing layer":
        layers, message = before[:-1], "one K/V pair"
    elif failure == "length":
        layers, message = before, "outside the model context"
        length = model.config.context_length + 1
    else:
        invalid_values, message = {
            "shape": (values[..., :-1, :], "must have shape"),
            "device": (torch.empty_like(values, device="meta"), "share a device and dtype"),
            "dtype": (values.double(), "share a device and dtype"),
        }[failure]
        layers = (*before[:-1], LayerKV(keys, invalid_values))
    with pytest.raises(ValueError, match=message):
        cache._publish(layers, length)
    assert cache.layers is before and cache.length == 3


@pytest.mark.parametrize("failure_site", ["second block", "logit projection"])
@torch.no_grad()
def test_failed_forward_leaves_all_cache_entries_unchanged(model, monkeypatch, failure_site):
    prefix = torch.tensor([[2, 7, 1]])
    chunk = torch.tensor([[18, 5]])
    cache = model.new_cache()
    model.forward_cached(prefix, cache)
    before = cache.layers
    copies = [LayerKV(pair.keys.clone(), pair.values.clone()) for pair in before]

    def fail(*args):
        raise RuntimeError("injected forward failure")

    with monkeypatch.context() as patch:
        if failure_site == "second block":
            patch.setattr(model.blocks[1], "forward_cached", fail)
        else:
            patch.setattr(model, "_project_logits", fail)
        with pytest.raises(RuntimeError, match="injected forward failure"):
            model.forward_cached(chunk, cache)
    assert cache.layers is before and cache.length == prefix.shape[1]
    for pair, copy in zip(cache.layers, copies):
        assert torch.equal(pair.keys, copy.keys)
        assert torch.equal(pair.values, copy.values)

    actual = model.forward_cached(chunk, cache)
    expected = model(torch.cat((prefix, chunk), dim=1))[:, -chunk.shape[1]:]
    assert_logit_agreement(actual, expected, f"retry after {failure_site} failure")


def test_uncached_backward_after_cached_forward(model):
    ids = torch.tensor([[2, 7, 1, 18]])
    lookup = model.forward_cached(ids[:, :-1], model.new_cache())
    assert not lookup.requires_grad
    logits = model(ids[:, :-1])
    loss = F.cross_entropy(logits.flatten(0, 1), ids[:, 1:].flatten())
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


@torch.no_grad()
def test_full_draft_cache_logits_parameters_and_context_limit(draft_config):
    model = seeded_model(draft_config)
    assert model.parameter_counts()["total"] == 35_658_240
    assert draft_config.context_length == 1024
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    ids = torch.randint(draft_config.vocab_size, (1, 8), generator=generator)
    cache = model.new_cache()
    expected = model(ids)
    start = 0
    for size in (3, 1, 4):
        end = start + size
        actual = model.forward_cached(ids[:, start:end], cache)
        assert actual.shape == (1, size, 32000)
        assert_logit_agreement(actual, expected[:, start:end], f"draft-a rows {start}:{end}")
        assert_cache_layout(model, cache, end)
        start = end

    before = cache.layers
    with pytest.raises(ValueError, match="1024-token context"):
        model.forward_cached(torch.zeros((1, 1017), dtype=torch.long), cache)
    assert cache.layers is before and cache.length == 8
    assert model.parameter_counts()["total"] == 35_658_240
    print(f"draft_a_trainable_parameters={model.parameter_counts()['total']}")
