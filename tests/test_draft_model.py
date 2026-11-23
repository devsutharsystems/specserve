import math
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from specserve.decode.prompting import TARGET_MODEL_ID, TARGET_REVISION, load_tokenizer
from specserve.model.config import DraftConfig
from specserve.model.draft import DraftModel
from specserve.model.layers import RMSNorm, RotaryEmbedding


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/draft_a.yaml"
SEED = 20261118


@pytest.fixture(scope="module")
def draft_config():
    return DraftConfig.from_yaml(CONFIG_PATH)


@pytest.fixture
def toy_config(draft_config):
    return replace(
        draft_config, vocab_size=32, d_model=16, n_layer=2,
        n_head=4, hidden_dim=32, context_length=16,
    )


def seeded_model(config):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        return DraftModel(config).float()


def test_draft_config_and_pinned_tokenizer(draft_config):
    assert asdict(draft_config) == {
        "d_model": 512, "n_layer": 6, "n_head": 8, "hidden_dim": 1408,
        "context_length": 1024, "vocab_size": 32000, "tokenizer_model_id": TARGET_MODEL_ID,
        "rope_base": 10000.0, "rms_norm_eps": 1e-6, "dropout": 0.0, "init_std": 0.02,
    }
    assert draft_config.head_dim == 64
    tokenizer = load_tokenizer(local_files_only=True)
    assert tokenizer.vocab_size == len(tokenizer) == draft_config.vocab_size
    assert set(tokenizer.get_vocab().values()) == set(range(draft_config.vocab_size))
    print(f"tokenizer={TARGET_MODEL_ID}@{TARGET_REVISION}, vocabulary={len(tokenizer)}")


@pytest.mark.parametrize(("changes", "message"), [
    ({"n_layer": True}, "n_layer must be a positive integer"),
    ({"context_length": 0}, "context_length must be a positive integer"),
    ({"d_model": 15}, "d_model must be divisible by n_head"),
    ({"d_model": 12}, "head_dim must be even"),
    ({"rope_base": float("inf")}, "rope_base must be finite and positive"),
    ({"rms_norm_eps": 0.0}, "rms_norm_eps must be finite and positive"),
    ({"dropout": 0.1}, "dropout must be 0.0"),
])
def test_draft_config_rejects_invalid_settings(toy_config, changes, message):
    with pytest.raises(ValueError, match=message):
        replace(toy_config, **changes)


@pytest.mark.parametrize(("content", "message"), [
    ("[]", "must contain a mapping"),
    ("unknown: 1", "unexpected keyword argument"),
])
def test_draft_config_rejects_invalid_yaml_fields(tmp_path, content, message):
    path = tmp_path / "draft.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        DraftConfig.from_yaml(path)


@pytest.mark.parametrize(("shape", "message"), [
    ((3,), "must have shape"),
    ((1, 0), "Sequence length"),
    ((1, 17), "Sequence length"),
])
def test_draft_rejects_invalid_input_shape(toy_config, shape, message):
    model = seeded_model(toy_config)
    with pytest.raises(ValueError, match=message):
        model(torch.zeros(shape, dtype=torch.long))


def test_rmsnorm_matches_formula(draft_config):
    eps = draft_config.rms_norm_eps
    norm = RMSNorm(3, eps)
    scales = [0.5, 2.0, -1.0]
    rows = [[3.0, -4.0, 12.0], [0.0001, -0.0003, 0.0002], [0.0, 0.0, 0.0]]
    expected = []
    for row in rows:
        rms = math.sqrt(sum(value ** 2 for value in row) / len(row) + eps)
        expected.append([scale * value / rms for scale, value in zip(scales, row)])
    with torch.no_grad():
        norm.weight.copy_(torch.tensor(scales))
        actual = norm(torch.tensor([rows]))
    torch.testing.assert_close(actual, torch.tensor([expected]))


def test_rope_matches_rotation_and_preserves_relative_position(toy_config):
    rope = RotaryEmbedding(toy_config.head_dim, toy_config.context_length, toy_config.rope_base)
    q = torch.tensor([[[[1.0, -2.0, 3.0, 4.0]]]])
    k = torch.tensor([[[[4.0, 1.0, -3.0, 2.0]]]])

    def rotate(vector, position):
        return rope(vector, torch.tensor([position]))

    position = 3
    expected = []
    for pair, (even, odd) in enumerate(zip(q.flatten()[::2], q.flatten()[1::2])):
        angle = position * toy_config.rope_base ** (-2 * pair / toy_config.head_dim)
        expected.extend((
            even.item() * math.cos(angle) - odd.item() * math.sin(angle),
            even.item() * math.sin(angle) + odd.item() * math.cos(angle),
        ))
    torch.testing.assert_close(rotate(q, position), torch.tensor(expected).reshape_as(q))

    def dot(query_position, key_position):
        return (rotate(q, query_position) * rotate(k, key_position)).sum()

    torch.testing.assert_close(dot(1, 4), dot(3, 6))
    assert not torch.isclose(dot(1, 4), dot(1, 5))
    assert rope.state_dict() == {}


def test_causal_attention_matches_explicit_mask(toy_config):
    attention = seeded_model(toy_config).blocks[0].attention
    x = torch.linspace(-1.0, 1.0, steps=3 * toy_config.d_model).reshape(1, 3, -1)
    positions = torch.arange(3)
    with torch.no_grad():
        for layer in (attention.q_proj, attention.k_proj, attention.v_proj, attention.o_proj):
            layer.weight.copy_(torch.eye(toy_config.d_model))
        actual = attention(x, positions)
        heads = x.reshape(1, 3, toy_config.n_head, toy_config.head_dim).transpose(1, 2)
        rotated = attention.rope(heads, positions)
        scores = rotated @ rotated.transpose(-2, -1) / math.sqrt(toy_config.head_dim)
        future = torch.ones(3, 3, dtype=torch.bool).triu(diagonal=1)
        probabilities = scores.masked_fill(future, -float("inf")).softmax(dim=-1)
        expected = (probabilities @ heads).transpose(1, 2).reshape_as(x)
    torch.testing.assert_close(actual, expected)


def test_swiglu_matches_gated_projection(toy_config):
    model = seeded_model(toy_config)
    swiglu = model.blocks[0].swiglu
    x = torch.linspace(-1.0, 1.0, steps=2 * toy_config.d_model).reshape(1, 2, -1)
    with torch.no_grad():
        gate = x @ swiglu.gate_proj.weight.T
        up = x @ swiglu.up_proj.weight.T
        assert gate.shape == up.shape == (1, 2, toy_config.hidden_dim)
        expected = (gate * torch.sigmoid(gate) * up) @ swiglu.down_proj.weight.T
        torch.testing.assert_close(swiglu(x), expected)


def test_future_token_cannot_change_earlier_draft_logits(toy_config):
    model = seeded_model(toy_config)
    original = torch.tensor([[1, 5, 9, 4, 7]])
    changed = original.clone()
    changed[0, 2] = 6
    with torch.no_grad():
        before, after = model(original), model(changed)
    torch.testing.assert_close(before[:, :2], after[:, :2])
    assert not torch.allclose(before[:, 2:], after[:, 2:])


def test_draft_initialization_scales_and_biases(toy_config, monkeypatch):
    calls = {}
    normal_ = nn.init.normal_

    def record_normal(tensor, mean=0.0, std=1.0, **kwargs):
        calls[id(tensor)] = (mean, std)
        return normal_(tensor, mean=mean, std=std, **kwargs)

    monkeypatch.setattr(nn.init, "normal_", record_normal)
    model = seeded_model(toy_config)
    expected_residual_std = toy_config.init_std / math.sqrt(2 * toy_config.n_layer)
    residuals = {
        projection
        for block in model.blocks
        for projection in (block.attention.o_proj, block.swiglu.down_proj)
    }
    for module in model.modules():
        if isinstance(module, (nn.Embedding, nn.Linear)):
            expected_std = expected_residual_std if module in residuals else toy_config.init_std
            assert calls[id(module.weight)] == (0.0, expected_std)
        if isinstance(module, nn.Linear):
            assert module.bias is None
        if isinstance(module, RMSNorm):
            torch.testing.assert_close(module.weight, torch.ones_like(module.weight), rtol=0, atol=0)


def test_draft_full_config_parameter_count_logits_and_initial_loss(draft_config):
    model = seeded_model(draft_config)
    counts = model.parameter_counts()
    assert counts == {
        "token_embedding": 16_384_000, "attention": 6_291_456, "swiglu": 12_976_128,
        "block_rmsnorm": 6_144, "final_rmsnorm": 512, "total": 35_658_240,
    }
    assert counts["total"] == sum(count for component, count in counts.items() if component != "total")
    assert counts["total"] == sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    tokens = torch.randint(draft_config.vocab_size, (1, 9), generator=generator)
    with torch.no_grad():
        logits = model(tokens[:, :-1])
        assert logits.shape == (1, 8, 32000)
        assert logits.dtype == torch.float32 and torch.isfinite(logits).all()
        loss = F.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten()).item()
    reference = math.log(draft_config.vocab_size)
    print(f"draft_config={asdict(draft_config)}")
    print(f"model_and_token_seed={SEED}")
    print(f"parameter_counts={counts}")
    print(f"initial_logits_shape={tuple(logits.shape)}")
    print(f"initial_loss={loss}, log_vocab={reference}, abs_deviation={abs(loss - reference)}")
    assert math.isfinite(loss) and abs(loss - reference) <= 0.5


def test_draft_tied_projection_and_backward(toy_config):
    model = seeded_model(toy_config)
    ids = torch.tensor([[1, 5, 9]])
    unused = toy_config.vocab_size - 1

    # Changing an unused lookup row isolates its role in the output projection.
    with torch.no_grad():
        before = model(ids)
        model.token_embedding.weight[unused, 0].add_(1.0)
        after = model(ids)
    torch.testing.assert_close(after[..., :unused], before[..., :unused])
    assert not torch.allclose(after[..., unused], before[..., unused])

    labels = torch.tensor([5, 9, 2])
    loss = F.cross_entropy(model(ids).flatten(0, 1), labels)
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    # An unused input row receives a gradient through the tied output matrix.
    assert torch.count_nonzero(model.token_embedding.weight.grad[unused])
    assert torch.count_nonzero(model.blocks[0].attention.q_proj.weight.grad)
    assert torch.count_nonzero(model.blocks[0].swiglu.gate_proj.weight.grad)
