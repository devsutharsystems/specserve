"""Pre-norm transformer layers for the draft model."""

import torch
import torch.nn.functional as F
from torch import nn

from .config import DraftConfig
from .kv_cache import LayerKV


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accumulate the mean square in FP32 for reduced-precision activations.
        values = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        inverse_rms = (values.square().mean(dim=-1, keepdim=True) + self.eps).rsqrt()
        return (values * inverse_rms * self.weight).to(dtype=x.dtype)


class RotaryEmbedding(nn.Module):
    """Rotate adjacent channel pairs in a tensor shaped [B, H, T, head_dim]."""

    def __init__(self, head_dim: int, context_length: int, base: float):
        super().__init__()
        frequencies = base ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        positions = torch.arange(context_length, dtype=torch.float32)
        angles = torch.outer(positions, frequencies)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[positions].to(dtype=x.dtype)
        sin = self.sin[positions].to(dtype=x.dtype)
        even = x[..., 0::2]
        odd = x[..., 1::2]
        # Stack on a new axis to keep each rotated pair adjacent when flattened.
        rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
        return rotated.flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: DraftConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.dropout = config.dropout
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = RotaryEmbedding(config.head_dim, config.context_length, config.rope_base)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        output, _ = self.forward_cached(x, positions, past=None, mask=None)
        return output

    def forward_cached(
        self, x: torch.Tensor, positions: torch.Tensor,
        past: LayerKV | None, mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, LayerKV]:
        batch, time, width = x.shape
        heads_shape = (batch, time, self.n_head, self.head_dim)
        query = self.q_proj(x).reshape(heads_shape).transpose(1, 2)
        key = self.k_proj(x).reshape(heads_shape).transpose(1, 2)
        value = self.v_proj(x).reshape(heads_shape).transpose(1, 2)
        query = self.rope(query, positions)
        layer = LayerKV(self.rope(key, positions), value)
        if past is not None:
            layer = past.append(layer)

        # SDPA's built-in causal mask aligns only when Q and K lengths match.
        attended = F.scaled_dot_product_attention(
            query, layer.keys, layer.values, attn_mask=mask,
            is_causal=past is None, dropout_p=self.dropout,
        )
        output = self.o_proj(attended.transpose(1, 2).reshape(batch, time, width))
        return output, layer


class SwiGLU(nn.Module):
    def __init__(self, config: DraftConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.hidden_dim, bias=False)
        self.down_proj = nn.Linear(config.hidden_dim, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: DraftConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.swiglu = SwiGLU(config)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self._feed_forward(x + self.attention(self.attn_norm(x), positions))

    def forward_cached(
        self, x: torch.Tensor, positions: torch.Tensor,
        past: LayerKV | None, mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, LayerKV]:
        attended, layer = self.attention.forward_cached(self.attn_norm(x), positions, past, mask)
        return self._feed_forward(x + attended), layer

    def _feed_forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.swiglu(self.ffn_norm(x))
