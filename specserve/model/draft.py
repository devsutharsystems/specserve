"""Draft transformer with tied token embeddings and request-local decoding."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import DraftConfig
from .kv_cache import DraftKVCache
from .layers import RMSNorm, TransformerBlock


class DraftModel(nn.Module):
    def __init__(self, config: DraftConfig):
        super().__init__()
        self.config = config
        self._cache_owner = object()
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.final_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self._initialize_weights()

    def _initialize_weights(self):
        std = self.config.init_std
        residual_std = std / math.sqrt(2 * self.config.n_layer)
        residual_projections = {
            projection
            for block in self.blocks
            for projection in (block.attention.o_proj, block.swiglu.down_proj)
        }
        for module in self.modules():
            if isinstance(module, (nn.Embedding, nn.Linear)):
                scale = residual_std if module in residual_projections else std
                nn.init.normal_(module.weight, mean=0.0, std=scale)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B, T]")
        time = input_ids.shape[1]
        if not 1 <= time <= self.config.context_length:
            raise ValueError(f"Sequence length must be between 1 and {self.config.context_length}")

        positions = torch.arange(time, device=input_ids.device)
        hidden = self.token_embedding(input_ids)
        for block in self.blocks:
            hidden = block(hidden, positions)
        return self._project_logits(hidden)

    def new_cache(self) -> DraftKVCache:
        """Create an empty cache for one decoding request on this model."""

        return DraftKVCache(self.config, self._cache_owner)

    @torch.no_grad()
    def forward_cached(self, input_ids: torch.Tensor, cache: DraftKVCache) -> torch.Tensor:
        """Append a [1, T] token chunk and return its [1, T, V] logits."""

        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("Cached input_ids must have shape [1, T]")
        if not isinstance(cache, DraftKVCache):
            raise TypeError("cache must be a DraftKVCache")
        cache._check_owner(self._cache_owner)
        retained = cache.length
        time = input_ids.shape[1]
        total = retained + time
        if time == 0 or total > self.config.context_length:
            raise ValueError(f"Cached chunk must be nonempty and fit the {self.config.context_length}-token context")

        positions = torch.arange(retained, total, device=input_ids.device)
        mask = None
        if retained and time > 1:
            # Each query sees the prefix and its causal portion of the chunk.
            key_positions = torch.arange(total, device=input_ids.device)
            mask = key_positions[None, :] <= positions[:, None]

        past_layers = cache.layers
        next_layers = []
        hidden = self.token_embedding(input_ids)
        for index, block in enumerate(self.blocks):
            past = past_layers[index] if retained else None
            hidden, layer = block.forward_cached(hidden, positions, past, mask)
            next_layers.append(layer)

        logits = self._project_logits(hidden)
        cache._publish(tuple(next_layers), total)
        return logits

    def _project_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.final_norm(hidden)
        # The vocabulary projection reads the exact parameter used for input lookup.
        return F.linear(hidden, self.token_embedding.weight)

    def parameter_counts(self) -> dict[str, int]:
        """Count trainable parameters by component and as a unique total."""

        def trainable_count(module: nn.Module) -> int:
            return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)

        return {
            "token_embedding": trainable_count(self.token_embedding),
            "attention": sum(trainable_count(block.attention) for block in self.blocks),
            "swiglu": sum(trainable_count(block.swiglu) for block in self.blocks),
            "block_rmsnorm": sum(
                trainable_count(block.attn_norm) + trainable_count(block.ffn_norm) for block in self.blocks
            ),
            "final_rmsnorm": trainable_count(self.final_norm),
            "total": trainable_count(self),
        }
