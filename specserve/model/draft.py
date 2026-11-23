"""Uncached draft transformer with tied token embeddings."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import DraftConfig
from .layers import RMSNorm, TransformerBlock


class DraftModel(nn.Module):
    def __init__(self, config: DraftConfig):
        super().__init__()
        self.config = config
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
