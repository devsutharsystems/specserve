"""Validated architecture settings for the draft transformer."""

import math
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DraftConfig:
    d_model: int
    n_layer: int
    n_head: int
    hidden_dim: int
    context_length: int
    vocab_size: int
    tokenizer_model_id: str
    rope_base: float
    rms_norm_eps: float
    dropout: float
    init_std: float

    def __post_init__(self):
        for name in ("d_model", "n_layer", "n_head", "hidden_dim", "context_length", "vocab_size"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.d_model % self.n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")
        for name in ("rope_base", "rms_norm_eps", "init_std"):
            value = getattr(self, name)
            if type(value) not in (int, float) or value <= 0 or not math.isfinite(value):
                raise ValueError(f"{name} must be finite and positive")
        if type(self.dropout) not in (int, float) or self.dropout != 0.0:
            raise ValueError("dropout must be 0.0 for the draft architecture")
        if not isinstance(self.tokenizer_model_id, str) or not self.tokenizer_model_id.strip():
            raise ValueError("tokenizer_model_id must be a nonempty string")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DraftConfig":
        settings = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(settings, dict):
            raise ValueError("Draft config YAML must contain a mapping")
        try:
            return cls(**settings)
        except TypeError as error:
            raise ValueError(f"Invalid draft config fields: {error}") from error
