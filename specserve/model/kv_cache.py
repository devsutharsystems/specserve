"""Request-local key/value state for batch-one draft decoding."""

from typing import NamedTuple

import torch

from .config import DraftConfig


class LayerKV(NamedTuple):
    keys: torch.Tensor
    values: torch.Tensor

    def append(self, chunk: "LayerKV") -> "LayerKV":
        for retained, new in zip(self, chunk):
            if retained.device != new.device or retained.dtype != new.dtype:
                raise ValueError("Cache device and dtype must match the projected keys and values")
        return LayerKV(
            torch.cat((self.keys, chunk.keys), dim=-2),
            torch.cat((self.values, chunk.values), dim=-2),
        )


class DraftKVCache:
    """One request's rotated keys and projected values, in model layer order.

    Treat retained tensors as read-only. Start a new cache for a new request
    or after changing model weights.
    """

    def __init__(self, config: DraftConfig, owner: object):
        self._config = config
        self._owner = owner
        self._layers: tuple[LayerKV, ...] = ()

    @property
    def layers(self) -> tuple[LayerKV, ...]:
        return self._layers

    @property
    def length(self) -> int:
        return self._layers[0].keys.shape[-2] if self._layers else 0

    def crop(self, length: int) -> None:
        """Set the retained length, keeping the first `length` positions."""

        if type(length) is not int or not 0 <= length <= self.length:
            raise ValueError(f"Crop length must be an integer between 0 and {self.length}")
        if length == self.length:
            return
        self._layers = tuple(
            LayerKV(pair.keys[..., :length, :], pair.values[..., :length, :])
            for pair in self._layers
        ) if length else ()

    def _check_owner(self, owner: object) -> None:
        if self._owner is not owner:
            raise ValueError("Cache must be created by this draft model")

    def _publish(self, layers: tuple[LayerKV, ...], length: int) -> None:
        if not 1 <= length <= self._config.context_length:
            raise ValueError("Cache length is outside the model context")
        if len(layers) != self._config.n_layer:
            raise ValueError("Cache must contain one K/V pair per model layer")
        expected_shape = (1, self._config.n_head, length, self._config.head_dim)
        reference = layers[0].keys
        for pair in layers:
            for tensor in pair:
                if tensor.shape != expected_shape:
                    raise ValueError(f"Cache tensors must have shape {expected_shape}")
                if tensor.device != reference.device or tensor.dtype != reference.dtype:
                    raise ValueError("All cache tensors must share a device and dtype")
        self._layers = layers
