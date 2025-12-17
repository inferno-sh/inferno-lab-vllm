from __future__ import annotations

from typing import List, Tuple

import torch


class FullKVCache:
    """
    Stores full-dimensional K/V for all tokens. Provides materialization helpers
    that match HuggingFace past_key_values format.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self.keys: List[List[torch.Tensor]] = [
            [] for _ in range(num_layers)
        ]  # per layer list over time of [num_heads, head_dim]
        self.vals: List[List[torch.Tensor]] = [list() for _ in range(num_layers)]

    def append(self, layer: int, key: torch.Tensor, value: torch.Tensor):
        """
        key/value: [num_heads, head_dim]
        """
        self.keys[layer].append(key.detach())
        self.vals[layer].append(value.detach())

    def append_step(
        self, pos: int, keys: List[torch.Tensor], values: List[torch.Tensor]
    ):
        # pos is kept for API parity; storage is append-only.
        for layer, (k, v) in enumerate(zip(keys, values)):
            self.append(layer, k, v)

    def extend_prefill(self, layer: int, keys: torch.Tensor, values: torch.Tensor):
        """
        keys/values: [num_heads, seq_len, head_dim]
        """
        # split along sequence dim
        for t in range(keys.shape[1]):
            self.append(layer, keys[:, t, :], values[:, t, :])

    def __len__(self):
        if not self.keys:
            return 0
        return len(self.keys[0])

    def materialize(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns list of (k, v) where each is [1, num_heads, seq, head_dim].
        """
        output: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer in range(self.num_layers):
            if len(self.keys[layer]) == 0:
                k_cat = torch.zeros(
                    1,
                    self.num_heads,
                    0,
                    self.head_dim,
                    device=self.device,
                    dtype=self.dtype,
                )
                v_cat = k_cat
            else:
                k_cat = (
                    torch.stack(self.keys[layer], dim=1)
                    .unsqueeze(0)
                    .to(device=self.device, dtype=self.dtype)
                )
                v_cat = (
                    torch.stack(self.vals[layer], dim=1)
                    .unsqueeze(0)
                    .to(device=self.device, dtype=self.dtype)
                )
            output.append((k_cat, v_cat))
        return output

    def memory_bytes(self) -> int:
        total = 0
        for layer in range(self.num_layers):
            for k in self.keys[layer]:
                total += k.numel() * k.element_size()
            for v in self.vals[layer]:
                total += v.numel() * v.element_size()
        return total
