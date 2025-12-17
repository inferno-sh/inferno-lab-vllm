from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Tuple

import torch


class PackedLayerCache:
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        idx_k: torch.Tensor,
        idx_v: torch.Tensor,
        window: int,
        dtype: torch.dtype,
        device: torch.device,
        store_full_v: bool,
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.idx_k = idx_k
        self.idx_v = idx_v
        self.window = window
        self.dtype = dtype
        self.device = device
        self.store_full_v = store_full_v

        self.window_k: Deque[torch.Tensor] = deque()
        self.window_v: Deque[torch.Tensor] = deque()
        self.window_pos: Deque[int] = deque()
        self.compressed_k: List[torch.Tensor] = []
        self.compressed_v: List[torch.Tensor] = []
        self.compressed_pos: List[int] = []

    def append(self, pos: int, key_full: torch.Tensor, value_full: torch.Tensor):
        self.window_k.append(key_full.detach())
        self.window_v.append(value_full.detach())
        self.window_pos.append(pos)

        if len(self.window_k) > self.window:
            # Evict oldest into compressed storage
            old_k = self.window_k.popleft()
            old_v = self.window_v.popleft()
            old_pos = self.window_pos.popleft()
            packed_k = old_k[:, self.idx_k] if self.idx_k.numel() > 0 else old_k
            if self.store_full_v:
                packed_v = old_v
            else:
                packed_v = old_v[:, self.idx_v] if self.idx_v.numel() > 0 else old_v
            self.compressed_k.append(packed_k.contiguous())
            self.compressed_v.append(packed_v.contiguous())
            self.compressed_pos.append(old_pos)

    def materialize_full(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns tensors shaped [1, num_heads, seq_len, head_dim] with zeros for
        dropped channels reconstructed via scatter.
        """
        parts_k: List[torch.Tensor] = []
        parts_v: List[torch.Tensor] = []
        if self.compressed_k:
            seq_len = len(self.compressed_k)
            k_full = torch.zeros(
                self.num_heads,
                seq_len,
                self.head_dim,
                device=self.device,
                dtype=self.dtype,
            )
            v_dim = self.head_dim if self.store_full_v else self.idx_v.numel()
            v_full = torch.zeros(
                self.num_heads,
                seq_len,
                self.head_dim,
                device=self.device,
                dtype=self.dtype,
            )
            for i, (k, v) in enumerate(zip(self.compressed_k, self.compressed_v)):
                if self.idx_k.numel() > 0 and k.shape[-1] == self.idx_k.numel():
                    k_full[:, i, self.idx_k] = k
                else:
                    k_full[:, i, :] = k
                if self.store_full_v or (
                    self.idx_v.numel() == v.shape[-1] == self.head_dim
                ):
                    v_full[:, i, :] = v
                else:
                    v_full[:, i, self.idx_v] = v
            parts_k.append(k_full.unsqueeze(0))
            parts_v.append(v_full.unsqueeze(0))

        if self.window_k:
            k_win = (
                torch.stack(list(self.window_k), dim=1)
                .unsqueeze(0)
                .to(self.device, self.dtype)
            )
            v_win = (
                torch.stack(list(self.window_v), dim=1)
                .unsqueeze(0)
                .to(self.device, self.dtype)
            )
            parts_k.append(k_win)
            parts_v.append(v_win)

        if not parts_k:
            k_out = torch.zeros(
                1,
                self.num_heads,
                0,
                self.head_dim,
                device=self.device,
                dtype=self.dtype,
            )
            v_out = k_out
        else:
            k_out = torch.cat(parts_k, dim=2)
            v_out = torch.cat(parts_v, dim=2)
        return k_out, v_out

    def memory_bytes(self) -> int:
        total = 0
        for t in self.window_k:
            total += t.numel() * t.element_size()
        for t in self.window_v:
            total += t.numel() * t.element_size()
        for t in self.compressed_k:
            total += t.numel() * t.element_size()
        for t in self.compressed_v:
            total += t.numel() * t.element_size()
        return total


class PackedKVCache:
    """
    Packed KV cache that stores a full-fidelity sliding window and compressed
    older tokens.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        idx_k: List[torch.Tensor],
        idx_v: List[torch.Tensor],
        window: int,
        dtype: torch.dtype,
        device: torch.device,
        store_full_v: bool = False,
    ):
        self.layers: List[PackedLayerCache] = []
        for layer in range(num_layers):
            layer_idx_k = idx_k[layer].to(device)
            layer_idx_v = idx_v[layer].to(device)
            self.layers.append(
                PackedLayerCache(
                    num_heads=num_heads,
                    head_dim=head_dim,
                    idx_k=layer_idx_k,
                    idx_v=layer_idx_v,
                    window=window,
                    dtype=dtype,
                    device=device,
                    store_full_v=store_full_v,
                )
            )
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.window = window
        self.device = device
        self.dtype = dtype
        self.store_full_v = store_full_v

    def append(self, pos: int, keys: List[torch.Tensor], values: List[torch.Tensor]):
        """
        keys/values: list over layers of [num_heads, head_dim] tensors.
        """
        for layer, (k, v) in enumerate(zip(keys, values)):
            self.layers[layer].append(pos, k, v)

    def append_step(
        self, pos: int, keys: List[torch.Tensor], values: List[torch.Tensor]
    ):
        self.append(pos, keys, values)

    def materialize(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        return [layer.materialize_full() for layer in self.layers]

    def memory_bytes(self) -> int:
        return sum(layer.memory_bytes() for layer in self.layers)

    def __len__(self) -> int:
        # All layers share the same lengths.
        return len(self.layers[0].compressed_k) + len(self.layers[0].window_k)

    def to_full(self) -> "FullKVCache":  # type: ignore[name-defined]
        from .kvcache_full import FullKVCache

        full = FullKVCache(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        # Reconstruct per-token keys from compressed + window
        # First, materialize and then split along sequence dim.
        mats = self.materialize()
        for layer, (k, v) in enumerate(mats):
            # k/v shape [1, H, T, D]
            seq_len = k.shape[2]
            for t in range(seq_len):
                full.append(layer, k[0, :, t, :], v[0, :, t, :])
        return full
