"""
Channel importance tracking for dynamic KV cache compression.

This module provides EMA-based importance tracking for attention head dimensions,
enabling dynamic selection of the most important channels for KV cache compression.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch

if TYPE_CHECKING:
    pass

LOG = logging.getLogger(__name__)


@dataclass
class ImportanceConfig:
    """Configuration for channel importance tracking."""

    r_k: int = 64  # Number of K channels to keep
    r_v: int = 96  # Number of V channels to keep
    beta: float = 0.98  # EMA decay factor
    update_interval: int = 32  # Tokens between mask updates
    warmup_tokens: int = 64  # Tokens before enabling compression
    enabled: bool = False


class ChannelImportanceManager:
    """
    Global manager for channel importance tracking across attention layers.

    This singleton tracks per-dimension importance using EMA of magnitudes,
    and periodically recomputes channel masks based on top-k selection.
    """

    _instance: "ChannelImportanceManager | None" = None

    def __init__(self) -> None:
        self.config = ImportanceConfig()
        self.imp_k: torch.Tensor | None = None  # [num_layers, num_heads, head_dim]
        self.imp_v: torch.Tensor | None = None
        self.token_count: int = 0
        self.layer_name_to_idx: dict[str, int] = {}
        self.num_layers: int = 0
        self.num_heads: int = 0
        self.head_dim: int = 0
        self.device: torch.device = torch.device("cpu")
        self._mask_update_callback: (
            Callable[[torch.Tensor, torch.Tensor], None] | None
        ) = None
        self._initialized: bool = False

    @classmethod
    def get(cls) -> "ChannelImportanceManager":
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (useful for testing)."""
        cls._instance = None

    def configure(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        device: torch.device | str,
        r_k: int = 64,
        r_v: int = 96,
        beta: float = 0.98,
        update_interval: int = 32,
        warmup_tokens: int = 64,
    ) -> None:
        """
        Configure and initialize importance tracking.

        Args:
            num_layers: Number of attention layers in the model
            num_heads: Number of KV heads per layer
            head_dim: Dimension of each attention head
            device: Device for importance tensors
            r_k: Number of K channels to keep
            r_v: Number of V channels to keep
            beta: EMA decay factor (higher = slower adaptation)
            update_interval: How often to recompute masks (in tokens)
            warmup_tokens: Tokens to wait before enabling compression
        """
        self.config = ImportanceConfig(
            r_k=min(r_k, head_dim),
            r_v=min(r_v, head_dim),
            beta=beta,
            update_interval=update_interval,
            warmup_tokens=warmup_tokens,
            enabled=True,
        )
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = torch.device(device) if isinstance(device, str) else device

        # Initialize importance tensors
        self.imp_k = torch.zeros(
            num_layers, num_heads, head_dim, device=self.device, dtype=torch.float32
        )
        self.imp_v = torch.zeros_like(self.imp_k)
        self.token_count = 0
        self._initialized = True

        LOG.info(
            "ChannelImportanceManager configured: layers=%d, heads=%d, head_dim=%d, "
            "r_k=%d, r_v=%d, update_interval=%d",
            num_layers,
            num_heads,
            head_dim,
            self.config.r_k,
            self.config.r_v,
            update_interval,
        )

    def register_layer(self, layer_name: str, layer_idx: int) -> None:
        """Register a layer name to index mapping."""
        self.layer_name_to_idx[layer_name] = layer_idx

    def set_mask_update_callback(
        self, callback: Callable[[torch.Tensor, torch.Tensor], None]
    ) -> None:
        """Set callback to be invoked when masks should be updated."""
        self._mask_update_callback = callback

    @property
    def is_enabled(self) -> bool:
        """Check if importance tracking is enabled and initialized."""
        return self.config.enabled and self._initialized

    @property
    def is_warmed_up(self) -> bool:
        """Check if warmup period has completed."""
        return self.token_count >= self.config.warmup_tokens

    @torch.no_grad()
    def update(
        self,
        layer_name: str,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """
        Update importance scores for a layer.

        Args:
            layer_name: Name of the attention layer
            key: Key tensor [num_tokens, num_kv_heads, head_dim]
            value: Value tensor [num_tokens, num_kv_heads, head_dim]
        """
        if not self.is_enabled:
            return

        layer_idx = self.layer_name_to_idx.get(layer_name, -1)
        if layer_idx < 0 or self.imp_k is None or self.imp_v is None:
            return

        # Compute mean absolute value across tokens as importance metric
        # key/value shape: [num_tokens, num_kv_heads, head_dim]
        k_importance = key.abs().mean(dim=0)  # [num_kv_heads, head_dim]
        v_importance = value.abs().mean(dim=0)

        # EMA update
        beta = self.config.beta
        self.imp_k[layer_idx] = beta * self.imp_k[layer_idx] + (1 - beta) * k_importance
        self.imp_v[layer_idx] = beta * self.imp_v[layer_idx] + (1 - beta) * v_importance

    def step(self) -> None:
        """
        Called once per forward pass to track tokens and trigger mask updates.
        """
        if not self.is_enabled:
            return

        self.token_count += 1

        # Check if we should update masks
        if (
            self.is_warmed_up
            and self.token_count % self.config.update_interval == 0
            and self._mask_update_callback is not None
        ):
            mask_k, mask_v = self.compute_masks()
            self._mask_update_callback(mask_k, mask_v)

    def compute_top_indices(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Compute top-k channel indices for each layer.

        Returns:
            Tuple of (idx_k_list, idx_v_list) where each is a list of tensors
            containing the selected channel indices per layer.
        """
        if self.imp_k is None or self.imp_v is None:
            return [], []

        idx_k_list: list[torch.Tensor] = []
        idx_v_list: list[torch.Tensor] = []

        for layer_idx in range(self.num_layers):
            # Average importance across heads
            k_mean = self.imp_k[layer_idx].mean(dim=0)  # [head_dim]
            v_mean = self.imp_v[layer_idx].mean(dim=0)

            # Get top-k indices
            _, idx_k = torch.topk(k_mean, k=self.config.r_k)
            _, idx_v = torch.topk(v_mean, k=self.config.r_v)

            idx_k_list.append(idx_k)
            idx_v_list.append(idx_v)

        return idx_k_list, idx_v_list

    def compute_masks(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute binary masks from importance scores.

        Returns:
            Tuple of (mask_k, mask_v) tensors of shape [head_dim] with 1s for
            selected channels and 0s for masked channels.
        """
        if self.imp_k is None or self.imp_v is None:
            # Return all-ones masks if not initialized
            return (
                torch.ones(self.head_dim, device=self.device),
                torch.ones(self.head_dim, device=self.device),
            )

        # Average importance across all layers and heads
        k_mean = self.imp_k.mean(dim=(0, 1))  # [head_dim]
        v_mean = self.imp_v.mean(dim=(0, 1))

        # Get top-k indices
        _, idx_k = torch.topk(k_mean, k=self.config.r_k)
        _, idx_v = torch.topk(v_mean, k=self.config.r_v)

        # Build masks
        mask_k = torch.zeros(self.head_dim, device=self.device, dtype=torch.float32)
        mask_v = torch.zeros(self.head_dim, device=self.device, dtype=torch.float32)
        mask_k[idx_k] = 1.0
        mask_v[idx_v] = 1.0

        return mask_k, mask_v

    def get_summary(self) -> dict:
        """Get summary statistics for logging."""
        return {
            "enabled": self.is_enabled,
            "token_count": self.token_count,
            "warmed_up": self.is_warmed_up,
            "r_k": self.config.r_k,
            "r_v": self.config.r_v,
            "imp_k_mean": float(self.imp_k.mean().item())
            if self.imp_k is not None
            else 0.0,
            "imp_v_mean": float(self.imp_v.mean().item())
            if self.imp_v is not None
            else 0.0,
        }

    def disable(self) -> None:
        """Disable importance tracking."""
        self.config.enabled = False

    def enable(self) -> None:
        """Enable importance tracking (must be configured first)."""
        if self._initialized:
            self.config.enabled = True
