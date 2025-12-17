# SPDX-License-Identifier: Apache-2.0
"""
Latent KV Attention module with learned low-rank K/V projections.

This module provides a drop-in replacement for standard attention that uses
low-rank projections for K and V, enabling significant KV cache size reduction.

Key features:
- Separate latent dimensions for K and V (r_k, r_v)
- Optional anchor vector for early layers (exploits mean direction structure)
- Compatible with GQA (grouped-query attention)
- RoPE support
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from latent_kv.config import LayerLatentConfig


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 32768,
        base: float = 10000.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (unused, just for device/dtype)
            position_ids: Position indices [batch, seq_len]

        Returns:
            cos, sin tensors for rotary embedding
        """
        inv_freq_expanded = self.inv_freq[None, :, None].expand(
            position_ids.shape[0], -1, 1
        )
        position_ids_expanded = position_ids[:, None, :].float()

        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)

        cos = emb.cos()
        sin = emb.sin()

        return cos.to(x.dtype), sin.to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LatentKVAttention(nn.Module):
    """
    Attention with learned low-rank KV projections.

    This module replaces standard K/V projections with low-rank factorizations:
        K = k_anchor + X @ W_k_down @ W_k_up  (if use_k_anchor)
        K = X @ W_k_down @ W_k_up             (otherwise)
        V = X @ W_v_down @ W_v_up

    The latent dimensions r_k and r_v can be different, allowing asymmetric
    compression based on the empirical finding that K is more compressible than V.

    STAGED COMPRESSION (v5):
    Parameters are sized to max ranks (r_k_max, r_v_max), but effective ranks
    (r_k_eff, r_v_eff) can be set lower. During forward, only the first
    r_k_eff/r_v_eff components are used. This enables rank curriculum during
    training without reallocating weights.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        d_head: int,
        layer_config: LayerLatentConfig,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 32768,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head

        # Max ranks (parameter size) vs effective ranks (used in forward)
        self.r_k_max = layer_config.r_k
        self.r_v_max = layer_config.r_v
        # Effective ranks can be changed during training for staged compression
        self._r_k_eff = layer_config.r_k
        self._r_v_eff = layer_config.r_v

        # Legacy attributes for compatibility
        self.r_k = layer_config.r_k
        self.r_v = layer_config.r_v

        self.layer_idx = layer_idx
        self.num_kv_groups = n_heads // n_kv_heads
        self.scaling = 1.0 / math.sqrt(d_head)

        # Query projection (full rank, unchanged)
        self.q_proj = nn.Linear(d_model, n_heads * d_head, bias=False)

        # Low-rank K projection: d_model -> r_k_max -> n_kv_heads * d_head
        # (sized to max, effective rank controlled by masking)
        self.k_down = nn.Linear(d_model, self.r_k_max, bias=False)
        self.k_up = nn.Linear(self.r_k_max, n_kv_heads * d_head, bias=False)

        # Low-rank V projection: d_model -> r_v_max -> n_kv_heads * d_head
        self.v_down = nn.Linear(d_model, self.r_v_max, bias=False)
        self.v_up = nn.Linear(self.r_v_max, n_kv_heads * d_head, bias=False)

        # Optional K anchor (for early layers with dominant mean direction)
        self.use_k_anchor = layer_config.use_k_anchor
        if self.use_k_anchor:
            self.k_anchor = nn.Parameter(torch.zeros(n_kv_heads * d_head))

        # Optional V anchor
        self.use_v_anchor = layer_config.use_v_anchor
        if self.use_v_anchor:
            self.v_anchor = nn.Parameter(torch.zeros(n_kv_heads * d_head))

        # Output projection (full rank, unchanged)
        self.o_proj = nn.Linear(n_heads * d_head, d_model, bias=False)

        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            d_head,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )

    # =========================================================================
    # STAGED COMPRESSION: Effective rank properties for rank curriculum
    # =========================================================================

    @property
    def r_k_eff(self) -> int:
        """Current effective K rank (can be < r_k_max during staged training)."""
        return self._r_k_eff

    @r_k_eff.setter
    def r_k_eff(self, value: int):
        """Set effective K rank. Must be <= r_k_max."""
        if value > self.r_k_max:
            raise ValueError(f"r_k_eff ({value}) cannot exceed r_k_max ({self.r_k_max})")
        self._r_k_eff = value

    @property
    def r_v_eff(self) -> int:
        """Current effective V rank (can be < r_v_max during staged training)."""
        return self._r_v_eff

    @r_v_eff.setter
    def r_v_eff(self, value: int):
        """Set effective V rank. Must be <= r_v_max."""
        if value > self.r_v_max:
            raise ValueError(f"r_v_eff ({value}) cannot exceed r_v_max ({self.r_v_max})")
        self._r_v_eff = value

    def set_effective_ranks(self, r_k_eff: int, r_v_eff: int):
        """Set effective ranks for staged compression."""
        self.r_k_eff = r_k_eff
        self.r_v_eff = r_v_eff

    def get_effective_compression(self) -> float:
        """Get current compression ratio based on effective ranks."""
        full_kv = 2 * self.n_kv_heads * self.d_head
        latent_kv = self._r_k_eff + self._r_v_eff
        return 1.0 - latent_kv / full_kv

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_values=None,  # HuggingFace compatibility
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with low-rank KV projections.

        Compatible with HuggingFace Qwen2Attention interface.

        Args:
            hidden_states: Input tensor [batch, seq_len, d_model]
            position_embeddings: Tuple of (cos, sin) from RoPE (HF style)
            attention_mask: Attention mask [batch, 1, seq_len, seq_len]
            position_ids: Position indices [batch, seq_len] (legacy)
            past_key_value: Cached (K, V) from previous steps (tuple style)
            past_key_values: HuggingFace Cache object (HF style)
            output_attentions: Whether to return attention weights
            use_cache: Whether to return updated cache
            cache_position: Position in cache (HF style)
            **kwargs: Additional arguments (ignored)

        Returns:
            output: Attention output [batch, seq_len, d_model]
            attn_weights: Attention weights (if output_attentions), else None
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Handle HF-style past_key_values
        if past_key_values is not None and past_key_value is None:
            # HF Cache object - extract past K/V if available
            if hasattr(past_key_values, "get"):
                past_key_value = past_key_values.get(self.layer_idx)

        # Query projection (full rank)
        query = self.q_proj(hidden_states)

        # Key projection (low-rank with rank masking for staged compression)
        k_latent_full = self.k_down(hidden_states)  # [B, S, r_k_max]

        # Apply rank masking: only use first r_k_eff components
        if self._r_k_eff < self.r_k_max:
            k_latent = k_latent_full[:, :, :self._r_k_eff]
            # Use only corresponding rows of k_up weight
            key = F.linear(k_latent, self.k_up.weight[:, :self._r_k_eff])
        else:
            key = self.k_up(k_latent_full)  # [B, S, n_kv_heads * d_head]

        if self.use_k_anchor:
            key = key + self.k_anchor

        # Value projection (low-rank with rank masking)
        v_latent_full = self.v_down(hidden_states)  # [B, S, r_v_max]

        # Apply rank masking: only use first r_v_eff components
        if self._r_v_eff < self.r_v_max:
            v_latent = v_latent_full[:, :, :self._r_v_eff]
            # Use only corresponding rows of v_up weight
            value = F.linear(v_latent, self.v_up.weight[:, :self._r_v_eff])
        else:
            value = self.v_up(v_latent_full)  # [B, S, n_kv_heads * d_head]

        if self.use_v_anchor:
            value = value + self.v_anchor

        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.n_kv_heads, self.d_head).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.n_kv_heads, self.d_head).transpose(1, 2)

        # Apply rotary embeddings
        if position_embeddings is not None:
            # HF style: position embeddings passed in
            cos, sin = position_embeddings
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
        else:
            # Legacy style: compute internally
            if position_ids is None:
                position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
            cos, sin = self.rotary_emb(query, position_ids)
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # Handle KV cache
        if past_key_value is not None:
            if isinstance(past_key_value, tuple):
                past_key, past_value = past_key_value
                key = torch.cat([past_key, key], dim=2)
                value = torch.cat([past_value, value], dim=2)

        # Update cache if using HF Cache object
        if past_key_values is not None and hasattr(past_key_values, "update"):
            key, value = past_key_values.update(key, value, self.layer_idx, {"cache_position": cache_position})

        # Expand KV for GQA
        key_expanded = key
        value_expanded = value
        if self.num_kv_groups > 1:
            key_expanded = key.repeat_interleave(self.num_kv_groups, dim=1)
            value_expanded = value.repeat_interleave(self.num_kv_groups, dim=1)

        # Compute attention scores
        attn_weights = torch.matmul(query, key_expanded.transpose(-2, -1)) * self.scaling

        # Apply attention mask
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :key_expanded.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_expanded)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.n_heads * self.d_head)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights

    def get_latent_kv(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get latent KV representations (for efficient caching).

        Instead of caching full K/V, cache the latent representations
        and expand at decode time.

        Returns:
            k_latent: [batch, seq_len, r_k]
            v_latent: [batch, seq_len, r_v]
        """
        k_latent = self.k_down(hidden_states)
        v_latent = self.v_down(hidden_states)
        return k_latent, v_latent

    def expand_latent_kv(
        self,
        k_latent: torch.Tensor,
        v_latent: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Expand latent representations to full K/V.

        Args:
            k_latent: [batch, seq_len, r_k]
            v_latent: [batch, seq_len, r_v]

        Returns:
            key: [batch, seq_len, n_kv_heads * d_head]
            value: [batch, seq_len, n_kv_heads * d_head]
        """
        key = self.k_up(k_latent)
        if self.use_k_anchor:
            key = key + self.k_anchor

        value = self.v_up(v_latent)
        if self.use_v_anchor:
            value = value + self.v_anchor

        return key, value

    @classmethod
    def from_standard_attention(
        cls,
        standard_attn: nn.Module,
        layer_config: LayerLatentConfig,
        layer_idx: int = 0,
        init_method: str = "svd",
    ) -> "LatentKVAttention":
        """
        Create a LatentKVAttention from a standard attention module.

        Args:
            standard_attn: Original attention module (e.g., Qwen2Attention)
            layer_config: Latent dimension configuration
            layer_idx: Layer index
            init_method: "svd" for SVD-based init, "random" for random init

        Returns:
            LatentKVAttention initialized from the standard attention
        """
        # Extract dimensions from original attention
        d_model = standard_attn.q_proj.in_features
        d_head = standard_attn.head_dim

        # Handle different attribute naming conventions
        # Qwen2 uses config.num_attention_heads, some models use num_heads directly
        if hasattr(standard_attn, "num_heads"):
            n_heads = standard_attn.num_heads
        elif hasattr(standard_attn, "config"):
            n_heads = standard_attn.config.num_attention_heads
        else:
            n_heads = standard_attn.q_proj.out_features // d_head

        if hasattr(standard_attn, "num_key_value_heads"):
            n_kv_heads = standard_attn.num_key_value_heads
        elif hasattr(standard_attn, "config"):
            n_kv_heads = getattr(standard_attn.config, "num_key_value_heads", n_heads)
        else:
            n_kv_heads = standard_attn.k_proj.out_features // d_head

        # Get RoPE config - try multiple sources
        if hasattr(standard_attn, "rotary_emb"):
            rope_theta = getattr(standard_attn.rotary_emb, "base", 10000.0)
            max_pos = getattr(standard_attn.rotary_emb, "max_position_embeddings", 32768)
        elif hasattr(standard_attn, "config"):
            rope_theta = getattr(standard_attn.config, "rope_theta", 10000.0)
            max_pos = getattr(standard_attn.config, "max_position_embeddings", 32768)
        else:
            rope_theta = 10000.0
            max_pos = 32768

        # Get device and dtype from original attention
        device = standard_attn.q_proj.weight.device
        dtype = standard_attn.q_proj.weight.dtype

        # Create latent attention module
        latent_attn = cls(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            d_head=d_head,
            layer_config=layer_config,
            rope_theta=rope_theta,
            max_position_embeddings=max_pos,
            layer_idx=layer_idx,
        )

        # Move to same device/dtype as original
        latent_attn = latent_attn.to(device=device, dtype=dtype)

        # Copy Q and O projections directly
        latent_attn.q_proj.weight.data.copy_(standard_attn.q_proj.weight.data)
        latent_attn.o_proj.weight.data.copy_(standard_attn.o_proj.weight.data)

        # Initialize low-rank projections
        if init_method == "svd":
            _init_from_svd(
                latent_attn,
                standard_attn.k_proj.weight.data,
                standard_attn.v_proj.weight.data,
                layer_config.r_k,
                layer_config.r_v,
            )
        elif init_method == "random":
            # Random initialization (will need more training)
            pass
        else:
            raise ValueError(f"Unknown init_method: {init_method}")

        return latent_attn


def _init_from_svd(
    latent_attn: LatentKVAttention,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    r_k: int,
    r_v: int,
):
    """
    Initialize low-rank projections using SVD of original weights.

    For W of shape [out_features, in_features]:
    - Compute SVD: W^T = U @ S @ V^T
    - Set W_down = (U[:, :r] @ S[:r, :r]^0.5)^T
    - Set W_up = S[:r, :r]^0.5 @ V[:r, :]
    """
    device = W_k.device
    dtype = W_k.dtype

    # K projection: W_k is [n_kv_heads * d_head, d_model]
    # We want to factorize W_k^T = [d_model, n_kv_heads * d_head]
    W_k_t = W_k.T.float()  # [d_model, out]
    U_k, S_k, Vh_k = torch.linalg.svd(W_k_t, full_matrices=False)

    # Take top r_k components
    sqrt_S_k = torch.sqrt(S_k[:r_k])
    # k_down.weight: [r_k, d_model] (Linear in_features=d_model, out_features=r_k)
    # k_up.weight: [n_kv_heads * d_head, r_k] (Linear in_features=r_k, out_features=n_kv_heads*d_head)
    latent_attn.k_down.weight.data = (U_k[:, :r_k] * sqrt_S_k).T.to(dtype)
    latent_attn.k_up.weight.data = (sqrt_S_k.unsqueeze(1) * Vh_k[:r_k, :]).T.to(dtype)

    # V projection
    W_v_t = W_v.T.float()
    U_v, S_v, Vh_v = torch.linalg.svd(W_v_t, full_matrices=False)

    sqrt_S_v = torch.sqrt(S_v[:r_v])
    latent_attn.v_down.weight.data = (U_v[:, :r_v] * sqrt_S_v).T.to(dtype)
    latent_attn.v_up.weight.data = (sqrt_S_v.unsqueeze(1) * Vh_v[:r_v, :]).T.to(dtype)

    # If using K anchor, initialize to zero (or could use mean of K)
    if latent_attn.use_k_anchor:
        latent_attn.k_anchor.data.zero_()

    if latent_attn.use_v_anchor:
        latent_attn.v_anchor.data.zero_()
