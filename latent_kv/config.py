# SPDX-License-Identifier: Apache-2.0
"""
Configuration for layer-adaptive latent KV dimensions.

Based on empirical rank analysis of KV cache tensors, we found:
- Layer 0 K: nearly rank-1 (mean + rank-13 residual)
- K is 3-10x more compressible than V across all layers
- Early layers allow more aggressive compression than late layers

This module provides configuration utilities that encode these findings.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class StagedCompressionConfig:
    """
    Configuration for staged compression training (v5).

    Staged compression trains at moderate ranks (stable), then gradually
    anneals to target aggressive ranks. This avoids optimization instability
    from training in a very tight bottleneck from the start.

    Attributes:
        r_k_max: Maximum K rank (parameter size)
        r_v_max: Maximum V rank (parameter size)
        r_k_start: Starting effective K rank (Stage A)
        r_v_start: Starting effective V rank (Stage A)
        r_k_target: Target K rank after annealing (Stage B end)
        r_v_target: Target V rank after annealing (Stage B end)
        anneal_start_frac: Fraction of training when annealing starts (default 0.7)
    """
    r_k_max: int
    r_v_max: int
    r_k_start: int
    r_v_start: int
    r_k_target: int
    r_v_target: int
    anneal_start_frac: float = 0.7

    def get_effective_ranks(self, step: int, total_steps: int) -> tuple[int, int]:
        """
        Get effective ranks for a given training step.

        Stage A (0 to anneal_start_frac): r_start (stable training)
        Stage B (anneal_start_frac to 1.0): linear anneal to r_target
        """
        progress = step / max(1, total_steps)

        if progress < self.anneal_start_frac:
            # Stage A: stable training at start ranks
            return self.r_k_start, self.r_v_start
        else:
            # Stage B: linear anneal to target
            anneal_progress = (progress - self.anneal_start_frac) / (1.0 - self.anneal_start_frac)
            anneal_progress = min(1.0, anneal_progress)

            r_k_eff = int(self.r_k_start + anneal_progress * (self.r_k_target - self.r_k_start))
            r_v_eff = int(self.r_v_start + anneal_progress * (self.r_v_target - self.r_v_start))

            return r_k_eff, r_v_eff

    def get_stage(self, step: int, total_steps: int) -> str:
        """Get current training stage name."""
        progress = step / max(1, total_steps)
        if progress < self.anneal_start_frac:
            return "A (stable)"
        else:
            return "B (annealing)"


@dataclass
class LayerLatentConfig:
    """Configuration for a single layer's latent dimensions."""

    r_k: int  # Latent dimension for Keys
    r_v: int  # Latent dimension for Values
    use_k_anchor: bool = False  # Whether to use an anchor vector for K
    use_v_anchor: bool = False  # Whether to use an anchor vector for V


@dataclass
class LatentKVConfig:
    """
    Full configuration for latent KV attention.

    Attributes:
        d_model: Model hidden dimension
        n_heads: Number of attention heads
        n_kv_heads: Number of KV heads (for GQA)
        d_head: Per-head dimension
        num_layers: Number of transformer layers
        layer_configs: Per-layer latent configurations
        rope_theta: RoPE theta parameter
        rope_scaling: RoPE scaling configuration
    """

    d_model: int
    n_heads: int
    n_kv_heads: int
    d_head: int
    num_layers: int
    layer_configs: list[LayerLatentConfig] = field(default_factory=list)
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None

    def __post_init__(self):
        if not self.layer_configs:
            self.layer_configs = self._generate_default_configs()

    def _generate_default_configs(self) -> list[LayerLatentConfig]:
        """Generate layer-adaptive configs based on empirical findings."""
        configs = []

        for layer_idx in range(self.num_layers):
            layer_frac = layer_idx / max(1, self.num_layers - 1)

            if layer_frac < 0.1:  # First 10% of layers (e.g., layers 0-2 in 24-layer)
                # Very aggressive K compression, moderate V
                # Based on finding: Layer 0 K is nearly rank-1
                r_k = max(4, self.d_head // 8)
                r_v = max(8, self.d_head // 3)
                use_k_anchor = True  # Exploit the "mean direction" structure

            elif layer_frac < 0.7:  # Middle 60% of layers
                # Moderate compression for both
                r_k = max(8, self.d_head // 4)
                r_v = max(16, self.d_head // 2)
                use_k_anchor = False

            else:  # Last 30% of layers
                # K still compressible, V needs more capacity
                r_k = max(8, self.d_head // 5)
                r_v = max(16, self.d_head // 2)
                use_k_anchor = False

            configs.append(
                LayerLatentConfig(
                    r_k=r_k,
                    r_v=r_v,
                    use_k_anchor=use_k_anchor,
                )
            )

        return configs

    def get_layer_config(self, layer_idx: int) -> LayerLatentConfig:
        """Get configuration for a specific layer."""
        return self.layer_configs[layer_idx]

    def total_latent_dims(self) -> tuple[int, int]:
        """Return total latent dimensions across all layers (for cache size estimation)."""
        total_k = sum(cfg.r_k for cfg in self.layer_configs)
        total_v = sum(cfg.r_v for cfg in self.layer_configs)
        return total_k, total_v

    def cache_size_reduction(self) -> float:
        """Calculate expected KV cache size reduction ratio."""
        original_per_layer = self.n_kv_heads * self.d_head * 2  # K + V
        original_total = original_per_layer * self.num_layers

        latent_total = sum(cfg.r_k + cfg.r_v for cfg in self.layer_configs)

        return 1.0 - (latent_total / original_total)

    def print_summary(self):
        """Print a summary of the configuration."""
        print(f"Latent KV Configuration Summary")
        print(f"=" * 50)
        print(f"Model dims: d_model={self.d_model}, d_head={self.d_head}")
        print(f"Attention: {self.n_heads} heads, {self.n_kv_heads} KV heads")
        print(f"Layers: {self.num_layers}")
        print()

        print(f"{'Layer':<8} {'r_k':<6} {'r_v':<6} {'K anchor':<10} {'K compress':<12} {'V compress':<12}")
        print("-" * 60)

        original_kv = self.n_kv_heads * self.d_head

        for idx, cfg in enumerate(self.layer_configs):
            k_compress = 1.0 - (cfg.r_k / original_kv)
            v_compress = 1.0 - (cfg.r_v / original_kv)
            print(
                f"{idx:<8} {cfg.r_k:<6} {cfg.r_v:<6} {str(cfg.use_k_anchor):<10} "
                f"{k_compress:>10.1%} {v_compress:>10.1%}"
            )

        print()
        total_k, total_v = self.total_latent_dims()
        reduction = self.cache_size_reduction()
        print(f"Total latent dims: K={total_k}, V={total_v}")
        print(f"Cache size reduction: {reduction:.1%}")


def get_default_config(
    model_name_or_config,
    compression_level: str = "moderate",
) -> LatentKVConfig:
    """
    Get default latent KV config for a model.

    Args:
        model_name_or_config: HuggingFace model name or config object
        compression_level: One of "aggressive", "moderate", "conservative"

    Returns:
        LatentKVConfig with appropriate settings
    """
    from transformers import AutoConfig

    if isinstance(model_name_or_config, str):
        hf_config = AutoConfig.from_pretrained(model_name_or_config)
    else:
        hf_config = model_name_or_config

    # Extract dimensions from HF config
    d_model = hf_config.hidden_size
    n_heads = hf_config.num_attention_heads
    n_kv_heads = getattr(hf_config, "num_key_value_heads", n_heads)
    d_head = d_model // n_heads
    num_layers = hf_config.num_hidden_layers

    # Create base config
    config = LatentKVConfig(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        d_head=d_head,
        num_layers=num_layers,
        rope_theta=getattr(hf_config, "rope_theta", 10000.0),
        rope_scaling=getattr(hf_config, "rope_scaling", None),
    )

    # Adjust based on compression level
    if compression_level == "aggressive":
        # Even more aggressive compression
        for cfg in config.layer_configs:
            cfg.r_k = max(2, cfg.r_k // 2)
            cfg.r_v = max(4, cfg.r_v * 2 // 3)
    elif compression_level == "conservative":
        # Less aggressive, prioritize quality
        for cfg in config.layer_configs:
            cfg.r_k = min(d_head, cfg.r_k * 2)
            cfg.r_v = min(d_head, cfg.r_v * 3 // 2)
            cfg.use_k_anchor = False

    return config


def config_from_empirical_ranks(
    model_config,
    empirical_ranks: dict[int, tuple[int, int]],
    energy_threshold: float = 0.90,
) -> LatentKVConfig:
    """
    Create config from empirically measured ranks.

    Args:
        model_config: HuggingFace model config
        empirical_ranks: Dict mapping layer_idx -> (r_k, r_v) at given energy threshold
        energy_threshold: The energy threshold these ranks were measured at

    Returns:
        LatentKVConfig with ranks set from empirical data
    """
    from transformers import AutoConfig

    if isinstance(model_config, str):
        hf_config = AutoConfig.from_pretrained(model_config)
    else:
        hf_config = model_config

    d_model = hf_config.hidden_size
    n_heads = hf_config.num_attention_heads
    n_kv_heads = getattr(hf_config, "num_key_value_heads", n_heads)
    d_head = d_model // n_heads
    num_layers = hf_config.num_hidden_layers

    # Create layer configs from empirical data
    layer_configs = []
    for layer_idx in range(num_layers):
        if layer_idx in empirical_ranks:
            r_k, r_v = empirical_ranks[layer_idx]
        else:
            # Interpolate for layers not measured
            # Find nearest measured layers
            measured = sorted(empirical_ranks.keys())
            if layer_idx < measured[0]:
                r_k, r_v = empirical_ranks[measured[0]]
            elif layer_idx > measured[-1]:
                r_k, r_v = empirical_ranks[measured[-1]]
            else:
                # Linear interpolation
                lower = max(m for m in measured if m < layer_idx)
                upper = min(m for m in measured if m > layer_idx)
                frac = (layer_idx - lower) / (upper - lower)
                r_k_low, r_v_low = empirical_ranks[lower]
                r_k_high, r_v_high = empirical_ranks[upper]
                r_k = int(r_k_low + frac * (r_k_high - r_k_low))
                r_v = int(r_v_low + frac * (r_v_high - r_v_low))

        # Use anchor for very early layers with low K rank
        use_k_anchor = layer_idx < 3 and r_k < d_head // 4

        layer_configs.append(
            LayerLatentConfig(r_k=r_k, r_v=r_v, use_k_anchor=use_k_anchor)
        )

    return LatentKVConfig(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        d_head=d_head,
        num_layers=num_layers,
        layer_configs=layer_configs,
        rope_theta=getattr(hf_config, "rope_theta", 10000.0),
        rope_scaling=getattr(hf_config, "rope_scaling", None),
    )
