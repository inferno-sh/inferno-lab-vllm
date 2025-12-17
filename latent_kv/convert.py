# SPDX-License-Identifier: Apache-2.0
"""
Model conversion utilities for Latent KV attention.

Converts standard transformer models to use learned low-rank KV projections.
Currently supports Qwen2 architecture.
"""

import copy
from typing import Optional

import torch
import torch.nn as nn

from latent_kv.attention import LatentKVAttention
from latent_kv.config import LatentKVConfig, LayerLatentConfig, get_default_config


def convert_qwen2_to_latent_kv(
    model: nn.Module,
    config: Optional[LatentKVConfig] = None,
    compression_level: str = "moderate",
    init_method: str = "svd",
    copy_model: bool = True,
) -> nn.Module:
    """
    Convert a Qwen2 model to use Latent KV attention.

    Args:
        model: Original Qwen2 model (e.g., from transformers)
        config: LatentKVConfig. If None, auto-generated from model config.
        compression_level: "aggressive", "moderate", or "conservative"
        init_method: "svd" for SVD-based init, "random" for random
        copy_model: If True, copy the model before modification

    Returns:
        Modified model with LatentKVAttention layers
    """
    if copy_model:
        model = copy.deepcopy(model)

    # Get model config
    hf_config = model.config

    # Generate latent config if not provided
    if config is None:
        config = get_default_config(hf_config, compression_level)

    # Find and replace attention modules
    _replace_attention_modules(model, config, init_method)

    # Store config on model for later reference
    model.latent_kv_config = config

    return model


def _replace_attention_modules(
    model: nn.Module,
    config: LatentKVConfig,
    init_method: str,
):
    """
    Recursively find and replace Qwen2Attention modules.
    """
    # Try to find the layers container
    # Qwen2 structure: model.model.layers[i].self_attn
    layers = None

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers

    if layers is None:
        raise ValueError(
            "Could not find transformer layers. Expected model.model.layers or model.layers"
        )

    print(f"Converting {len(layers)} attention layers to Latent KV...")

    for layer_idx, layer in enumerate(layers):
        if not hasattr(layer, "self_attn"):
            print(f"  Layer {layer_idx}: No self_attn found, skipping")
            continue

        original_attn = layer.self_attn
        layer_config = config.get_layer_config(layer_idx)

        # Create latent attention module
        latent_attn = LatentKVAttention.from_standard_attention(
            original_attn,
            layer_config,
            layer_idx=layer_idx,
            init_method=init_method,
        )

        # Replace the attention module
        layer.self_attn = latent_attn

        # Log compression stats
        original_kv_dim = config.n_kv_heads * config.d_head * 2
        latent_kv_dim = layer_config.r_k + layer_config.r_v
        compression = 1.0 - (latent_kv_dim / original_kv_dim)
        print(
            f"  Layer {layer_idx}: r_k={layer_config.r_k}, r_v={layer_config.r_v}, "
            f"anchor={layer_config.use_k_anchor}, compression={compression:.1%}"
        )

    print(f"\nTotal cache size reduction: {config.cache_size_reduction():.1%}")


def convert_attention_only(
    original_attn: nn.Module,
    layer_config: LayerLatentConfig,
    layer_idx: int = 0,
    init_method: str = "svd",
) -> LatentKVAttention:
    """
    Convert a single attention module (useful for testing).

    Args:
        original_attn: Original attention module
        layer_config: Latent configuration for this layer
        layer_idx: Layer index
        init_method: Initialization method

    Returns:
        LatentKVAttention module
    """
    return LatentKVAttention.from_standard_attention(
        original_attn,
        layer_config,
        layer_idx=layer_idx,
        init_method=init_method,
    )


class LatentKVModelWrapper(nn.Module):
    """
    Wrapper that converts a model to use latent KV caching.

    This wrapper intercepts the forward pass to use latent representations
    for KV caching instead of full K/V tensors.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[LatentKVConfig] = None,
        compression_level: str = "moderate",
    ):
        super().__init__()

        # Convert the model
        self.model = convert_qwen2_to_latent_kv(
            model,
            config=config,
            compression_level=compression_level,
            copy_model=True,
        )
        self.config = self.model.latent_kv_config

    def forward(self, *args, **kwargs):
        """Forward pass through the converted model."""
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """Generation with the converted model."""
        return self.model.generate(*args, **kwargs)

    @property
    def device(self):
        """Get model device."""
        return next(self.model.parameters()).device

    def get_cache_size_info(self, seq_len: int) -> dict:
        """
        Calculate cache size for a given sequence length.

        Args:
            seq_len: Sequence length

        Returns:
            Dict with original_size, latent_size, reduction
        """
        # Original: 2 (K+V) * n_layers * n_kv_heads * d_head * seq_len * dtype_size
        original_per_token = (
            2 * self.config.num_layers * self.config.n_kv_heads * self.config.d_head
        )

        # Latent: sum(r_k + r_v) * seq_len * dtype_size
        latent_per_token = sum(
            cfg.r_k + cfg.r_v for cfg in self.config.layer_configs
        )

        return {
            "original_floats_per_token": original_per_token,
            "latent_floats_per_token": latent_per_token,
            "original_total_floats": original_per_token * seq_len,
            "latent_total_floats": latent_per_token * seq_len,
            "reduction": 1.0 - (latent_per_token / original_per_token),
        }


def verify_conversion(
    original_model: nn.Module,
    converted_model: nn.Module,
    tokenizer,
    test_prompts: list[str] = None,
    atol: float = 0.1,
    rtol: float = 0.1,
) -> dict:
    """
    Verify conversion by comparing outputs.

    Note: With low-rank approximation, outputs won't match exactly.
    This checks that the converted model produces reasonable outputs.

    Args:
        original_model: Original model
        converted_model: Converted model
        tokenizer: Tokenizer
        test_prompts: Test prompts (default: simple prompts)
        atol: Absolute tolerance for comparison
        rtol: Relative tolerance for comparison

    Returns:
        Dict with verification results
    """
    if test_prompts is None:
        test_prompts = [
            "Hello, how are you?",
            "The capital of France is",
            "In the year 2024,",
        ]

    original_model.eval()
    converted_model.eval()

    results = {
        "prompts": [],
        "max_logit_diff": [],
        "mean_logit_diff": [],
        "top1_match": [],
    }

    device = next(original_model.parameters()).device

    with torch.no_grad():
        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # Get outputs
            orig_out = original_model(**inputs)
            conv_out = converted_model(**inputs)

            # Compare logits
            orig_logits = orig_out.logits
            conv_logits = conv_out.logits

            diff = (orig_logits - conv_logits).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()

            # Check top-1 prediction match
            orig_top1 = orig_logits[:, -1, :].argmax(dim=-1)
            conv_top1 = conv_logits[:, -1, :].argmax(dim=-1)
            top1_match = (orig_top1 == conv_top1).all().item()

            results["prompts"].append(prompt)
            results["max_logit_diff"].append(max_diff)
            results["mean_logit_diff"].append(mean_diff)
            results["top1_match"].append(top1_match)

    # Summary
    results["all_top1_match"] = all(results["top1_match"])
    results["avg_max_diff"] = sum(results["max_logit_diff"]) / len(results["max_logit_diff"])
    results["avg_mean_diff"] = sum(results["mean_logit_diff"]) / len(results["mean_logit_diff"])

    return results


def print_model_comparison(original_model: nn.Module, converted_model: nn.Module):
    """
    Print parameter count comparison between original and converted models.
    """
    def count_params(model):
        return sum(p.numel() for p in model.parameters())

    def count_trainable(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    orig_params = count_params(original_model)
    conv_params = count_params(converted_model)

    print("\nParameter Comparison:")
    print("=" * 50)
    print(f"Original model:  {orig_params:,} parameters")
    print(f"Converted model: {conv_params:,} parameters")
    print(f"Difference:      {conv_params - orig_params:,} ({(conv_params/orig_params - 1)*100:+.1f}%)")

    # Note: Low-rank factorization typically adds parameters
    # (d_model * r + r * d_kv) vs (d_model * d_kv)
    # But the KV cache size is reduced
