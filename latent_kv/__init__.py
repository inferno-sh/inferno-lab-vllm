# SPDX-License-Identifier: Apache-2.0
"""
Latent KV Heads: Empirically-guided KV cache compression via learned low-rank projections.

This package provides tools to convert standard transformer attention to use
learned low-rank KV projections, enabling significant KV cache size reduction
while maintaining model quality through distillation.

Key components:
- LatentKVAttention: Drop-in replacement attention module with low-rank K/V
- LatentKVConfig: Layer-adaptive configuration based on empirical rank analysis
- convert_model: Convert a standard model to use latent KV attention
- LatentKVDistillationTrainer: Distillation training from full-rank teacher
"""

from latent_kv.attention import LatentKVAttention
from latent_kv.config import LatentKVConfig, get_default_config
from latent_kv.convert import convert_qwen2_to_latent_kv

__all__ = [
    "LatentKVAttention",
    "LatentKVConfig",
    "get_default_config",
    "convert_qwen2_to_latent_kv",
]

__version__ = "0.1.0"
