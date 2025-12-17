"""
kvchan: dynamic KV cache compression prototype for vLLM/HF.

This package implements a channel-selection-based KV cache with a quality-first
control loop. See README in kvchan/ for details.
"""

__all__ = [
    "prompts",
    "importance",
    "control",
    "kvcache_full",
    "kvcache_packed",
    "fidelity",
    "hf_backend",
    "vllm_backend",
    "cli",
]
