"""
PyTorch hooks for capturing KV tensors during vLLM inference.

This module provides hooks that can be installed on vLLM's attention layers
to capture key and value tensors as they are computed.
"""

from __future__ import annotations

import re
from typing import Any, Callable

import torch
import torch.nn as nn

from .collector import KVNoveltyCollector


# Global registry of installed hooks
_installed_hooks: dict[str, torch.utils.hooks.RemovableHandle] = {}
_collector: KVNoveltyCollector | None = None


def get_collector() -> KVNoveltyCollector | None:
    """Get the currently active collector."""
    return _collector


def set_collector(collector: KVNoveltyCollector | None) -> None:
    """Set the active collector."""
    global _collector
    _collector = collector


def _extract_layer_index(name: str) -> int | None:
    """Extract layer index from a module name like 'model.layers.5.self_attn'."""
    # Try various patterns
    patterns = [
        r"layers\.(\d+)",
        r"layer_(\d+)",
        r"h\.(\d+)",
        r"block_(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            return int(match.group(1))
    return None


def _attention_forward_hook(
    module: nn.Module,
    inputs: tuple[Any, ...],
    output: Any,
    layer_name: str,
) -> None:
    """
    Forward hook for Attention layers.

    This captures the K, V tensors after they've been processed but before
    they're used for attention computation.

    The inputs to Attention.forward() are: (query, key, value, output_shape?)
    """
    collector = get_collector()
    if collector is None:
        return

    layer_idx = _extract_layer_index(layer_name)
    if layer_idx is None:
        return

    if not collector.should_capture_layer(layer_idx):
        return

    # Unpack inputs
    # Attention.forward(query, key, value, output_shape=None)
    if len(inputs) >= 3:
        query, key, value = inputs[0], inputs[1], inputs[2]

        if key is not None and value is not None:
            collector.record_kv(
                layer_idx=layer_idx,
                key=key,
                value=value,
            )


def _attention_pre_forward_hook(
    module: nn.Module,
    inputs: tuple[Any, ...],
    layer_name: str,
) -> None:
    """
    Pre-forward hook for Attention layers (captures inputs before forward).
    """
    collector = get_collector()
    if collector is None:
        return

    layer_idx = _extract_layer_index(layer_name)
    if layer_idx is None:
        return

    if not collector.should_capture_layer(layer_idx):
        return

    # Unpack inputs - Attention.forward(query, key, value, ...)
    if len(inputs) >= 3:
        query, key, value = inputs[0], inputs[1], inputs[2]

        if key is not None and value is not None:
            collector.record_kv(
                layer_idx=layer_idx,
                key=key,
                value=value,
            )


def install_kv_hooks(
    model: nn.Module,
    collector: KVNoveltyCollector,
    use_pre_hook: bool = True,
) -> int:
    """
    Install hooks on attention layers to capture KV tensors.

    Args:
        model: The model to instrument (should be the vLLM model)
        collector: The collector to store captured data
        use_pre_hook: Use pre-forward hook (True) or post-forward hook (False)

    Returns:
        Number of hooks installed
    """
    global _installed_hooks

    # Remove any existing hooks first
    remove_kv_hooks()

    # Set the collector
    set_collector(collector)

    num_hooks = 0

    # Find attention layers
    for name, module in model.named_modules():
        # Look for Attention class from vLLM
        module_class_name = module.__class__.__name__

        # Match vLLM's Attention layer or model-specific attention layers
        if module_class_name in ("Attention", "LlamaAttention", "MistralAttention",
                                  "Qwen2Attention", "GemmaAttention"):
            # For model-specific attention layers (e.g., LlamaAttention),
            # we want to hook the inner self.attn (Attention) layer
            if hasattr(module, "attn") and module_class_name != "Attention":
                target = module.attn
                hook_name = f"{name}.attn"
            else:
                target = module
                hook_name = name

            # Create the hook
            if use_pre_hook:
                handle = target.register_forward_pre_hook(
                    lambda mod, inp, n=hook_name: _attention_pre_forward_hook(mod, inp, n)
                )
            else:
                handle = target.register_forward_hook(
                    lambda mod, inp, out, n=hook_name: _attention_forward_hook(mod, inp, out, n)
                )

            _installed_hooks[hook_name] = handle
            num_hooks += 1

    return num_hooks


def remove_kv_hooks() -> int:
    """
    Remove all installed KV hooks.

    Returns:
        Number of hooks removed
    """
    global _installed_hooks

    num_removed = 0
    for name, handle in _installed_hooks.items():
        handle.remove()
        num_removed += 1

    _installed_hooks.clear()
    set_collector(None)

    return num_removed


class KVCaptureContext:
    """
    Context manager for capturing KV data during generation.

    Usage:
        collector = KVNoveltyCollector()
        with KVCaptureContext(model, collector):
            # Generate tokens...
            pass
        # Analyze collector.kv_data
    """

    def __init__(
        self,
        model: nn.Module,
        collector: KVNoveltyCollector,
        model_name: str = "",
        prompt: str = "",
        generation_config: dict[str, Any] | None = None,
    ):
        self.model = model
        self.collector = collector
        self.model_name = model_name
        self.prompt = prompt
        self.generation_config = generation_config or {}
        self._num_hooks = 0

    def __enter__(self) -> KVNoveltyCollector:
        self.collector.start_collection(
            model_name=self.model_name,
            prompt=self.prompt,
            generation_config=self.generation_config,
        )
        self._num_hooks = install_kv_hooks(self.model, self.collector)
        return self.collector

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.collector.stop_collection()
        remove_kv_hooks()
        return False


# Alternative approach: Monkey-patching for when hooks don't work
class KVCapturePatch:
    """
    Alternative to hooks: patches the Attention.forward method directly.

    This may be necessary when hooks interfere with compiled models or
    custom backends.
    """

    def __init__(self, collector: KVNoveltyCollector):
        self.collector = collector
        self._original_forwards: dict[int, Callable] = {}

    def patch(self, model: nn.Module) -> int:
        """Patch all Attention layers."""
        set_collector(self.collector)
        num_patched = 0

        for name, module in model.named_modules():
            if module.__class__.__name__ == "Attention":
                layer_idx = _extract_layer_index(name)
                if layer_idx is not None:
                    original = module.forward
                    self._original_forwards[id(module)] = original

                    def make_patched_forward(orig, idx):
                        def patched_forward(query, key, value, output_shape=None):
                            collector = get_collector()
                            if collector is not None and key is not None:
                                collector.record_kv(
                                    layer_idx=idx,
                                    key=key,
                                    value=value,
                                )
                            return orig(query, key, value, output_shape)
                        return patched_forward

                    module.forward = make_patched_forward(original, layer_idx)
                    num_patched += 1

        return num_patched

    def unpatch(self, model: nn.Module) -> int:
        """Restore original forward methods."""
        num_unpatched = 0

        for name, module in model.named_modules():
            if module.__class__.__name__ == "Attention":
                if id(module) in self._original_forwards:
                    module.forward = self._original_forwards[id(module)]
                    num_unpatched += 1

        self._original_forwards.clear()
        set_collector(None)
        return num_unpatched
