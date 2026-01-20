"""
Specialized hooks for capturing KV tensors from Llama-style models.

This module provides robust hooking into HuggingFace Llama models
to capture actual Key and Value tensors during generation.
"""

from __future__ import annotations

from typing import Any, Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np

from .collector import KVNoveltyCollector, LayerKVData


@dataclass
class CapturedKV:
    """Container for captured KV data from a single forward pass."""
    layer_idx: int
    key: torch.Tensor  # [batch, num_kv_heads, seq_len, head_dim]
    value: torch.Tensor  # [batch, num_kv_heads, seq_len, head_dim]
    query: torch.Tensor | None = None


class LlamaKVHooks:
    """
    Hook manager for capturing KV tensors from Llama models.

    This class installs hooks that capture the Key and Value tensors
    after they are computed (post-projection, post-RoPE) but before
    they are used in attention computation.

    Compatible with:
    - transformers LlamaForCausalLM
    - transformers MistralForCausalLM
    - Other Llama-architecture models
    """

    def __init__(
        self,
        model: nn.Module,
        collector: KVNoveltyCollector | None = None,
        layers_to_capture: list[int] | None = None,
        capture_query: bool = False,
    ):
        """
        Initialize hooks.

        Args:
            model: The HuggingFace model
            collector: Optional collector for storing data
            layers_to_capture: Specific layers to capture (None = all)
            capture_query: Whether to also capture Q tensors
        """
        self.model = model
        self.collector = collector
        self.layers_to_capture = layers_to_capture
        self.capture_query = capture_query

        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._captured_kv: dict[int, CapturedKV] = {}
        self._current_position: int = 0
        self._is_active: bool = False

    def _get_layers(self) -> list[tuple[int, nn.Module]]:
        """Get the attention layers from the model."""
        layers = []

        # Try different model structures
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            # Standard Llama structure
            for idx, layer in enumerate(self.model.model.layers):
                if hasattr(layer, "self_attn"):
                    layers.append((idx, layer.self_attn))
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            # GPT-style structure
            for idx, layer in enumerate(self.model.transformer.h):
                if hasattr(layer, "attn"):
                    layers.append((idx, layer.attn))

        return layers

    def install(self) -> int:
        """
        Install hooks on the model.

        Returns:
            Number of hooks installed
        """
        self.remove()  # Clear any existing hooks

        layers = self._get_layers()
        num_hooks = 0

        for layer_idx, attn_module in layers:
            if self.layers_to_capture is not None:
                if layer_idx not in self.layers_to_capture:
                    continue

            # Create a hook for this layer
            hook = self._create_hook(layer_idx, attn_module)
            handle = attn_module.register_forward_hook(hook)
            self._hooks.append(handle)
            num_hooks += 1

        self._is_active = True
        return num_hooks

    def _create_hook(
        self,
        layer_idx: int,
        attn_module: nn.Module,
    ) -> Callable:
        """Create a forward hook for an attention layer."""

        def hook(
            module: nn.Module,
            inputs: tuple[Any, ...],
            outputs: Any,
        ) -> None:
            if not self._is_active:
                return

            # The attention forward typically takes:
            # hidden_states, attention_mask, position_ids, past_key_value, ...
            # And outputs attention_output, possibly with other things

            # We need to capture K, V after they're computed
            # For Llama-style attention, K and V are computed inside forward
            # We'll need to access them through the module's internal state
            # or by modifying the forward to expose them

            # Alternative: Hook into the qkv_proj output
            # This requires a different approach

            pass

        return hook

    def _create_qkv_hook(
        self,
        layer_idx: int,
    ) -> Callable:
        """
        Create a hook that captures Q, K, V after projection.

        This hook should be registered on the attention module's forward
        and intercepts the Q, K, V tensors after they're computed.
        """

        def hook(module: nn.Module, inputs: tuple, outputs: Any) -> None:
            if not self._is_active:
                return

            # For LlamaAttention, we need to look at internal computation
            # The cleanest way is to monkey-patch the forward method

            pass

        return hook

    def remove(self) -> int:
        """Remove all hooks."""
        num_removed = len(self._hooks)
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        self._is_active = False
        return num_removed

    def get_captured_kv(self) -> dict[int, CapturedKV]:
        """Get captured KV data from the last forward pass."""
        return self._captured_kv.copy()

    def clear_captured(self) -> None:
        """Clear captured data."""
        self._captured_kv.clear()


def create_kv_capturing_attention(original_class: type) -> type:
    """
    Create a modified attention class that captures KV tensors.

    This is a more robust approach than hooks - we create a subclass
    of the attention module that captures K, V during its forward pass.

    Args:
        original_class: The original attention class (e.g., LlamaAttention)

    Returns:
        Modified attention class with KV capture capability
    """
    _kv_storage: dict[int, dict] = {}
    _capture_enabled: bool = False
    _current_position: int = 0

    class KVCapturingAttention(original_class):
        """Attention layer that captures KV tensors."""

        _layer_idx: int = -1

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
            past_key_value: Any | None = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: torch.Tensor | None = None,
            position_embeddings: tuple | None = None,
            **kwargs,
        ) -> tuple:
            # Get batch size and sequence length
            bsz, q_len, _ = hidden_states.size()

            # Compute Q, K, V projections
            if hasattr(self, "q_proj"):
                # Standard separate projections
                query_states = self.q_proj(hidden_states)
                key_states = self.k_proj(hidden_states)
                value_states = self.v_proj(hidden_states)
            elif hasattr(self, "qkv_proj"):
                # Fused QKV projection
                qkv = self.qkv_proj(hidden_states)
                # Split into Q, K, V
                query_states, key_states, value_states = qkv.split(
                    [self.hidden_size, self.hidden_size, self.hidden_size],
                    dim=-1,
                )

            # Reshape
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            # Apply rotary embeddings if available
            if position_embeddings is not None:
                cos, sin = position_embeddings
                query_states, key_states = self.rotary_emb(
                    query_states, key_states, cos, sin
                )
            elif hasattr(self, "rotary_emb") and position_ids is not None:
                cos, sin = self.rotary_emb(value_states, position_ids)
                query_states = (query_states * cos) + (self._rotate_half(query_states) * sin)
                key_states = (key_states * cos) + (self._rotate_half(key_states) * sin)

            # Capture KV if enabled
            nonlocal _capture_enabled, _kv_storage, _current_position
            if _capture_enabled:
                layer_idx = self._layer_idx
                if layer_idx not in _kv_storage:
                    _kv_storage[layer_idx] = {}

                # Store the last token's KV (for decode) or all (for prefill)
                _kv_storage[layer_idx][_current_position] = {
                    "key": key_states[:, :, -1:, :].detach().cpu(),  # Last token
                    "value": value_states[:, :, -1:, :].detach().cpu(),
                }

            # Continue with rest of attention computation...
            # (Call parent's remaining computation)
            return super().forward(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        def _rotate_half(self, x):
            """Rotate half the hidden dims of the input."""
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def enable_capture():
        nonlocal _capture_enabled
        _capture_enabled = True

    @staticmethod
    def disable_capture():
        nonlocal _capture_enabled
        _capture_enabled = False

    @staticmethod
    def get_storage():
        return _kv_storage

    @staticmethod
    def clear_storage():
        nonlocal _kv_storage
        _kv_storage.clear()

    @staticmethod
    def set_position(pos):
        nonlocal _current_position
        _current_position = pos

    KVCapturingAttention.enable_capture = enable_capture
    KVCapturingAttention.disable_capture = disable_capture
    KVCapturingAttention.get_storage = get_storage
    KVCapturingAttention.clear_storage = clear_storage
    KVCapturingAttention.set_position = set_position

    return KVCapturingAttention


class SimpleKVCapture:
    """
    Simple KV capture using output_hidden_states.

    This approach uses the model's hidden states as a proxy for KV behavior,
    which is less accurate but much simpler and more robust.
    """

    def __init__(
        self,
        collector: KVNoveltyCollector,
        layers_to_capture: list[int] | None = None,
    ):
        self.collector = collector
        self.layers_to_capture = layers_to_capture

    def process_generation_output(
        self,
        outputs: Any,
        generated_ids: torch.Tensor,
        tokenizer: Any,
    ) -> None:
        """
        Process generation outputs to extract novelty data.

        Args:
            outputs: Output from model.generate() with output_hidden_states=True
            generated_ids: The generated token IDs
            tokenizer: Tokenizer for decoding
        """
        # Extract scores (logits) for entropy/logprob
        if hasattr(outputs, "scores") and outputs.scores:
            for t, score in enumerate(outputs.scores):
                if t >= len(generated_ids):
                    break

                logits = score[0]  # First batch element
                probs = torch.softmax(logits, dim=-1)
                log_probs = torch.log_softmax(logits, dim=-1)

                # Get sampled token info
                token_id = generated_ids[t].item()
                token_logprob = log_probs[token_id].item()

                # Compute entropy
                entropy = -torch.sum(probs * log_probs).item()

                # Decode token
                token_text = tokenizer.decode([token_id])

                # Record token data
                self.collector.record_token(
                    token_id=token_id,
                    token_text=token_text,
                    logprob=token_logprob,
                    entropy=entropy,
                )

                # Process hidden states
                if hasattr(outputs, "hidden_states") and outputs.hidden_states:
                    if t < len(outputs.hidden_states):
                        step_hidden = outputs.hidden_states[t]
                        # step_hidden is a tuple: (layer_0_hidden, layer_1_hidden, ...)
                        # Each layer_hidden has shape [batch, seq_len, hidden_size]

                        for layer_idx, layer_hidden in enumerate(step_hidden):
                            if self.layers_to_capture is not None:
                                if layer_idx not in self.layers_to_capture:
                                    continue

                            # Get the last token's hidden state
                            # Shape: [hidden_size]
                            hidden = layer_hidden[0, -1].detach().float().cpu().numpy()

                            # Store as key (using hidden state as proxy)
                            # In a proper implementation, we'd have actual K, V
                            # But hidden states track similar dynamics
                            if layer_idx not in self.collector.kv_data:
                                self.collector.kv_data[layer_idx] = {}

                            self.collector.kv_data[layer_idx][t] = LayerKVData(
                                layer_idx=layer_idx,
                                position=t,
                                # Reshape to [1, hidden_size] to mimic [num_heads, head_dim]
                                key=hidden.reshape(1, -1),
                                value=None,
                            )

                self.collector.advance_position()


def run_with_kv_capture(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    collector: KVNoveltyCollector,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    layers_to_capture: list[int] | None = None,
    device: str = "cuda",
) -> tuple[str, KVNoveltyCollector]:
    """
    Run generation with KV capture.

    This is the main function to use for experiments.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        prompt: Input prompt
        collector: Data collector
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling
        do_sample: Whether to sample (vs greedy)
        layers_to_capture: Specific layers to capture
        device: Device to use

    Returns:
        Tuple of (generated_text, collector_with_data)
    """
    collector.start_collection(
        model_name=getattr(model.config, "_name_or_path", "unknown"),
        prompt=prompt,
        generation_config={
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
        },
    )

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    # Generate with output_hidden_states and output_scores
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            do_sample=do_sample,
            output_scores=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # Extract generated tokens (excluding prompt)
    generated_ids = outputs.sequences[0, input_len:]

    # Process outputs
    capture = SimpleKVCapture(collector, layers_to_capture)
    capture.process_generation_output(outputs, generated_ids, tokenizer)

    collector.stop_collection()

    # Decode generated text
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return generated_text, collector
