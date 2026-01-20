"""
Direct KV-cache capture with online novelty computation.

This module captures actual K/V tensors per-layer per-head and computes
novelty metrics online to avoid I/O explosion. Raw K/V snapshots are
only saved around detected spikes.

Design principles:
1. Compute novelty online - no full K/V dump
2. Log compact summaries per token
3. Snapshot raw K/V only around spikes (or 1/N sampling)
4. Include causal impact metric via K/V ablation
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


@dataclass
class HeadNovelty:
    """Novelty score for a single attention head."""
    layer_idx: int
    head_idx: int
    novelty: float  # Cosine distance from running mean


@dataclass
class TokenNoveltyRecord:
    """Compact per-token novelty summary."""
    position: int
    token_id: int
    token_text: str

    # Per-layer aggregated novelty (mean across heads)
    layer_novelty: dict[int, float] = field(default_factory=dict)

    # Per-layer top-k spiking heads
    layer_top_heads: dict[int, list[HeadNovelty]] = field(default_factory=dict)

    # Global statistics
    global_max_novelty: float = 0.0
    global_mean_novelty: float = 0.0

    # Causal impact (if computed)
    impact_delta_logprob: float | None = None
    impact_kl_divergence: float | None = None

    # Whether this is a detected spike
    is_spike: bool = False


@dataclass
class KVSnapshot:
    """Raw K/V snapshot around a spike for ground truth analysis."""
    spike_position: int
    spike_token: str
    spike_novelty: float

    # Window of K/V tensors [t-window..t+window]
    # Keys: (layer_idx, head_idx, relative_position)
    # Values: K or V tensor (head_dim,)
    keys: dict[tuple[int, int, int], np.ndarray] = field(default_factory=dict)
    values: dict[tuple[int, int, int], np.ndarray] = field(default_factory=dict)

    # Which layers/heads were captured
    captured_layers: list[int] = field(default_factory=list)
    captured_heads: dict[int, list[int]] = field(default_factory=dict)


class OnlineKVNoveltyTracker:
    """
    Tracks K/V novelty online with minimal memory footprint.

    For each (layer, head), maintains:
    - Running mean of K vectors (exponential moving average)
    - Recent K vectors for novelty computation (small window)

    Computes cosine novelty: 1 - cos_sim(new_k, running_mean)
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        window_size: int = 16,
        ema_alpha: float = 0.1,
        spike_threshold_std: float = 2.0,
        top_k_heads: int = 3,
        device: str = "cuda",
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.ema_alpha = ema_alpha
        self.spike_threshold_std = spike_threshold_std
        self.top_k_heads = top_k_heads
        self.device = device

        # Running mean for each (layer, head): [num_layers, num_heads, head_dim]
        self.k_running_mean = torch.zeros(
            num_layers, num_heads, head_dim, device=device
        )
        self.v_running_mean = torch.zeros(
            num_layers, num_heads, head_dim, device=device
        )

        # Count of updates (for initial mean computation)
        self.update_count = 0

        # Recent novelty scores for adaptive thresholding
        self.recent_novelty: deque[float] = deque(maxlen=100)

        # Small buffer for K/V around potential spikes
        # Only keeps last few tokens for snapshot capture
        self.kv_buffer: deque[dict] = deque(maxlen=5)

    def compute_novelty(
        self,
        k_new: torch.Tensor,  # [num_layers, num_heads, head_dim]
        v_new: torch.Tensor,  # [num_layers, num_heads, head_dim]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-head novelty for new K/V vectors.

        Returns:
            k_novelty: [num_layers, num_heads] cosine novelty scores
            v_novelty: [num_layers, num_heads] cosine novelty scores
        """
        if self.update_count == 0:
            # First token - no novelty
            k_novelty = torch.zeros(self.num_layers, self.num_heads, device=self.device)
            v_novelty = torch.zeros(self.num_layers, self.num_heads, device=self.device)
        else:
            # Cosine similarity with running mean
            # Normalize vectors
            k_new_norm = F.normalize(k_new, dim=-1)
            k_mean_norm = F.normalize(self.k_running_mean, dim=-1)
            v_new_norm = F.normalize(v_new, dim=-1)
            v_mean_norm = F.normalize(self.v_running_mean, dim=-1)

            # Cosine similarity: sum over head_dim
            k_cos_sim = (k_new_norm * k_mean_norm).sum(dim=-1)  # [num_layers, num_heads]
            v_cos_sim = (v_new_norm * v_mean_norm).sum(dim=-1)

            # Novelty = 1 - cos_sim (ranges from 0 to 2)
            k_novelty = 1.0 - k_cos_sim
            v_novelty = 1.0 - v_cos_sim

        # Update running mean with EMA
        if self.update_count == 0:
            self.k_running_mean = k_new.clone()
            self.v_running_mean = v_new.clone()
        else:
            self.k_running_mean = (
                self.ema_alpha * k_new +
                (1 - self.ema_alpha) * self.k_running_mean
            )
            self.v_running_mean = (
                self.ema_alpha * v_new +
                (1 - self.ema_alpha) * self.v_running_mean
            )

        self.update_count += 1

        return k_novelty, v_novelty

    def is_spike(self, novelty: float) -> bool:
        """Check if a novelty score is a spike using adaptive threshold."""
        if len(self.recent_novelty) < 10:
            return False  # Not enough data for threshold

        mean = np.mean(self.recent_novelty)
        std = np.std(self.recent_novelty)
        threshold = mean + self.spike_threshold_std * std

        return novelty > threshold

    def update_novelty_history(self, novelty: float):
        """Update running novelty statistics."""
        self.recent_novelty.append(novelty)

    def buffer_kv(self, position: int, k: torch.Tensor, v: torch.Tensor):
        """Buffer K/V for potential spike snapshot."""
        self.kv_buffer.append({
            "position": position,
            "k": k.detach().float().cpu().clone(),
            "v": v.detach().float().cpu().clone(),
        })

    def get_buffered_kv(self) -> list[dict]:
        """Get buffered K/V tensors."""
        return list(self.kv_buffer)

    def reset(self):
        """Reset all state."""
        self.k_running_mean.zero_()
        self.v_running_mean.zero_()
        self.update_count = 0
        self.recent_novelty.clear()
        self.kv_buffer.clear()


class KVCaptureHooks:
    """
    Hooks to capture actual K/V tensors from attention layers.

    Installs forward hooks that intercept K, V after projection and RoPE,
    computes novelty online, and optionally captures snapshots around spikes.
    """

    def __init__(
        self,
        model: nn.Module,
        tracker: OnlineKVNoveltyTracker,
        layers_to_capture: list[int] | None = None,
        snapshot_window: int = 2,
        snapshot_sample_rate: int = 50,  # 1/N sampling for non-spikes
    ):
        self.model = model
        self.tracker = tracker
        self.layers_to_capture = layers_to_capture
        self.snapshot_window = snapshot_window
        self.snapshot_sample_rate = snapshot_sample_rate

        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._is_active = False
        self._current_position = 0

        # Per-token results
        self._current_k: dict[int, torch.Tensor] = {}  # layer -> [num_heads, head_dim]
        self._current_v: dict[int, torch.Tensor] = {}

        # Results storage
        self.token_records: list[TokenNoveltyRecord] = []
        self.snapshots: list[KVSnapshot] = []

        # Model config
        self._num_layers = 0
        self._num_heads = 0
        self._head_dim = 0

    def _find_attention_layers(self) -> list[tuple[int, nn.Module]]:
        """Find attention modules in the model."""
        layers = []

        # Try different architectures
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            # Llama/Mistral/Qwen style
            for idx, layer in enumerate(self.model.model.layers):
                if hasattr(layer, "self_attn"):
                    layers.append((idx, layer.self_attn))
        elif hasattr(self.model, "transformer"):
            # GPT-2 style
            if hasattr(self.model.transformer, "h"):
                for idx, layer in enumerate(self.model.transformer.h):
                    if hasattr(layer, "attn"):
                        layers.append((idx, layer.attn))
        elif hasattr(self.model, "gpt_neox"):
            # GPT-NeoX style
            for idx, layer in enumerate(self.model.gpt_neox.layers):
                if hasattr(layer, "attention"):
                    layers.append((idx, layer.attention))

        return layers

    def _create_hook(self, layer_idx: int, attn_module: nn.Module) -> Callable:
        """Create a forward hook to capture K/V after computation."""

        def hook(module: nn.Module, inputs: tuple, outputs: Any) -> None:
            if not self._is_active:
                return

            # Try to extract K, V from the attention module
            # This depends on the specific architecture

            # For models with cache, the past_key_value output contains K, V
            if isinstance(outputs, tuple) and len(outputs) >= 2:
                # outputs typically: (attn_output, present_key_value, ...)
                # present_key_value is often (key, value) tuple
                if outputs[1] is not None:
                    past_kv = outputs[1]
                    if isinstance(past_kv, tuple) and len(past_kv) == 2:
                        key, value = past_kv
                        # key, value shape: [batch, num_heads, seq_len, head_dim]

                        # Get the last token's K/V
                        k_last = key[0, :, -1, :].detach()  # [num_heads, head_dim]
                        v_last = value[0, :, -1, :].detach()

                        self._current_k[layer_idx] = k_last
                        self._current_v[layer_idx] = v_last

        return hook

    def install(self) -> int:
        """Install hooks on attention layers."""
        self.remove()

        layers = self._find_attention_layers()
        if not layers:
            raise RuntimeError("Could not find attention layers in model")

        self._num_layers = len(layers)

        # Get head config from first attention module
        first_attn = layers[0][1]
        if hasattr(first_attn, "num_heads"):
            self._num_heads = first_attn.num_heads
        elif hasattr(first_attn, "num_attention_heads"):
            self._num_heads = first_attn.num_attention_heads
        else:
            self._num_heads = getattr(self.model.config, "num_attention_heads", 32)

        self._head_dim = getattr(self.model.config, "head_dim",
                                  self.model.config.hidden_size // self._num_heads)

        num_hooks = 0
        for layer_idx, attn_module in layers:
            if self.layers_to_capture is not None:
                if layer_idx not in self.layers_to_capture:
                    continue

            hook = self._create_hook(layer_idx, attn_module)
            handle = attn_module.register_forward_hook(hook)
            self._hooks.append(handle)
            num_hooks += 1

        self._is_active = True
        return num_hooks

    def remove(self):
        """Remove all hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        self._is_active = False

    def process_token(
        self,
        token_id: int,
        token_text: str,
        tokenizer: Any = None,
    ) -> TokenNoveltyRecord:
        """
        Process captured K/V for the current token.

        Computes novelty, logs summary, and potentially captures snapshot.
        """
        if not self._current_k:
            # No K/V captured - return empty record
            return TokenNoveltyRecord(
                position=self._current_position,
                token_id=token_id,
                token_text=token_text,
            )

        # Stack K/V across layers
        layers = sorted(self._current_k.keys())
        k_stack = torch.stack([self._current_k[l] for l in layers])  # [num_layers, num_heads, head_dim]
        v_stack = torch.stack([self._current_v[l] for l in layers])

        # Compute novelty
        k_novelty, v_novelty = self.tracker.compute_novelty(k_stack, v_stack)

        # Aggregate: use max of K and V novelty
        combined_novelty = torch.maximum(k_novelty, v_novelty)

        # Per-layer stats
        layer_novelty = {}
        layer_top_heads = {}

        for i, layer_idx in enumerate(layers):
            layer_nov = combined_novelty[i]

            # Mean across heads
            layer_novelty[layer_idx] = layer_nov.mean().item()

            # Top-k heads
            top_k = min(self.tracker.top_k_heads, self._num_heads)
            top_vals, top_idxs = torch.topk(layer_nov, top_k)

            layer_top_heads[layer_idx] = [
                HeadNovelty(
                    layer_idx=layer_idx,
                    head_idx=idx.item(),
                    novelty=val.item(),
                )
                for val, idx in zip(top_vals, top_idxs)
            ]

        # Global stats
        global_max = combined_novelty.max().item()
        global_mean = combined_novelty.mean().item()

        # Check if spike
        is_spike = self.tracker.is_spike(global_max)
        self.tracker.update_novelty_history(global_max)

        # Buffer K/V for potential snapshot
        self.tracker.buffer_kv(self._current_position, k_stack, v_stack)

        # Create record
        record = TokenNoveltyRecord(
            position=self._current_position,
            token_id=token_id,
            token_text=token_text,
            layer_novelty=layer_novelty,
            layer_top_heads=layer_top_heads,
            global_max_novelty=global_max,
            global_mean_novelty=global_mean,
            is_spike=is_spike,
        )

        self.token_records.append(record)

        # Capture snapshot if spike (or sampled)
        if is_spike or (self._current_position % self.snapshot_sample_rate == 0):
            self._capture_snapshot(record)

        # Clear current K/V
        self._current_k.clear()
        self._current_v.clear()
        self._current_position += 1

        return record

    def _capture_snapshot(self, record: TokenNoveltyRecord):
        """Capture raw K/V snapshot around a spike."""
        buffered = self.tracker.get_buffered_kv()
        if not buffered:
            return

        snapshot = KVSnapshot(
            spike_position=record.position,
            spike_token=record.token_text,
            spike_novelty=record.global_max_novelty,
        )

        # Determine which layers/heads to capture (top spiking ones)
        top_layers = sorted(
            record.layer_novelty.keys(),
            key=lambda l: record.layer_novelty[l],
            reverse=True,
        )[:3]  # Top 3 layers

        for buf in buffered:
            pos = buf["position"]
            k = buf["k"]  # [num_layers, num_heads, head_dim]
            v = buf["v"]

            rel_pos = pos - record.position

            for layer_idx in top_layers:
                if layer_idx >= k.shape[0]:
                    continue

                # Get top heads for this layer
                if layer_idx in record.layer_top_heads:
                    top_heads = [h.head_idx for h in record.layer_top_heads[layer_idx]]
                else:
                    top_heads = list(range(min(3, k.shape[1])))

                for head_idx in top_heads:
                    if head_idx >= k.shape[1]:
                        continue

                    key = (layer_idx, head_idx, rel_pos)
                    snapshot.keys[key] = k[layer_idx, head_idx].numpy()
                    snapshot.values[key] = v[layer_idx, head_idx].numpy()

        snapshot.captured_layers = top_layers
        snapshot.captured_heads = {
            l: [h.head_idx for h in record.layer_top_heads.get(l, [])]
            for l in top_layers
        }

        self.snapshots.append(snapshot)

    def reset(self):
        """Reset for new generation."""
        self._current_position = 0
        self._current_k.clear()
        self._current_v.clear()
        self.token_records.clear()
        self.snapshots.clear()
        self.tracker.reset()


def compute_kv_ablation_impact(
    model: nn.Module,
    tokenizer: Any,
    input_ids: torch.Tensor,
    target_position: int,
    target_layer: int,
    target_heads: list[int],
    ablation_type: str = "zero",  # "zero", "mean", "random"
    device: str = "cuda",
) -> dict[str, float]:
    """
    Compute causal impact of K/V at a specific position by ablation.

    Measures how much the next-token distribution changes when we
    ablate (zero out, replace with mean, etc.) the K/V at target position.

    Args:
        model: The model
        tokenizer: Tokenizer
        input_ids: Input token IDs up to and including target position
        target_position: Position in KV cache to ablate
        target_layer: Layer to ablate
        target_heads: Heads to ablate
        ablation_type: Type of ablation
        device: Device

    Returns:
        Dictionary with impact metrics:
        - delta_logprob: Change in log probability of actual next token
        - kl_divergence: KL divergence between original and ablated distributions
        - max_prob_change: Maximum change in any token's probability
    """
    # Import DynamicCache for newer transformers versions
    try:
        from transformers.cache_utils import DynamicCache
        has_dynamic_cache = True
    except ImportError:
        has_dynamic_cache = False

    model.eval()

    with torch.no_grad():
        # Forward pass to get original next-token distribution
        outputs_original = model(
            input_ids.to(device),
            use_cache=True,
            return_dict=True,
        )

        logits_original = outputs_original.logits[0, -1]  # [vocab_size]
        probs_original = F.softmax(logits_original, dim=-1)
        log_probs_original = F.log_softmax(logits_original, dim=-1)

        # Get the KV cache
        past_kv = outputs_original.past_key_values

        if past_kv is None:
            return {
                "delta_logprob": 0.0,
                "kl_divergence": 0.0,
                "max_prob_change": 0.0,
            }

        # Check if past_kv is a DynamicCache or similar object
        is_cache_object = hasattr(past_kv, 'get_seq_length')

        if is_cache_object:
            # Modern transformers with DynamicCache
            # Convert to legacy format, modify, convert back
            legacy_cache = past_kv.to_legacy_cache()

            modified_legacy = []
            for layer_idx, (key, value) in enumerate(legacy_cache):
                if layer_idx == target_layer:
                    key = key.clone()
                    value = value.clone()

                    for head_idx in target_heads:
                        if head_idx < key.shape[1]:  # Check head exists
                            if ablation_type == "zero":
                                key[0, head_idx, target_position, :] = 0
                                value[0, head_idx, target_position, :] = 0
                            elif ablation_type == "mean":
                                key[0, head_idx, target_position, :] = key[0, head_idx, :, :].mean(dim=0)
                                value[0, head_idx, target_position, :] = value[0, head_idx, :, :].mean(dim=0)
                            elif ablation_type == "random":
                                key[0, head_idx, target_position, :] = torch.randn_like(
                                    key[0, head_idx, target_position, :]
                                )
                                value[0, head_idx, target_position, :] = torch.randn_like(
                                    value[0, head_idx, target_position, :]
                                )

                modified_legacy.append((key, value))

            # Convert back to DynamicCache
            modified_past_kv = DynamicCache.from_legacy_cache(tuple(modified_legacy))
        else:
            # Legacy tuple format
            modified_past_kv = []
            for layer_idx, (key, value) in enumerate(past_kv):
                if layer_idx == target_layer:
                    key = key.clone()
                    value = value.clone()

                    for head_idx in target_heads:
                        if ablation_type == "zero":
                            key[0, head_idx, target_position, :] = 0
                            value[0, head_idx, target_position, :] = 0
                        elif ablation_type == "mean":
                            key[0, head_idx, target_position, :] = key[0, head_idx, :, :].mean(dim=0)
                            value[0, head_idx, target_position, :] = value[0, head_idx, :, :].mean(dim=0)
                        elif ablation_type == "random":
                            key[0, head_idx, target_position, :] = torch.randn_like(
                                key[0, head_idx, target_position, :]
                            )
                            value[0, head_idx, target_position, :] = torch.randn_like(
                                value[0, head_idx, target_position, :]
                            )

                modified_past_kv.append((key, value))

            modified_past_kv = tuple(modified_past_kv)

        # Forward pass with modified cache
        # We need to run just the last token with the modified cache
        last_token = input_ids[:, -1:].to(device)

        outputs_ablated = model(
            last_token,
            past_key_values=modified_past_kv,
            use_cache=False,
            return_dict=True,
        )

        logits_ablated = outputs_ablated.logits[0, -1]
        probs_ablated = F.softmax(logits_ablated, dim=-1)
        log_probs_ablated = F.log_softmax(logits_ablated, dim=-1)

        # Compute impact metrics

        # KL divergence: KL(original || ablated)
        kl_div = F.kl_div(
            log_probs_ablated,
            probs_original,
            reduction="sum",
        ).item()

        # Delta log prob for most likely token
        top_token = probs_original.argmax()
        delta_logprob = (log_probs_original[top_token] - log_probs_ablated[top_token]).item()

        # Max probability change
        prob_change = (probs_original - probs_ablated).abs()
        max_prob_change = prob_change.max().item()

        return {
            "delta_logprob": delta_logprob,
            "kl_divergence": kl_div,
            "max_prob_change": max_prob_change,
        }
