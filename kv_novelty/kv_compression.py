"""
KV-cache compression experiments based on novelty.

Tests the hypothesis that low-novelty KV entries can be dropped/compressed
without significant quality degradation.

Compression strategies:
1. Drop low-novelty K entries (keep V) - "K-only compression"
2. Drop both K and V for low-novelty entries - "Full compression"
3. Preserve window around spikes - "Conservative compression"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


@dataclass
class CompressionConfig:
    """Configuration for KV compression experiment."""
    # Novelty threshold (percentile) - entries below this are "low novelty"
    novelty_percentile: float = 50.0  # Bottom 50% are low-novelty

    # Compression strategy
    strategy: str = "k_only"  # "k_only", "full", "conservative"

    # For conservative strategy: preserve N tokens around each spike
    spike_window: int = 2

    # What to do with dropped entries
    drop_method: str = "zero"  # "zero", "mean", "first"

    # Which layers to compress (None = all)
    layers_to_compress: list[int] | None = None

    # Which heads to compress (None = all)
    heads_to_compress: list[int] | None = None


@dataclass
class CompressionResult:
    """Results from a compression experiment."""
    config: dict

    # Compression stats
    total_entries: int
    compressed_entries: int
    compression_ratio: float

    # Quality metrics
    original_loss: float
    compressed_loss: float
    loss_increase: float
    loss_increase_pct: float

    # Per-token metrics
    mean_kl_divergence: float
    max_kl_divergence: float

    # Tokens where compression hurt most
    worst_affected_tokens: list[dict] = field(default_factory=list)


class NoveltyTracker:
    """
    Tracks novelty scores during generation for compression decisions.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        ema_alpha: float = 0.1,
        device: str = "cuda",
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.ema_alpha = ema_alpha
        self.device = device

        # Running mean for novelty computation
        self.k_running_mean = torch.zeros(
            num_layers, num_heads, head_dim, device=device
        )
        self.update_count = 0

        # Store novelty scores for each position
        self.novelty_scores: list[torch.Tensor] = []  # [num_layers, num_heads] per position

    def compute_and_store_novelty(
        self,
        k_new: torch.Tensor,  # [num_layers, num_heads, head_dim]
    ) -> torch.Tensor:
        """Compute novelty for new K vectors and store."""

        if self.update_count == 0:
            novelty = torch.zeros(self.num_layers, self.num_heads, device=self.device)
        else:
            # Cosine distance from running mean
            k_new_norm = F.normalize(k_new, dim=-1)
            k_mean_norm = F.normalize(self.k_running_mean, dim=-1)
            cos_sim = (k_new_norm * k_mean_norm).sum(dim=-1)
            novelty = 1.0 - cos_sim

        # Update running mean
        if self.update_count == 0:
            self.k_running_mean = k_new.clone()
        else:
            self.k_running_mean = (
                self.ema_alpha * k_new +
                (1 - self.ema_alpha) * self.k_running_mean
            )

        self.update_count += 1
        self.novelty_scores.append(novelty.cpu())

        return novelty

    def get_compression_mask(
        self,
        config: CompressionConfig,
    ) -> torch.Tensor:
        """
        Get mask indicating which positions to compress.

        Returns:
            mask: [seq_len, num_layers, num_heads] - True = compress this entry
        """
        if not self.novelty_scores:
            return torch.zeros(0, self.num_layers, self.num_heads, dtype=torch.bool)

        # Stack all novelty scores: [seq_len, num_layers, num_heads]
        all_novelty = torch.stack(self.novelty_scores)
        seq_len = all_novelty.shape[0]

        # Compute threshold based on percentile
        flat_novelty = all_novelty.flatten()
        threshold = torch.quantile(flat_novelty, config.novelty_percentile / 100.0)

        # Mark low-novelty entries for compression
        compress_mask = all_novelty < threshold

        if config.strategy == "conservative":
            # Find spike positions (high novelty)
            global_novelty = all_novelty.mean(dim=(1, 2))  # [seq_len]
            spike_threshold = torch.quantile(global_novelty, 0.9)  # Top 10% are spikes
            spike_positions = (global_novelty >= spike_threshold).nonzero().squeeze(-1)

            # Protect window around spikes
            for spike_pos in spike_positions:
                start = max(0, spike_pos - config.spike_window)
                end = min(seq_len, spike_pos + config.spike_window + 1)
                compress_mask[start:end] = False

        # Apply layer/head filters
        if config.layers_to_compress is not None:
            layer_mask = torch.zeros(self.num_layers, dtype=torch.bool)
            for l in config.layers_to_compress:
                if l < self.num_layers:
                    layer_mask[l] = True
            compress_mask = compress_mask & layer_mask.unsqueeze(0).unsqueeze(-1)

        if config.heads_to_compress is not None:
            head_mask = torch.zeros(self.num_heads, dtype=torch.bool)
            for h in config.heads_to_compress:
                if h < self.num_heads:
                    head_mask[h] = True
            compress_mask = compress_mask & head_mask.unsqueeze(0).unsqueeze(0)

        return compress_mask

    def reset(self):
        """Reset tracker state."""
        self.k_running_mean.zero_()
        self.update_count = 0
        self.novelty_scores.clear()


def apply_compression_to_cache(
    past_key_values: Any,
    compress_mask: torch.Tensor,  # [seq_len, num_layers, num_heads]
    config: CompressionConfig,
    prompt_len: int = 0,
) -> Any:
    """
    Apply compression to KV cache based on mask.

    Args:
        past_key_values: KV cache (DynamicCache or tuple)
        compress_mask: Which entries to compress
        config: Compression configuration
        prompt_len: Length of prompt (don't compress prompt tokens)

    Returns:
        Modified KV cache
    """
    try:
        from transformers.cache_utils import DynamicCache
        has_dynamic_cache = True
    except ImportError:
        has_dynamic_cache = False

    is_cache_object = hasattr(past_key_values, 'get_seq_length')

    if is_cache_object:
        # Convert to legacy, modify, convert back
        legacy_cache = past_key_values.to_legacy_cache()
    else:
        legacy_cache = past_key_values

    modified_cache = []

    for layer_idx, (key, value) in enumerate(legacy_cache):
        # key, value: [batch, num_heads, seq_len, head_dim]
        key = key.clone()
        value = value.clone()

        seq_len = key.shape[2]

        for pos in range(prompt_len, seq_len):
            mask_pos = pos - prompt_len
            if mask_pos >= compress_mask.shape[0]:
                continue

            for head_idx in range(key.shape[1]):
                if layer_idx >= compress_mask.shape[1]:
                    continue
                if head_idx >= compress_mask.shape[2]:
                    continue

                if compress_mask[mask_pos, layer_idx, head_idx]:
                    # Compress this entry
                    if config.drop_method == "zero":
                        if config.strategy in ["k_only", "conservative"]:
                            key[0, head_idx, pos, :] = 0
                        else:  # full
                            key[0, head_idx, pos, :] = 0
                            value[0, head_idx, pos, :] = 0

                    elif config.drop_method == "mean":
                        if config.strategy in ["k_only", "conservative"]:
                            key[0, head_idx, pos, :] = key[0, head_idx, :pos, :].mean(dim=0)
                        else:
                            key[0, head_idx, pos, :] = key[0, head_idx, :pos, :].mean(dim=0)
                            value[0, head_idx, pos, :] = value[0, head_idx, :pos, :].mean(dim=0)

                    elif config.drop_method == "first":
                        # Copy from first position
                        if config.strategy in ["k_only", "conservative"]:
                            key[0, head_idx, pos, :] = key[0, head_idx, 0, :]
                        else:
                            key[0, head_idx, pos, :] = key[0, head_idx, 0, :]
                            value[0, head_idx, pos, :] = value[0, head_idx, 0, :]

        modified_cache.append((key, value))

    modified_cache = tuple(modified_cache)

    if is_cache_object and has_dynamic_cache:
        return DynamicCache.from_legacy_cache(modified_cache)

    return modified_cache


def run_compression_experiment(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    config: CompressionConfig,
    max_new_tokens: int = 100,
    device: str = "cuda",
) -> CompressionResult:
    """
    Run a KV-cache compression experiment.

    Generates text with full KV cache, then measures quality degradation
    when using compressed cache.
    """
    model.eval()

    # Get model config
    model_config = model.config
    num_layers = getattr(model_config, "num_hidden_layers", 32)
    num_heads = getattr(model_config, "num_attention_heads", 32)
    num_kv_heads = getattr(model_config, "num_key_value_heads", num_heads)
    head_dim = getattr(model_config, "head_dim",
                       model_config.hidden_size // num_heads)

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    # Initialize novelty tracker
    tracker = NoveltyTracker(
        num_layers=num_layers,
        num_heads=num_kv_heads,
        head_dim=head_dim,
        device=device,
    )

    # Phase 1: Generate with full cache, track novelty
    print("Phase 1: Generating with full cache...")

    generated_ids = inputs["input_ids"].clone()
    past_key_values = None
    original_logprobs = []

    with torch.no_grad():
        for step in range(max_new_tokens):
            outputs = model(
                generated_ids if past_key_values is None else generated_ids[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

            # Track novelty from K cache
            if past_key_values is not None:
                is_cache_object = hasattr(past_key_values, 'get_seq_length')
                if is_cache_object:
                    legacy = past_key_values.to_legacy_cache()
                else:
                    legacy = past_key_values

                # Extract K from each layer
                k_stack = []
                for layer_idx in range(min(num_layers, len(legacy))):
                    key, _ = legacy[layer_idx]
                    k_last = key[0, :, -1, :].detach()  # [num_heads, head_dim]
                    k_stack.append(k_last)

                if k_stack:
                    k_tensor = torch.stack(k_stack)  # [num_layers, num_heads, head_dim]
                    tracker.compute_and_store_novelty(k_tensor)

            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Store logprob of chosen token
            log_probs = F.log_softmax(logits, dim=-1)
            original_logprobs.append(log_probs[0, next_token[0, 0]].item())

            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            if next_token[0, 0].item() == tokenizer.eos_token_id:
                break

    original_loss = -np.mean(original_logprobs)
    full_cache = past_key_values
    generated_text = tokenizer.decode(generated_ids[0, prompt_len:], skip_special_tokens=True)

    print(f"  Generated {len(original_logprobs)} tokens, loss={original_loss:.4f}")

    # Phase 2: Get compression mask
    compress_mask = tracker.get_compression_mask(config)
    total_entries = compress_mask.numel()
    compressed_entries = compress_mask.sum().item()
    compression_ratio = compressed_entries / total_entries if total_entries > 0 else 0

    print(f"Phase 2: Compression mask - {compressed_entries}/{total_entries} entries ({compression_ratio*100:.1f}%)")

    # Phase 3: Apply compression and measure quality
    print("Phase 3: Measuring quality with compressed cache...")

    compressed_cache = apply_compression_to_cache(
        full_cache,
        compress_mask,
        config,
        prompt_len=prompt_len,
    )

    # Re-run with compressed cache to measure per-token quality
    compressed_logprobs = []
    kl_divergences = []
    worst_tokens = []

    with torch.no_grad():
        # Need to re-run generation with compressed cache
        # We'll compute logits for each position using the compressed cache

        for step in range(len(original_logprobs)):
            # Get position in sequence
            pos = prompt_len + step

            # Run forward with compressed cache up to this point
            input_ids = generated_ids[:, :pos]

            outputs_compressed = model(
                input_ids,
                past_key_values=None,  # Recompute
                use_cache=True,
                return_dict=True,
            )

            # Get the cache and apply compression
            temp_cache = outputs_compressed.past_key_values
            temp_compressed = apply_compression_to_cache(
                temp_cache,
                compress_mask[:step] if step > 0 else compress_mask[:1],
                config,
                prompt_len=prompt_len,
            )

            # Run one more step with compressed cache
            outputs_final = model(
                generated_ids[:, pos:pos+1],
                past_key_values=temp_compressed,
                use_cache=False,
                return_dict=True,
            )

            logits_compressed = outputs_final.logits[:, -1, :]

            # Compute metrics
            log_probs_compressed = F.log_softmax(logits_compressed, dim=-1)
            actual_token = generated_ids[0, pos + 1].item() if pos + 1 < generated_ids.shape[1] else generated_ids[0, pos].item()
            compressed_logprobs.append(log_probs_compressed[0, actual_token].item())

            # KL divergence
            probs_original = F.softmax(torch.tensor([original_logprobs[step]]), dim=-1)
            # Simplified: just measure logprob difference
            kl = abs(original_logprobs[step] - compressed_logprobs[-1])
            kl_divergences.append(kl)

            if kl > 0.5:  # Significant difference
                token_text = tokenizer.decode([actual_token])
                worst_tokens.append({
                    "position": step,
                    "token": token_text,
                    "original_logprob": original_logprobs[step],
                    "compressed_logprob": compressed_logprobs[-1],
                    "delta": kl,
                })

    compressed_loss = -np.mean(compressed_logprobs) if compressed_logprobs else original_loss
    loss_increase = compressed_loss - original_loss
    loss_increase_pct = (loss_increase / original_loss * 100) if original_loss > 0 else 0

    # Sort worst tokens by delta
    worst_tokens.sort(key=lambda x: -x["delta"])

    return CompressionResult(
        config=vars(config) if hasattr(config, '__dict__') else {},
        total_entries=total_entries,
        compressed_entries=compressed_entries,
        compression_ratio=compression_ratio,
        original_loss=original_loss,
        compressed_loss=compressed_loss,
        loss_increase=loss_increase,
        loss_increase_pct=loss_increase_pct,
        mean_kl_divergence=float(np.mean(kl_divergences)) if kl_divergences else 0,
        max_kl_divergence=float(max(kl_divergences)) if kl_divergences else 0,
        worst_affected_tokens=worst_tokens[:10],
    )


def print_compression_results(result: CompressionResult):
    """Print compression experiment results."""
    print("\n" + "=" * 70)
    print("KV-CACHE COMPRESSION RESULTS")
    print("=" * 70)

    print(f"\n--- Compression Stats ---")
    print(f"Total KV entries: {result.total_entries}")
    print(f"Compressed entries: {result.compressed_entries}")
    print(f"Compression ratio: {result.compression_ratio*100:.1f}%")

    print(f"\n--- Quality Metrics ---")
    print(f"Original loss: {result.original_loss:.4f}")
    print(f"Compressed loss: {result.compressed_loss:.4f}")
    print(f"Loss increase: {result.loss_increase:.4f} ({result.loss_increase_pct:.2f}%)")
    print(f"Mean |Δlogprob|: {result.mean_kl_divergence:.4f}")
    print(f"Max |Δlogprob|: {result.max_kl_divergence:.4f}")

    if result.worst_affected_tokens:
        print(f"\n--- Most Affected Tokens ---")
        for t in result.worst_affected_tokens[:5]:
            print(f"  [{t['position']}] {repr(t['token'])}: Δ={t['delta']:.3f}")

    print("=" * 70)
