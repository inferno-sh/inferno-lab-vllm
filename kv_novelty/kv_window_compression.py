"""
Window-based KV-cache compression using low-rank approximation and clustering.

Instead of dropping low-novelty tokens, compress them into a smaller representation
that preserves attention geometry.

Methods:
1. Low-rank (SVD): Approximate K matrix with top-k singular vectors
2. Clustering: Replace tokens with cluster centroids
3. Hybrid: Cluster then low-rank within clusters
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CompressionMethod(str, Enum):
    LOW_RANK = "low_rank"
    CLUSTER = "cluster"
    HYBRID = "hybrid"


@dataclass
class WindowCompressionConfig:
    """Configuration for window-based KV compression."""

    # Window parameters
    window_size: int = 16  # Tokens per compression window
    stride: int = 8  # Overlap between windows

    # Which tokens to compress (by novelty percentile)
    novelty_threshold: float = 50.0  # Compress bottom 50%

    # Compression method
    method: CompressionMethod = CompressionMethod.LOW_RANK

    # Low-rank parameters
    rank_ratio: float = 0.5  # Keep this fraction of singular values
    min_rank: int = 2  # Minimum rank to preserve

    # Clustering parameters
    n_clusters_ratio: float = 0.5  # Reduce to this fraction of tokens
    min_clusters: int = 2

    # Don't compress prompt or recent tokens
    prompt_protection: int = 0  # Protect first N tokens
    recency_protection: int = 4  # Protect last N tokens

    # Compression scope
    compress_keys: bool = True
    compress_values: bool = True  # Usually want both for consistency


@dataclass
class WindowCompressionResult:
    """Results from window compression experiment."""
    config: dict

    # Compression stats
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float

    # Per-window stats
    windows_compressed: int
    avg_window_compression: float

    # Quality metrics
    original_loss: float
    compressed_loss: float
    loss_increase_pct: float

    # Detailed metrics
    mean_logprob_delta: float
    max_logprob_delta: float

    # Reconstruction quality (for K)
    mean_reconstruction_error: float


class KVWindowCompressor:
    """
    Compresses KV cache using sliding windows over low-novelty regions.
    """

    def __init__(self, config: WindowCompressionConfig, device: str = "cuda"):
        self.config = config
        self.device = device

    def identify_compression_windows(
        self,
        novelty_scores: torch.Tensor,  # [seq_len]
    ) -> list[tuple[int, int]]:
        """
        Identify windows of low-novelty tokens to compress.

        Returns list of (start, end) indices for compression windows.
        """
        if novelty_scores.numel() == 0:
            return []

        seq_len = novelty_scores.shape[0]
        if seq_len < self.config.window_size // 2:
            return []

        threshold = torch.quantile(novelty_scores, self.config.novelty_threshold / 100.0)

        # Mark low-novelty positions
        low_novelty = novelty_scores < threshold

        # Protect prompt and recent tokens
        if self.config.prompt_protection > 0:
            low_novelty[:self.config.prompt_protection] = False
        if self.config.recency_protection > 0:
            low_novelty[-self.config.recency_protection:] = False

        # Find contiguous low-novelty regions
        windows = []
        start = None

        for i in range(seq_len):
            if low_novelty[i] and start is None:
                start = i
            elif not low_novelty[i] and start is not None:
                if i - start >= self.config.window_size // 2:  # Min window size
                    windows.append((start, i))
                start = None

        # Handle trailing window
        if start is not None and seq_len - start >= self.config.window_size // 2:
            windows.append((start, seq_len))

        return windows

    def compress_window_low_rank(
        self,
        k_window: torch.Tensor,  # [num_heads, window_len, head_dim]
        v_window: torch.Tensor,  # [num_heads, window_len, head_dim]
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Compress a window using SVD (low-rank approximation).

        For each head, compute SVD of K and keep top-k singular vectors.
        The compressed representation is: K_compressed = U_k @ S_k @ V_k^T

        Returns compressed K, V and metadata.
        """
        num_heads, window_len, head_dim = k_window.shape

        # Determine rank
        target_rank = max(
            self.config.min_rank,
            int(window_len * self.config.rank_ratio)
        )
        target_rank = min(target_rank, window_len, head_dim)

        k_compressed_list = []
        v_compressed_list = []
        reconstruction_errors = []

        for h in range(num_heads):
            k_h = k_window[h]  # [window_len, head_dim]
            v_h = v_window[h]

            # SVD on K
            try:
                U, S, Vh = torch.linalg.svd(k_h, full_matrices=False)

                # Keep top-k components
                U_k = U[:, :target_rank]  # [window_len, rank]
                S_k = S[:target_rank]  # [rank]
                Vh_k = Vh[:target_rank, :]  # [rank, head_dim]

                # Reconstruct with reduced rank
                k_reconstructed = U_k @ torch.diag(S_k) @ Vh_k  # [window_len, head_dim]

                # For a true compression, we'd store U_k, S_k, Vh_k separately
                # But for this experiment, we just use the reconstructed K
                # to measure quality impact

                # Compute reconstruction error
                error = torch.norm(k_h - k_reconstructed) / torch.norm(k_h)
                reconstruction_errors.append(error.item())

                k_compressed_list.append(k_reconstructed)

            except Exception:
                # SVD failed, keep original
                k_compressed_list.append(k_h)
                reconstruction_errors.append(0.0)

            # Apply same transformation to V for consistency
            # (In practice, might want independent compression)
            if self.config.compress_values:
                try:
                    U_v, S_v, Vh_v = torch.linalg.svd(v_h, full_matrices=False)
                    U_vk = U_v[:, :target_rank]
                    S_vk = S_v[:target_rank]
                    Vh_vk = Vh_v[:target_rank, :]
                    v_reconstructed = U_vk @ torch.diag(S_vk) @ Vh_vk
                    v_compressed_list.append(v_reconstructed)
                except Exception:
                    v_compressed_list.append(v_h)
            else:
                v_compressed_list.append(v_h)

        k_compressed = torch.stack(k_compressed_list)
        v_compressed = torch.stack(v_compressed_list)

        metadata = {
            "method": "low_rank",
            "original_len": window_len,
            "target_rank": target_rank,
            "mean_reconstruction_error": np.mean(reconstruction_errors),
        }

        return k_compressed, v_compressed, metadata

    def compress_window_cluster(
        self,
        k_window: torch.Tensor,  # [num_heads, window_len, head_dim]
        v_window: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Compress a window by clustering tokens and keeping centroids.

        This actually reduces the sequence length (true compression).
        """
        num_heads, window_len, head_dim = k_window.shape

        n_clusters = max(
            self.config.min_clusters,
            int(window_len * self.config.n_clusters_ratio)
        )
        n_clusters = min(n_clusters, window_len)

        # Cluster based on K vectors (averaged across heads for assignment)
        k_mean = k_window.mean(dim=0)  # [window_len, head_dim]

        # Simple k-means
        centroids, assignments = self._kmeans(k_mean, n_clusters)

        # For each cluster, compute centroid K and V for each head
        k_centroids = torch.zeros(num_heads, n_clusters, head_dim, device=k_window.device)
        v_centroids = torch.zeros(num_heads, n_clusters, head_dim, device=v_window.device)

        for c in range(n_clusters):
            mask = assignments == c
            if mask.sum() > 0:
                k_centroids[:, c, :] = k_window[:, mask, :].mean(dim=1)
                v_centroids[:, c, :] = v_window[:, mask, :].mean(dim=1)

        metadata = {
            "method": "cluster",
            "original_len": window_len,
            "compressed_len": n_clusters,
            "compression_ratio": n_clusters / window_len,
        }

        return k_centroids, v_centroids, metadata

    def _kmeans(
        self,
        data: torch.Tensor,  # [n_points, dim]
        k: int,
        max_iters: int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Simple k-means clustering."""
        n_points, dim = data.shape

        # Initialize centroids randomly
        indices = torch.randperm(n_points)[:k]
        centroids = data[indices].clone()

        for _ in range(max_iters):
            # Assign points to nearest centroid
            distances = torch.cdist(data, centroids)  # [n_points, k]
            assignments = distances.argmin(dim=1)  # [n_points]

            # Update centroids
            new_centroids = torch.zeros_like(centroids)
            for c in range(k):
                mask = assignments == c
                if mask.sum() > 0:
                    new_centroids[c] = data[mask].mean(dim=0)
                else:
                    new_centroids[c] = centroids[c]

            if torch.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        return centroids, assignments

    def compress_cache(
        self,
        past_key_values: Any,
        novelty_scores: torch.Tensor,  # [seq_len] global novelty per position
        prompt_len: int = 0,
    ) -> tuple[Any, dict]:
        """
        Apply window compression to KV cache.

        Note: For low_rank, we modify in place (same shape).
        For cluster, we'd need to actually reduce sequence length,
        which requires more careful handling of attention masks.

        For this experiment, we use low_rank which preserves shape
        but approximates the values.
        """
        try:
            from transformers.cache_utils import DynamicCache
            has_dynamic_cache = True
        except ImportError:
            has_dynamic_cache = False

        is_cache_object = hasattr(past_key_values, 'get_seq_length')

        if is_cache_object:
            legacy_cache = past_key_values.to_legacy_cache()
        else:
            legacy_cache = past_key_values

        # Adjust novelty scores to exclude prompt
        gen_novelty = novelty_scores[prompt_len:] if prompt_len > 0 else novelty_scores

        # Identify compression windows
        windows = self.identify_compression_windows(gen_novelty)

        if not windows:
            return past_key_values, {"windows_compressed": 0}

        modified_cache = []
        total_reconstruction_error = 0.0
        n_windows = 0

        for layer_idx, (key, value) in enumerate(legacy_cache):
            # key, value: [batch, num_heads, seq_len, head_dim]
            key = key.clone()
            value = value.clone()

            for win_start, win_end in windows:
                # Adjust for prompt offset
                actual_start = prompt_len + win_start
                actual_end = prompt_len + win_end

                if actual_end > key.shape[2]:
                    continue

                # Extract window
                k_window = key[0, :, actual_start:actual_end, :]  # [heads, win_len, dim]
                v_window = value[0, :, actual_start:actual_end, :]

                # Compress
                if self.config.method == CompressionMethod.LOW_RANK:
                    k_comp, v_comp, meta = self.compress_window_low_rank(k_window, v_window)

                    # Replace in cache (same shape for low_rank)
                    key[0, :, actual_start:actual_end, :] = k_comp
                    value[0, :, actual_start:actual_end, :] = v_comp

                    total_reconstruction_error += meta.get("mean_reconstruction_error", 0)
                    n_windows += 1

                elif self.config.method == CompressionMethod.CLUSTER:
                    # Clustering reduces length - more complex to handle
                    # For now, we use low_rank for in-place compression
                    k_comp, v_comp, meta = self.compress_window_low_rank(k_window, v_window)
                    key[0, :, actual_start:actual_end, :] = k_comp
                    value[0, :, actual_start:actual_end, :] = v_comp
                    n_windows += 1

            modified_cache.append((key, value))

        modified_cache = tuple(modified_cache)

        if is_cache_object and has_dynamic_cache:
            result_cache = DynamicCache.from_legacy_cache(modified_cache)
        else:
            result_cache = modified_cache

        stats = {
            "windows_compressed": len(windows) * len(legacy_cache),
            "unique_windows": len(windows),
            "mean_reconstruction_error": total_reconstruction_error / max(n_windows, 1),
        }

        return result_cache, stats


def run_window_compression_experiment(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    config: WindowCompressionConfig,
    max_new_tokens: int = 100,
    device: str = "cuda",
) -> WindowCompressionResult:
    """
    Run a window-based KV compression experiment.
    """
    from .kv_compression import NoveltyTracker

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

    # Initialize trackers
    novelty_tracker = NoveltyTracker(
        num_layers=num_layers,
        num_heads=num_kv_heads,
        head_dim=head_dim,
        device=device,
    )
    compressor = KVWindowCompressor(config, device=device)

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

            # Track novelty
            if past_key_values is not None:
                is_cache_object = hasattr(past_key_values, 'get_seq_length')
                if is_cache_object:
                    legacy = past_key_values.to_legacy_cache()
                else:
                    legacy = past_key_values

                k_stack = []
                for layer_idx in range(min(num_layers, len(legacy))):
                    key, _ = legacy[layer_idx]
                    k_last = key[0, :, -1, :].detach()
                    k_stack.append(k_last)

                if k_stack:
                    k_tensor = torch.stack(k_stack)
                    novelty_tracker.compute_and_store_novelty(k_tensor)

            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            log_probs = F.log_softmax(logits, dim=-1)
            original_logprobs.append(log_probs[0, next_token[0, 0]].item())

            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            if next_token[0, 0].item() == tokenizer.eos_token_id:
                break

    original_loss = -np.mean(original_logprobs)
    full_cache = past_key_values
    num_generated = len(original_logprobs)

    print(f"  Generated {num_generated} tokens, loss={original_loss:.4f}")

    # Phase 2: Compute global novelty and identify windows
    all_novelty = torch.stack(novelty_tracker.novelty_scores)  # [seq_len, layers, heads]
    global_novelty = all_novelty.mean(dim=(1, 2))  # [seq_len]

    windows = compressor.identify_compression_windows(global_novelty)
    total_window_tokens = sum(end - start for start, end in windows)

    print(f"Phase 2: Found {len(windows)} compression windows ({total_window_tokens} tokens)")

    # Phase 3: Apply compression and measure quality
    print("Phase 3: Measuring quality with compressed cache...")

    compressed_logprobs = []
    logprob_deltas = []

    with torch.no_grad():
        for step in range(num_generated):
            pos = prompt_len + step
            input_ids = generated_ids[:, :pos]

            # Run forward to get cache
            outputs_temp = model(
                input_ids,
                past_key_values=None,
                use_cache=True,
                return_dict=True,
            )

            # Apply window compression (only if we have enough history)
            if step < config.window_size // 2:
                compressed_cache = outputs_temp.past_key_values
            else:
                compressed_cache, comp_stats = compressor.compress_cache(
                    outputs_temp.past_key_values,
                    global_novelty[:step],
                    prompt_len=prompt_len,
                )

            # Run one more step with compressed cache
            outputs_final = model(
                generated_ids[:, pos:pos+1],
                past_key_values=compressed_cache,
                use_cache=False,
                return_dict=True,
            )

            logits_compressed = outputs_final.logits[:, -1, :]
            log_probs_compressed = F.log_softmax(logits_compressed, dim=-1)

            actual_token = generated_ids[0, pos + 1].item() if pos + 1 < generated_ids.shape[1] else generated_ids[0, pos].item()
            compressed_logprobs.append(log_probs_compressed[0, actual_token].item())

            delta = abs(original_logprobs[step] - compressed_logprobs[-1])
            logprob_deltas.append(delta)

    compressed_loss = -np.mean(compressed_logprobs) if compressed_logprobs else original_loss
    loss_increase_pct = ((compressed_loss - original_loss) / original_loss * 100) if original_loss > 0 else 0

    # Get final compression stats
    _, final_stats = compressor.compress_cache(full_cache, global_novelty, prompt_len)

    compression_ratio = total_window_tokens / num_generated if num_generated > 0 else 0

    return WindowCompressionResult(
        config=vars(config) if hasattr(config, '__dict__') else {},
        original_tokens=num_generated,
        compressed_tokens=num_generated - total_window_tokens,
        compression_ratio=compression_ratio,
        windows_compressed=len(windows),
        avg_window_compression=total_window_tokens / len(windows) if windows else 0,
        original_loss=original_loss,
        compressed_loss=compressed_loss,
        loss_increase_pct=loss_increase_pct,
        mean_logprob_delta=float(np.mean(logprob_deltas)) if logprob_deltas else 0,
        max_logprob_delta=float(max(logprob_deltas)) if logprob_deltas else 0,
        mean_reconstruction_error=final_stats.get("mean_reconstruction_error", 0),
    )


def print_window_compression_results(result: WindowCompressionResult):
    """Print window compression results."""
    print("\n" + "=" * 70)
    print("WINDOW-BASED KV COMPRESSION RESULTS")
    print("=" * 70)

    print(f"\n--- Compression Stats ---")
    print(f"Windows compressed: {result.windows_compressed}")
    print(f"Tokens in windows: {result.original_tokens - result.compressed_tokens}")
    print(f"Window coverage: {result.compression_ratio*100:.1f}% of generated tokens")
    print(f"Mean reconstruction error: {result.mean_reconstruction_error:.4f}")

    print(f"\n--- Quality Metrics ---")
    print(f"Original loss: {result.original_loss:.4f}")
    print(f"Compressed loss: {result.compressed_loss:.4f}")
    print(f"Loss increase: {result.loss_increase_pct:.2f}%")
    print(f"Mean |Δlogprob|: {result.mean_logprob_delta:.4f}")
    print(f"Max |Δlogprob|: {result.max_logprob_delta:.4f}")

    print("=" * 70)
