"""
Data collection and storage for KV-cache novelty analysis.

This module provides the KVNoveltyCollector class which accumulates
KV tensors, logprobs, and other data during generation for later analysis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import numpy as np


@dataclass
class TokenData:
    """Data associated with a single generated token."""

    position: int
    token_id: int
    token_text: str
    logprob: float
    entropy: float
    top_k_logprobs: dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "token_id": self.token_id,
            "token_text": self.token_text,
            "logprob": self.logprob,
            "entropy": self.entropy,
            "top_k_logprobs": self.top_k_logprobs,
        }


@dataclass
class LayerKVData:
    """KV data for a single layer at a single token position."""

    layer_idx: int
    position: int
    # Shape: [num_kv_heads, head_size]
    key: np.ndarray | None = None
    value: np.ndarray | None = None
    # Attention weights if captured (optional)
    attention_weights: np.ndarray | None = None


class KVNoveltyCollector:
    """
    Collects KV-cache data during generation for novelty analysis.

    This class accumulates:
    - Key and value tensors from each layer at each token position
    - Token logprobs and entropy
    - Attention weights (if available)

    Data is stored in memory during generation and can be saved to disk
    for later analysis.
    """

    def __init__(
        self,
        capture_kv: bool = True,
        capture_attention: bool = False,
        layers_to_capture: list[int] | None = None,
        heads_to_capture: list[int] | None = None,
        max_positions: int = 10000,
        device: str = "cpu",
    ):
        """
        Initialize the collector.

        Args:
            capture_kv: Whether to capture K, V tensors
            capture_attention: Whether to capture attention weights (expensive)
            layers_to_capture: Specific layers to capture (None = all)
            heads_to_capture: Specific heads to capture (None = all)
            max_positions: Maximum number of positions to capture
            device: Device to store captured tensors ("cpu" recommended for memory)
        """
        self.capture_kv = capture_kv
        self.capture_attention = capture_attention
        self.layers_to_capture = layers_to_capture
        self.heads_to_capture = heads_to_capture
        self.max_positions = max_positions
        self.device = device

        # Storage
        self.token_data: list[TokenData] = []
        # layer_idx -> position -> LayerKVData
        self.kv_data: dict[int, dict[int, LayerKVData]] = {}

        # Metadata
        self.model_name: str = ""
        self.prompt: str = ""
        self.generation_config: dict[str, Any] = {}
        self.num_layers: int = 0
        self.num_kv_heads: int = 0
        self.head_size: int = 0

        # Current state
        self._current_position: int = 0
        self._is_collecting: bool = False

    def start_collection(
        self,
        model_name: str = "",
        prompt: str = "",
        generation_config: dict[str, Any] | None = None,
    ) -> None:
        """Start collecting data for a new generation."""
        self.clear()
        self.model_name = model_name
        self.prompt = prompt
        self.generation_config = generation_config or {}
        self._is_collecting = True

    def stop_collection(self) -> None:
        """Stop collecting data."""
        self._is_collecting = False

    def clear(self) -> None:
        """Clear all collected data."""
        self.token_data = []
        self.kv_data = {}
        self._current_position = 0
        self._is_collecting = False

    def should_capture_layer(self, layer_idx: int) -> bool:
        """Check if this layer should be captured."""
        if not self._is_collecting:
            return False
        if self._current_position >= self.max_positions:
            return False
        if self.layers_to_capture is not None:
            return layer_idx in self.layers_to_capture
        return True

    def record_token(
        self,
        token_id: int,
        token_text: str,
        logprob: float,
        entropy: float,
        top_k_logprobs: dict[int, float] | None = None,
    ) -> None:
        """Record token-level data."""
        if not self._is_collecting:
            return
        if self._current_position >= self.max_positions:
            return

        self.token_data.append(TokenData(
            position=self._current_position,
            token_id=token_id,
            token_text=token_text,
            logprob=logprob,
            entropy=entropy,
            top_k_logprobs=top_k_logprobs or {},
        ))

    def record_kv(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_weights: torch.Tensor | None = None,
    ) -> None:
        """
        Record KV tensors for a layer.

        Args:
            layer_idx: Layer index
            key: Key tensor, shape [num_tokens, num_kv_heads, head_size]
            value: Value tensor, shape [num_tokens, num_kv_heads, head_size]
            attention_weights: Optional attention weights
        """
        if not self.should_capture_layer(layer_idx):
            return
        if not self.capture_kv:
            return

        # Initialize layer storage if needed
        if layer_idx not in self.kv_data:
            self.kv_data[layer_idx] = {}

        # Store model dimensions if not set
        if self.num_kv_heads == 0 and key is not None:
            if key.dim() == 3:
                self.num_kv_heads = key.shape[1]
                self.head_size = key.shape[2]
            elif key.dim() == 2:
                # Might be flattened
                pass

        # We capture only the last token's KV (for decode steps)
        # or all tokens for prefill
        if key is not None and key.shape[0] > 0:
            # Get the last token's KV (most relevant for decode)
            k_last = key[-1].detach().cpu().numpy()
            v_last = value[-1].detach().cpu().numpy() if value is not None else None

            # Optionally filter heads
            if self.heads_to_capture is not None and k_last.ndim >= 2:
                k_last = k_last[self.heads_to_capture]
                if v_last is not None:
                    v_last = v_last[self.heads_to_capture]

            attn = None
            if self.capture_attention and attention_weights is not None:
                attn = attention_weights[-1].detach().cpu().numpy()

            self.kv_data[layer_idx][self._current_position] = LayerKVData(
                layer_idx=layer_idx,
                position=self._current_position,
                key=k_last,
                value=v_last,
                attention_weights=attn,
            )

    def advance_position(self) -> None:
        """Advance to the next token position."""
        self._current_position += 1

    def get_keys_for_layer(self, layer_idx: int) -> np.ndarray | None:
        """
        Get all captured keys for a layer as a single array.

        Returns:
            Array of shape [num_positions, num_kv_heads, head_size]
        """
        if layer_idx not in self.kv_data:
            return None

        positions = sorted(self.kv_data[layer_idx].keys())
        if not positions:
            return None

        keys = []
        for pos in positions:
            kv_data = self.kv_data[layer_idx][pos]
            if kv_data.key is not None:
                keys.append(kv_data.key)

        if not keys:
            return None

        return np.stack(keys, axis=0)

    def get_values_for_layer(self, layer_idx: int) -> np.ndarray | None:
        """
        Get all captured values for a layer as a single array.

        Returns:
            Array of shape [num_positions, num_kv_heads, head_size]
        """
        if layer_idx not in self.kv_data:
            return None

        positions = sorted(self.kv_data[layer_idx].keys())
        if not positions:
            return None

        values = []
        for pos in positions:
            kv_data = self.kv_data[layer_idx][pos]
            if kv_data.value is not None:
                values.append(kv_data.value)

        if not values:
            return None

        return np.stack(values, axis=0)

    def get_logprobs(self) -> np.ndarray:
        """Get logprobs for all tokens."""
        return np.array([t.logprob for t in self.token_data])

    def get_entropies(self) -> np.ndarray:
        """Get entropies for all tokens."""
        return np.array([t.entropy for t in self.token_data])

    def get_tokens(self) -> list[str]:
        """Get token strings."""
        return [t.token_text for t in self.token_data]

    def get_generated_text(self) -> str:
        """Get the full generated text."""
        return "".join(self.get_tokens())

    @property
    def num_positions(self) -> int:
        """Number of captured positions."""
        return len(self.token_data)

    @property
    def captured_layers(self) -> list[int]:
        """List of layer indices with captured data."""
        return sorted(self.kv_data.keys())

    def save(self, path: str | Path) -> None:
        """
        Save collected data to disk.

        Saves in a directory structure:
        - metadata.json: Model info, config, token data
        - kv_layer_{i}.npz: KV tensors for each layer
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save metadata and token data
        metadata = {
            "model_name": self.model_name,
            "prompt": self.prompt,
            "generation_config": self.generation_config,
            "num_layers": self.num_layers,
            "num_kv_heads": self.num_kv_heads,
            "head_size": self.head_size,
            "num_positions": self.num_positions,
            "captured_layers": self.captured_layers,
            "token_data": [t.to_dict() for t in self.token_data],
        }

        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save KV data for each layer
        for layer_idx, layer_data in self.kv_data.items():
            keys = self.get_keys_for_layer(layer_idx)
            values = self.get_values_for_layer(layer_idx)

            arrays = {}
            if keys is not None:
                arrays["keys"] = keys
            if values is not None:
                arrays["values"] = values

            if arrays:
                np.savez_compressed(path / f"kv_layer_{layer_idx}.npz", **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "KVNoveltyCollector":
        """Load collected data from disk."""
        path = Path(path)

        # Load metadata
        with open(path / "metadata.json") as f:
            metadata = json.load(f)

        collector = cls()
        collector.model_name = metadata["model_name"]
        collector.prompt = metadata["prompt"]
        collector.generation_config = metadata["generation_config"]
        collector.num_layers = metadata["num_layers"]
        collector.num_kv_heads = metadata["num_kv_heads"]
        collector.head_size = metadata["head_size"]

        # Load token data
        for t in metadata["token_data"]:
            collector.token_data.append(TokenData(
                position=t["position"],
                token_id=t["token_id"],
                token_text=t["token_text"],
                logprob=t["logprob"],
                entropy=t["entropy"],
                top_k_logprobs=t.get("top_k_logprobs", {}),
            ))

        # Load KV data for each layer
        for layer_idx in metadata["captured_layers"]:
            kv_file = path / f"kv_layer_{layer_idx}.npz"
            if kv_file.exists():
                data = np.load(kv_file)
                keys = data.get("keys")
                values = data.get("values")

                if keys is not None:
                    collector.kv_data[layer_idx] = {}
                    for pos in range(keys.shape[0]):
                        collector.kv_data[layer_idx][pos] = LayerKVData(
                            layer_idx=layer_idx,
                            position=pos,
                            key=keys[pos],
                            value=values[pos] if values is not None else None,
                        )

        return collector

    def __repr__(self) -> str:
        return (
            f"KVNoveltyCollector("
            f"positions={self.num_positions}, "
            f"layers={len(self.kv_data)}, "
            f"model='{self.model_name}')"
        )
