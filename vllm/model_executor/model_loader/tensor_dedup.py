# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Process-wide tensor deduplication utilities."""

from __future__ import annotations

import hashlib
import logging
import threading
from collections.abc import Generator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

try:
    import xxhash  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    xxhash = None

if TYPE_CHECKING:
    from vllm.config.load import TensorDedupConfig


@dataclass(frozen=True)
class TensorFingerprint:
    """Uniquely identifies a tensor by metadata + hash."""

    shape: tuple[int, ...]
    dtype: str
    digest: str


class TensorDedupRegistry:
    """Process-wide cache of tensors indexed by TensorFingerprint."""

    _lock = threading.Lock()
    _cache: dict[TensorFingerprint, torch.Tensor] = {}
    _registered = 0
    _reused = 0
    _saved_bytes = 0

    @classmethod
    def register(
        cls, fingerprint: TensorFingerprint, tensor: torch.Tensor
    ) -> None:
        with cls._lock:
            cls._cache[fingerprint] = tensor
            cls._registered += 1

    @classmethod
    def get(cls, fingerprint: TensorFingerprint) -> torch.Tensor | None:
        with cls._lock:
            return cls._cache.get(fingerprint)

    @classmethod
    def record_reuse(cls, bytes_saved: int) -> None:
        with cls._lock:
            cls._reused += 1
            cls._saved_bytes += bytes_saved

    @classmethod
    def clear(cls) -> None:
        with cls._lock:
            cls._cache.clear()
            cls._registered = 0
            cls._reused = 0
            cls._saved_bytes = 0

    @classmethod
    def get_stats(cls) -> dict[str, float]:
        with cls._lock:
            total_bytes = sum(_tensor_size_bytes(t) for t in cls._cache.values())
            return {
                "entries": len(cls._cache),
                "total_bytes": total_bytes,
                "registered": cls._registered,
                "reused": cls._reused,
                "saved_bytes": cls._saved_bytes,
            }


def wrap_tensor_dedup(
    iterator: Generator[tuple[str, torch.Tensor], None, None],
    logger: logging.Logger,
    tensor_dedup: TensorDedupConfig | None = None,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Apply deduplication to tensors yielded by an iterator."""

    dedup_enabled = bool(tensor_dedup and tensor_dedup.enabled)
    if not dedup_enabled:
        yield from iterator
        return

    assert tensor_dedup is not None  # help type-checkers
    logger.info(
        "[TensorDedup] Enabled (hash=%s, verify=%s, min_bytes=%d)",
        tensor_dedup.hash_algorithm or "blake2b",
        tensor_dedup.verify_bytes,
        tensor_dedup.min_tensor_bytes,
    )

    registered = 0
    reused = 0
    saved_mb = 0.0

    for tensor_name, tensor in iterator:
        tensor_bytes = _tensor_size_bytes(tensor)
        if tensor_bytes < max(tensor_dedup.min_tensor_bytes, 0):
            yield tensor_name, tensor
            continue

        raw_bytes = _tensor_to_bytes(tensor, logger)
        digest = _hash_bytes(raw_bytes, tensor_dedup.hash_algorithm, logger)
        fingerprint = TensorFingerprint(
            shape=tuple(int(dim) for dim in tensor.shape),
            dtype=str(tensor.dtype),
            digest=digest,
        )

        cached = TensorDedupRegistry.get(fingerprint)
        if cached is not None:
            if tensor_dedup.verify_bytes:
                cached_bytes = _tensor_to_bytes(cached, logger)
                if cached_bytes != raw_bytes:
                    logger.warning(
                        "[TensorDedup] Hash collision detected for %s (shape=%s, dtype=%s)",
                        tensor_name,
                        tuple(tensor.shape),
                        tensor.dtype,
                    )
                    yield tensor_name, tensor
                    continue
            reused += 1
            saved_mb += _tensor_size_mb(cached)
            TensorDedupRegistry.record_reuse(_tensor_size_bytes(cached))
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[TensorDedup] REUSED: %s (%.2f MB saved, hash=%s)",
                    tensor_name,
                    _tensor_size_mb(cached),
                    digest[:12],
                )
            yield tensor_name, cached
            continue

        TensorDedupRegistry.register(fingerprint, tensor)
        registered += 1
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[TensorDedup] REGISTERED: %s (%.2f MB, hash=%s)",
                tensor_name,
                _tensor_size_mb(tensor),
                digest[:12],
            )
        yield tensor_name, tensor

    logger.info(
        "[TensorDedup] SUMMARY: Registered %d tensors, reused %d tensors (%.2f MB saved)",
        registered,
        reused,
        saved_mb,
    )


def _tensor_size_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _tensor_size_mb(tensor: torch.Tensor) -> float:
    return _tensor_size_bytes(tensor) / (1024**2)


def _tensor_to_bytes(
    tensor: torch.Tensor, logger: logging.Logger | None = None
) -> bytes:
    tensor_cpu = tensor.detach()
    if tensor_cpu.device.type != "cpu":
        tensor_cpu = tensor_cpu.cpu()
    tensor_cpu = tensor_cpu.contiguous()
    try:
        np_array = tensor_cpu.numpy()
    except TypeError:
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[TensorDedup] Falling back to float32 copy for dtype=%s", tensor.dtype
            )
        np_array = tensor_cpu.to(torch.float32).numpy()
    return np_array.tobytes()


def _hash_bytes(data: bytes, algorithm: str | None, logger: logging.Logger) -> str:
    algo = (algorithm or "blake2b").lower()
    if algo == "xxhash64":
        if xxhash is None:
            logger.warning(
                "[TensorDedup] Hash algorithm xxhash64 requested but python-xxhash "
                "is not installed. Falling back to blake2b."
            )
            algo = "blake2b"
        else:
            hasher = xxhash.xxh64()
            hasher.update(data)
            return hasher.hexdigest()

    try:
        hasher = hashlib.new(algo)
    except ValueError:
        logger.warning(
            "[TensorDedup] Unknown hash algorithm '%s'; falling back to blake2b.",
            algorithm,
        )
        hasher = hashlib.blake2b()
    hasher.update(data)
    return hasher.hexdigest()

