"""
KV-Cache Novelty Analysis Framework

This module provides tools for studying the temporal dynamics of KV-cache
informativeness during Transformer inference. It implements multiple novelty
metrics and provides instrumentation for vLLM's attention layers.

NOTE: Research suggests focusing on models >= 7B parameters, as smaller
models show qualitatively different novelty patterns (punctuation-dominated
vs semantic-dominated spikes).
"""

from .collector import KVNoveltyCollector
from .hooks import install_kv_hooks, remove_kv_hooks
from .metrics import (
    cosine_novelty,
    projection_residual_novelty,
    mahalanobis_novelty,
    compute_all_novelty_metrics,
    detect_spikes,
    compute_burstiness_index,
    test_heavy_tail,
)
from .kv_capture import (
    OnlineKVNoveltyTracker,
    KVCaptureHooks,
    TokenNoveltyRecord,
    KVSnapshot,
    compute_kv_ablation_impact,
)
from .kv_experiments import (
    KVExperimentConfig,
    KVExperimentResult,
    run_kv_experiment,
    print_experiment_summary,
)
from .kv_compression import (
    CompressionConfig,
    CompressionResult,
    run_compression_experiment,
    print_compression_results,
)
from .kv_window_compression import (
    WindowCompressionConfig,
    WindowCompressionResult,
    CompressionMethod,
    KVWindowCompressor,
    run_window_compression_experiment,
    print_window_compression_results,
)

__all__ = [
    # Original collector/hooks
    "KVNoveltyCollector",
    "install_kv_hooks",
    "remove_kv_hooks",
    # Metrics
    "cosine_novelty",
    "projection_residual_novelty",
    "mahalanobis_novelty",
    "compute_all_novelty_metrics",
    "detect_spikes",
    "compute_burstiness_index",
    "test_heavy_tail",
    # Direct KV capture
    "OnlineKVNoveltyTracker",
    "KVCaptureHooks",
    "TokenNoveltyRecord",
    "KVSnapshot",
    "compute_kv_ablation_impact",
    # Experiments
    "KVExperimentConfig",
    "KVExperimentResult",
    "run_kv_experiment",
    "print_experiment_summary",
    # Compression
    "CompressionConfig",
    "CompressionResult",
    "run_compression_experiment",
    "print_compression_results",
    # Window compression
    "WindowCompressionConfig",
    "WindowCompressionResult",
    "CompressionMethod",
    "KVWindowCompressor",
    "run_window_compression_experiment",
    "print_window_compression_results",
]

__version__ = "0.2.0"
