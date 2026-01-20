"""
Analysis and visualization tools for KV-cache novelty research.

This module provides functions to analyze collected data, generate
visualizations, and produce reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import numpy as np

from .collector import KVNoveltyCollector
from .metrics import (
    NoveltyMetrics,
    compute_all_novelty_metrics,
    detect_spikes,
    test_heavy_tail,
    test_uniformity,
    compute_burstiness_index,
)


@dataclass
class AnalysisResult:
    """Results from analyzing a single generation."""

    # Basic info
    model_name: str
    prompt: str
    num_tokens: int
    generated_text: str

    # Metrics per layer
    layer_metrics: dict[int, NoveltyMetrics]

    # Aggregate statistics
    aggregate_stats: dict[str, Any]

    # Spike analysis
    spike_positions: np.ndarray
    spike_tokens: list[str]
    spike_contexts: list[str]

    # Distribution tests
    heavy_tail_test: dict[str, Any]
    uniformity_test: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        def convert_numpy(obj):
            """Convert numpy types to Python types for JSON serialization."""
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        return convert_numpy({
            "model_name": self.model_name,
            "prompt": self.prompt,
            "num_tokens": self.num_tokens,
            "generated_text": self.generated_text,
            "aggregate_stats": self.aggregate_stats,
            "spike_positions": self.spike_positions.tolist(),
            "spike_tokens": self.spike_tokens,
            "spike_contexts": self.spike_contexts,
            "heavy_tail_test": self.heavy_tail_test,
            "uniformity_test": self.uniformity_test,
            "layer_metrics": {
                str(k): v.to_dict() for k, v in self.layer_metrics.items()
            },
        })


def analyze_collection(
    collector: KVNoveltyCollector,
    layers_to_analyze: list[int] | None = None,
    window_size: int = 16,
    spike_threshold_std: float = 2.0,
    context_window: int = 5,
) -> AnalysisResult:
    """
    Analyze collected KV data and compute novelty metrics.

    Args:
        collector: The data collector with captured KV tensors
        layers_to_analyze: Specific layers to analyze (None = all)
        window_size: Window size for novelty computation
        spike_threshold_std: Std threshold for spike detection
        context_window: Number of tokens around spikes to include

    Returns:
        AnalysisResult with all computed metrics and statistics
    """
    layers = layers_to_analyze or collector.captured_layers
    tokens = collector.get_tokens()
    logprobs = collector.get_logprobs()
    entropies = collector.get_entropies()

    # Compute metrics for each layer
    layer_metrics = {}
    all_novelty = []

    for layer_idx in layers:
        keys = collector.get_keys_for_layer(layer_idx)
        values = collector.get_values_for_layer(layer_idx)

        if keys is None and values is None:
            continue

        metrics = compute_all_novelty_metrics(
            keys=keys,
            values=values,
            logprobs=logprobs,
            entropies=entropies,
            tokens=tokens,
            window_size=window_size,
        )
        layer_metrics[layer_idx] = metrics

        if metrics.cosine_novelty is not None:
            all_novelty.append(metrics.cosine_novelty)

    # Aggregate novelty across layers
    if all_novelty:
        mean_novelty = np.mean(all_novelty, axis=0)
    else:
        mean_novelty = np.array([])

    # Detect spikes in aggregate novelty
    if len(mean_novelty) > 0:
        spike_positions = detect_spikes(mean_novelty, spike_threshold_std)
    else:
        spike_positions = np.array([])

    # Get spike tokens and contexts
    spike_tokens = []
    spike_contexts = []

    for pos in spike_positions:
        pos = int(pos)
        if pos < len(tokens):
            spike_tokens.append(tokens[pos])

            # Get context
            start = max(0, pos - context_window)
            end = min(len(tokens), pos + context_window + 1)
            context = "".join(tokens[start:end])
            spike_contexts.append(f"[{start}:{end}] {context}")

    # Compute aggregate statistics
    aggregate_stats = {}
    if len(mean_novelty) > 0:
        aggregate_stats = {
            "mean_novelty": float(mean_novelty.mean()),
            "std_novelty": float(mean_novelty.std()),
            "max_novelty": float(mean_novelty.max()),
            "min_novelty": float(mean_novelty.min()),
            "num_spikes": len(spike_positions),
            "spike_rate": len(spike_positions) / len(mean_novelty),
            "burstiness_index": compute_burstiness_index(spike_positions),
        }

        if len(logprobs) > 0:
            aggregate_stats["mean_logprob"] = float(logprobs.mean())
            aggregate_stats["std_logprob"] = float(logprobs.std())

        if len(entropies) > 0:
            aggregate_stats["mean_entropy"] = float(entropies.mean())
            aggregate_stats["std_entropy"] = float(entropies.std())

    # Distribution tests
    heavy_tail_test = {}
    uniformity_test = {}
    if len(mean_novelty) > 10:
        heavy_tail_test = test_heavy_tail(mean_novelty)
        uniformity_test = test_uniformity(mean_novelty)

    return AnalysisResult(
        model_name=collector.model_name,
        prompt=collector.prompt,
        num_tokens=collector.num_positions,
        generated_text=collector.get_generated_text(),
        layer_metrics=layer_metrics,
        aggregate_stats=aggregate_stats,
        spike_positions=spike_positions,
        spike_tokens=spike_tokens,
        spike_contexts=spike_contexts,
        heavy_tail_test=heavy_tail_test,
        uniformity_test=uniformity_test,
    )


def generate_text_report(result: AnalysisResult) -> str:
    """Generate a text report from analysis results."""
    lines = []
    lines.append("=" * 80)
    lines.append("KV-CACHE NOVELTY ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append("")

    # Basic info
    lines.append(f"Model: {result.model_name}")
    lines.append(f"Tokens generated: {result.num_tokens}")
    lines.append(f"Prompt: {result.prompt[:100]}..." if len(result.prompt) > 100 else f"Prompt: {result.prompt}")
    lines.append("")

    # Aggregate statistics
    lines.append("-" * 40)
    lines.append("AGGREGATE STATISTICS")
    lines.append("-" * 40)
    for key, value in result.aggregate_stats.items():
        if isinstance(value, float):
            lines.append(f"  {key}: {value:.4f}")
        else:
            lines.append(f"  {key}: {value}")
    lines.append("")

    # Distribution tests
    lines.append("-" * 40)
    lines.append("DISTRIBUTION TESTS")
    lines.append("-" * 40)

    if result.heavy_tail_test:
        lines.append("Heavy-tail test:")
        lines.append(f"  Kurtosis: {result.heavy_tail_test.get('kurtosis', 0):.4f}")
        lines.append(f"  Skewness: {result.heavy_tail_test.get('skewness', 0):.4f}")
        lines.append(f"  Is heavy-tailed: {result.heavy_tail_test.get('is_heavy_tailed', False)}")

    if result.uniformity_test:
        lines.append("Uniformity test:")
        lines.append(f"  Chi-square statistic: {result.uniformity_test.get('chi2_statistic', 0):.4f}")
        lines.append(f"  p-value: {result.uniformity_test.get('p_value', 0):.4f}")
        lines.append(f"  Is uniform: {result.uniformity_test.get('is_uniform', False)}")
    lines.append("")

    # Spikes
    lines.append("-" * 40)
    lines.append(f"DETECTED SPIKES ({len(result.spike_positions)})")
    lines.append("-" * 40)
    for i, (pos, token, context) in enumerate(zip(
        result.spike_positions, result.spike_tokens, result.spike_contexts
    )):
        lines.append(f"  Spike {i+1} at position {int(pos)}:")
        lines.append(f"    Token: '{token}'")
        lines.append(f"    Context: {context}")
    lines.append("")

    # Generated text sample
    lines.append("-" * 40)
    lines.append("GENERATED TEXT (first 500 chars)")
    lines.append("-" * 40)
    lines.append(result.generated_text[:500])
    lines.append("")

    lines.append("=" * 80)
    return "\n".join(lines)


def save_analysis(result: AnalysisResult, path: str | Path) -> None:
    """Save analysis results to JSON file."""
    path = Path(path)
    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)


def generate_matplotlib_plots(
    result: AnalysisResult,
    output_dir: str | Path,
    layer_idx: int | None = None,
) -> list[str]:
    """
    Generate matplotlib visualizations.

    Args:
        result: Analysis result
        output_dir: Directory to save plots
        layer_idx: Specific layer to plot (None = use first available)

    Returns:
        List of saved plot paths
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not available, skipping plots")
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_plots = []

    # Get metrics for a layer
    if layer_idx is None:
        layer_idx = list(result.layer_metrics.keys())[0] if result.layer_metrics else None

    if layer_idx is None:
        return []

    metrics = result.layer_metrics[layer_idx]

    # Plot 1: Novelty time series
    if metrics.cosine_novelty is not None:
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        # Cosine novelty
        axes[0].plot(metrics.cosine_novelty, 'b-', alpha=0.7, linewidth=0.8)
        axes[0].set_ylabel('Cosine Novelty')
        axes[0].set_title(f'KV-Cache Novelty Over Time (Layer {layer_idx})')

        # Mark spikes
        for pos in result.spike_positions:
            axes[0].axvline(x=pos, color='r', alpha=0.3, linewidth=1)

        # Projection residual
        if metrics.projection_residual is not None:
            axes[1].plot(metrics.projection_residual, 'g-', alpha=0.7, linewidth=0.8)
            axes[1].set_ylabel('Projection Residual')
            for pos in result.spike_positions:
                axes[1].axvline(x=pos, color='r', alpha=0.3, linewidth=1)

        # Token entropy (if available)
        if metrics.token_entropy is not None:
            axes[2].plot(metrics.token_entropy, 'purple', alpha=0.7, linewidth=0.8)
            axes[2].set_ylabel('Token Entropy')
            axes[2].set_xlabel('Token Position')
            for pos in result.spike_positions:
                axes[2].axvline(x=pos, color='r', alpha=0.3, linewidth=1)

        plt.tight_layout()
        path = output_dir / "novelty_timeseries.png"
        plt.savefig(path, dpi=150)
        plt.close()
        saved_plots.append(str(path))

    # Plot 2: Novelty distribution
    if metrics.cosine_novelty is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Histogram
        axes[0].hist(metrics.cosine_novelty, bins=50, density=True, alpha=0.7, color='blue')
        axes[0].set_xlabel('Cosine Novelty')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Novelty Distribution')

        # Add statistics
        stats_text = f"Mean: {metrics.cosine_novelty.mean():.4f}\n"
        stats_text += f"Std: {metrics.cosine_novelty.std():.4f}\n"
        if result.heavy_tail_test:
            stats_text += f"Kurtosis: {result.heavy_tail_test.get('kurtosis', 0):.2f}\n"
            stats_text += f"Heavy-tailed: {result.heavy_tail_test.get('is_heavy_tailed', False)}"
        axes[0].text(0.95, 0.95, stats_text, transform=axes[0].transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Autocorrelation
        if metrics.autocorrelation is not None:
            axes[1].bar(range(len(metrics.autocorrelation)), metrics.autocorrelation, alpha=0.7)
            axes[1].set_xlabel('Lag')
            axes[1].set_ylabel('Autocorrelation')
            axes[1].set_title('Novelty Autocorrelation')
            axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)

        plt.tight_layout()
        path = output_dir / "novelty_distribution.png"
        plt.savefig(path, dpi=150)
        plt.close()
        saved_plots.append(str(path))

    # Plot 3: Spike context analysis
    if len(result.spike_positions) > 0 and metrics.cosine_novelty is not None:
        fig, ax = plt.subplots(figsize=(14, 6))

        # Plot novelty with spike markers
        ax.plot(metrics.cosine_novelty, 'b-', alpha=0.5, linewidth=0.8, label='Novelty')
        ax.scatter(result.spike_positions,
                  metrics.cosine_novelty[result.spike_positions.astype(int)],
                  c='red', s=50, zorder=5, label='Spikes')

        # Annotate spikes with tokens
        for i, (pos, token) in enumerate(zip(result.spike_positions, result.spike_tokens)):
            pos = int(pos)
            if i < 10:  # Limit annotations to avoid clutter
                ax.annotate(
                    repr(token)[:10],
                    (pos, metrics.cosine_novelty[pos]),
                    xytext=(5, 10),
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.8,
                )

        ax.set_xlabel('Token Position')
        ax.set_ylabel('Cosine Novelty')
        ax.set_title('Novelty Spikes with Token Labels')
        ax.legend()

        plt.tight_layout()
        path = output_dir / "spike_analysis.png"
        plt.savefig(path, dpi=150)
        plt.close()
        saved_plots.append(str(path))

    # Plot 4: Layer comparison (if multiple layers)
    if len(result.layer_metrics) > 1:
        fig, ax = plt.subplots(figsize=(14, 6))

        for idx, (layer_idx, lm) in enumerate(result.layer_metrics.items()):
            if lm.cosine_novelty is not None:
                ax.plot(lm.cosine_novelty, alpha=0.5, label=f'Layer {layer_idx}')

        ax.set_xlabel('Token Position')
        ax.set_ylabel('Cosine Novelty')
        ax.set_title('Novelty Across Layers')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        path = output_dir / "layer_comparison.png"
        plt.savefig(path, dpi=150)
        plt.close()
        saved_plots.append(str(path))

    return saved_plots


def compare_experiments(
    results: list[AnalysisResult],
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """
    Compare results across multiple experiments.

    Args:
        results: List of analysis results
        labels: Labels for each result

    Returns:
        Comparison statistics
    """
    if labels is None:
        labels = [f"Exp {i+1}" for i in range(len(results))]

    comparison = {
        "labels": labels,
        "num_tokens": [],
        "mean_novelty": [],
        "std_novelty": [],
        "num_spikes": [],
        "spike_rate": [],
        "burstiness": [],
        "is_heavy_tailed": [],
        "is_uniform": [],
    }

    for result in results:
        comparison["num_tokens"].append(int(result.num_tokens))
        comparison["mean_novelty"].append(float(result.aggregate_stats.get("mean_novelty", 0)))
        comparison["std_novelty"].append(float(result.aggregate_stats.get("std_novelty", 0)))
        comparison["num_spikes"].append(int(result.aggregate_stats.get("num_spikes", 0)))
        comparison["spike_rate"].append(float(result.aggregate_stats.get("spike_rate", 0)))
        comparison["burstiness"].append(float(result.aggregate_stats.get("burstiness_index", 0)))
        comparison["is_heavy_tailed"].append(bool(result.heavy_tail_test.get("is_heavy_tailed", False)))
        comparison["is_uniform"].append(bool(result.uniformity_test.get("is_uniform", False)))

    return comparison
