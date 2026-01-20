"""
Novelty metrics for KV-cache analysis.

This module implements multiple operationalizations of "novelty" to measure
how different a KV entry is from recent history. These metrics are designed
to triangulate the phenomenon of "thinking turns" from different perspectives.

Metric Categories:
1. Geometric: Distance-based measures in embedding space
2. Attention: Changes in attention patterns
3. Predictive: Uncertainty and surprise measures
4. Temporal: Dynamics and structure over time
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.linalg import norm, svd, inv


@dataclass
class NoveltyMetrics:
    """Collection of novelty metrics for a sequence."""

    # Geometric metrics (per-position arrays)
    cosine_novelty: np.ndarray | None = None
    l2_novelty: np.ndarray | None = None
    projection_residual: np.ndarray | None = None
    mahalanobis: np.ndarray | None = None
    per_head_variance: np.ndarray | None = None

    # Attention-based metrics
    attention_entropy_change: np.ndarray | None = None
    attention_kl_divergence: np.ndarray | None = None

    # Predictive metrics (from token data)
    token_logprobs: np.ndarray | None = None
    token_entropy: np.ndarray | None = None
    entropy_gradient: np.ndarray | None = None

    # Temporal structure
    burstiness_index: float | None = None
    autocorrelation: np.ndarray | None = None

    # Combined/aggregate
    combined_novelty: np.ndarray | None = None

    # Metadata
    positions: np.ndarray | None = None
    tokens: list[str] | None = None

    def to_dict(self) -> dict[str, list | float | None]:
        """Convert to dictionary with lists instead of arrays."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
            else:
                result[key] = value
        return result


# =============================================================================
# Geometric Metrics
# =============================================================================

def cosine_novelty(
    kv_sequence: np.ndarray,
    window_size: int = 16,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Compute cosine distance from the mean of recent KV entries.

    Args:
        kv_sequence: Array of shape [num_positions, num_heads, head_size]
                    or [num_positions, hidden_size]
        window_size: Number of recent positions to compute mean from
        eps: Small value for numerical stability

    Returns:
        Array of shape [num_positions] with novelty scores
    """
    n_positions = kv_sequence.shape[0]

    # Flatten if necessary
    if kv_sequence.ndim == 3:
        kv_flat = kv_sequence.reshape(n_positions, -1)
    else:
        kv_flat = kv_sequence

    novelty = np.zeros(n_positions)

    for t in range(n_positions):
        if t == 0:
            novelty[t] = 0.0  # First position has no history
            continue

        # Get window of recent history
        start = max(0, t - window_size)
        history = kv_flat[start:t]

        # Compute mean of history
        mean_history = history.mean(axis=0)

        # Compute cosine distance
        current = kv_flat[t]
        cos_sim = np.dot(current, mean_history) / (
            norm(current) * norm(mean_history) + eps
        )
        novelty[t] = 1.0 - cos_sim

    return novelty


def l2_novelty(
    kv_sequence: np.ndarray,
    window_size: int = 16,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute L2 distance from the mean of recent KV entries.

    Args:
        kv_sequence: Array of shape [num_positions, num_heads, head_size]
                    or [num_positions, hidden_size]
        window_size: Number of recent positions
        normalize: Whether to normalize by dimension

    Returns:
        Array of shape [num_positions] with novelty scores
    """
    n_positions = kv_sequence.shape[0]

    if kv_sequence.ndim == 3:
        kv_flat = kv_sequence.reshape(n_positions, -1)
    else:
        kv_flat = kv_sequence

    dim = kv_flat.shape[1]
    novelty = np.zeros(n_positions)

    for t in range(n_positions):
        if t == 0:
            novelty[t] = 0.0
            continue

        start = max(0, t - window_size)
        history = kv_flat[start:t]
        mean_history = history.mean(axis=0)

        dist = norm(kv_flat[t] - mean_history)
        if normalize:
            dist /= np.sqrt(dim)
        novelty[t] = dist

    return novelty


def projection_residual_novelty(
    kv_sequence: np.ndarray,
    window_size: int = 16,
    n_components: int | None = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Compute the projection residual onto the subspace of recent KV history.

    This measures how much of the new KV entry lies outside the subspace
    spanned by recent entries (via SVD).

    Args:
        kv_sequence: Array of shape [num_positions, num_heads, head_size]
                    or [num_positions, hidden_size]
        window_size: Number of recent positions for subspace
        n_components: Number of SVD components (None = use window_size)
        eps: Small value for numerical stability

    Returns:
        Array of shape [num_positions] with novelty scores (0-1)
    """
    n_positions = kv_sequence.shape[0]

    if kv_sequence.ndim == 3:
        kv_flat = kv_sequence.reshape(n_positions, -1)
    else:
        kv_flat = kv_sequence

    novelty = np.zeros(n_positions)

    for t in range(n_positions):
        if t < 2:  # Need at least 2 points for SVD
            novelty[t] = 0.0
            continue

        start = max(0, t - window_size)
        history = kv_flat[start:t]

        # Center the history
        mean_history = history.mean(axis=0)
        history_centered = history - mean_history

        # SVD to get principal components
        try:
            U, S, Vt = svd(history_centered, full_matrices=False)

            # Determine number of components
            if n_components is None:
                k = min(len(S), window_size)
            else:
                k = min(n_components, len(S))

            # Keep only top k components
            V_k = Vt[:k].T  # Shape: [hidden_dim, k]

            # Project current vector onto subspace
            current_centered = kv_flat[t] - mean_history
            projection = V_k @ (V_k.T @ current_centered)

            # Compute residual
            residual = current_centered - projection
            residual_norm = norm(residual)
            original_norm = norm(current_centered)

            novelty[t] = residual_norm / (original_norm + eps)

        except np.linalg.LinAlgError:
            novelty[t] = 0.0

    return novelty


def mahalanobis_novelty(
    kv_sequence: np.ndarray,
    window_size: int = 32,
    regularization: float = 1e-5,
) -> np.ndarray:
    """
    Compute Mahalanobis distance from running distribution.

    This accounts for the covariance structure of recent KV entries.

    Args:
        kv_sequence: Array of shape [num_positions, num_heads, head_size]
                    or [num_positions, hidden_size]
        window_size: Number of recent positions for covariance estimation
        regularization: Regularization for covariance matrix inversion

    Returns:
        Array of shape [num_positions] with novelty scores
    """
    n_positions = kv_sequence.shape[0]

    if kv_sequence.ndim == 3:
        kv_flat = kv_sequence.reshape(n_positions, -1)
    else:
        kv_flat = kv_sequence

    dim = kv_flat.shape[1]
    novelty = np.zeros(n_positions)

    # For high-dimensional data, use a simplified approach
    # (diagonal covariance or PCA-based)
    use_diagonal = dim > 256

    for t in range(n_positions):
        if t < 3:  # Need enough points for covariance
            novelty[t] = 0.0
            continue

        start = max(0, t - window_size)
        history = kv_flat[start:t]

        mean = history.mean(axis=0)
        diff = kv_flat[t] - mean

        if use_diagonal:
            # Use diagonal covariance (independent dimensions)
            var = history.var(axis=0) + regularization
            mahal_sq = np.sum(diff**2 / var)
        else:
            # Full covariance
            cov = np.cov(history.T)
            if cov.ndim == 0:
                cov = np.array([[cov]])
            cov += regularization * np.eye(dim)
            try:
                cov_inv = inv(cov)
                mahal_sq = diff @ cov_inv @ diff
            except np.linalg.LinAlgError:
                # Fallback to diagonal
                var = history.var(axis=0) + regularization
                mahal_sq = np.sum(diff**2 / var)

        novelty[t] = np.sqrt(max(0, mahal_sq))

    return novelty


def per_head_novelty_variance(
    kv_sequence: np.ndarray,
    window_size: int = 16,
) -> np.ndarray:
    """
    Compute variance of novelty across attention heads.

    High variance suggests disagreement among heads about whether
    this is a novel moment.

    Args:
        kv_sequence: Array of shape [num_positions, num_heads, head_size]
        window_size: Number of recent positions

    Returns:
        Array of shape [num_positions] with variance scores
    """
    if kv_sequence.ndim != 3:
        raise ValueError("per_head_novelty_variance requires 3D input")

    n_positions, n_heads, head_size = kv_sequence.shape
    per_head_novelty = np.zeros((n_positions, n_heads))

    # Compute cosine novelty for each head separately
    for h in range(n_heads):
        head_seq = kv_sequence[:, h, :]  # [n_positions, head_size]
        per_head_novelty[:, h] = cosine_novelty(head_seq, window_size)

    # Compute variance across heads
    variance = per_head_novelty.var(axis=1)
    return variance


# =============================================================================
# Predictive/Uncertainty Metrics
# =============================================================================

def entropy_from_logprobs(logprobs: np.ndarray) -> float:
    """Compute entropy from log probabilities."""
    probs = np.exp(logprobs)
    probs = probs / probs.sum()  # Normalize
    return -np.sum(probs * np.log(probs + 1e-10))


def compute_entropy_gradient(entropies: np.ndarray) -> np.ndarray:
    """Compute first difference of entropy sequence."""
    gradient = np.zeros_like(entropies)
    gradient[1:] = entropies[1:] - entropies[:-1]
    return gradient


def inverse_logprob_novelty(logprobs: np.ndarray) -> np.ndarray:
    """Convert logprobs to surprise/novelty (lower logprob = higher novelty)."""
    return -logprobs


# =============================================================================
# Temporal Structure Metrics
# =============================================================================

def compute_burstiness_index(spike_times: np.ndarray) -> float:
    """
    Compute burstiness index from inter-spike intervals.

    B = (sigma - mu) / (sigma + mu)

    B > 0: bursty (clustered spikes)
    B < 0: regular (evenly spaced)
    B ~ 0: Poisson (random)

    Args:
        spike_times: Array of positions where spikes occurred

    Returns:
        Burstiness index in range [-1, 1]
    """
    if len(spike_times) < 2:
        return 0.0

    intervals = np.diff(spike_times)
    if len(intervals) == 0:
        return 0.0

    mu = intervals.mean()
    sigma = intervals.std()

    if mu + sigma == 0:
        return 0.0

    return (sigma - mu) / (sigma + mu)


def compute_autocorrelation(
    signal: np.ndarray,
    max_lag: int = 50,
) -> np.ndarray:
    """
    Compute autocorrelation of a signal.

    Args:
        signal: 1D array
        max_lag: Maximum lag to compute

    Returns:
        Autocorrelation values for lags 0 to max_lag
    """
    n = len(signal)
    max_lag = min(max_lag, n - 1)

    # Normalize
    signal = signal - signal.mean()
    var = signal.var()
    if var == 0:
        return np.zeros(max_lag + 1)

    autocorr = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        autocorr[lag] = np.sum(signal[:n-lag] * signal[lag:]) / ((n - lag) * var)

    return autocorr


def detect_spikes(
    signal: np.ndarray,
    threshold_std: float = 2.0,
    min_separation: int = 3,
) -> np.ndarray:
    """
    Detect spikes in a novelty signal.

    Args:
        signal: 1D novelty array
        threshold_std: Number of standard deviations above mean
        min_separation: Minimum positions between spikes

    Returns:
        Array of spike positions
    """
    mean = signal.mean()
    std = signal.std()
    threshold = mean + threshold_std * std

    # Find positions above threshold
    above_threshold = np.where(signal > threshold)[0]

    if len(above_threshold) == 0:
        return np.array([])

    # Filter for minimum separation
    spikes = [above_threshold[0]]
    for pos in above_threshold[1:]:
        if pos - spikes[-1] >= min_separation:
            spikes.append(pos)

    return np.array(spikes)


# =============================================================================
# Combined Analysis
# =============================================================================

def compute_all_novelty_metrics(
    keys: np.ndarray | None = None,
    values: np.ndarray | None = None,
    logprobs: np.ndarray | None = None,
    entropies: np.ndarray | None = None,
    tokens: list[str] | None = None,
    window_size: int = 16,
    compute_expensive: bool = False,
) -> NoveltyMetrics:
    """
    Compute all novelty metrics from collected data.

    Args:
        keys: Key tensors, shape [num_positions, num_heads, head_size]
        values: Value tensors, shape [num_positions, num_heads, head_size]
        logprobs: Token log probabilities
        entropies: Token entropies
        tokens: Token strings
        window_size: Window size for geometric metrics
        compute_expensive: Whether to compute expensive metrics (Mahalanobis)

    Returns:
        NoveltyMetrics dataclass with all computed metrics
    """
    metrics = NoveltyMetrics()

    # Use keys if available, otherwise values
    kv = keys if keys is not None else values

    if kv is not None:
        n_positions = kv.shape[0]
        metrics.positions = np.arange(n_positions)

        # Geometric metrics on keys
        metrics.cosine_novelty = cosine_novelty(kv, window_size)
        metrics.l2_novelty = l2_novelty(kv, window_size)
        metrics.projection_residual = projection_residual_novelty(kv, window_size)

        if compute_expensive:
            metrics.mahalanobis = mahalanobis_novelty(kv, window_size * 2)

        # Per-head variance (if 3D)
        if kv.ndim == 3:
            metrics.per_head_variance = per_head_novelty_variance(kv, window_size)

        # Temporal structure
        if metrics.cosine_novelty is not None:
            spikes = detect_spikes(metrics.cosine_novelty)
            metrics.burstiness_index = compute_burstiness_index(spikes)
            metrics.autocorrelation = compute_autocorrelation(metrics.cosine_novelty)

    # Predictive metrics
    if logprobs is not None:
        metrics.token_logprobs = logprobs

    if entropies is not None:
        metrics.token_entropy = entropies
        metrics.entropy_gradient = compute_entropy_gradient(entropies)

    metrics.tokens = tokens

    # Combined novelty score (weighted average of normalized metrics)
    if metrics.cosine_novelty is not None:
        combined = normalize_metric(metrics.cosine_novelty)

        if metrics.projection_residual is not None:
            combined = 0.5 * combined + 0.5 * normalize_metric(metrics.projection_residual)

        if metrics.token_logprobs is not None:
            # Align lengths if needed
            min_len = min(len(combined), len(metrics.token_logprobs))
            combined = combined[:min_len]
            surprise = normalize_metric(-metrics.token_logprobs[:min_len])
            combined = 0.7 * combined + 0.3 * surprise

        metrics.combined_novelty = combined

    return metrics


def normalize_metric(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize a metric to [0, 1] range."""
    min_val = arr.min()
    max_val = arr.max()
    if max_val - min_val < eps:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)


# =============================================================================
# Statistical Tests
# =============================================================================

def test_heavy_tail(
    data: np.ndarray,
    threshold_percentile: float = 90,
) -> dict:
    """
    Test if the data has a heavy-tailed distribution.

    Returns statistics about tail behavior.
    """
    threshold = np.percentile(data, threshold_percentile)
    tail = data[data > threshold]

    # Compute statistics
    mean = data.mean()
    std = data.std()
    skewness = ((data - mean) ** 3).mean() / (std ** 3 + 1e-10)
    kurtosis = ((data - mean) ** 4).mean() / (std ** 4 + 1e-10) - 3  # Excess kurtosis

    return {
        "mean": float(mean),
        "std": float(std),
        "skewness": float(skewness),
        "kurtosis": float(kurtosis),  # Positive = heavy tails
        "threshold": float(threshold),
        "tail_fraction": len(tail) / len(data),
        "tail_mean": float(tail.mean()) if len(tail) > 0 else 0.0,
        "is_heavy_tailed": kurtosis > 0 and skewness > 0,
    }


def test_uniformity(
    data: np.ndarray,
    n_bins: int = 20,
) -> dict:
    """
    Test if the data is uniformly distributed (would weaken the hypothesis).

    Uses chi-square test against uniform distribution.
    """
    from scipy import stats

    # Histogram
    hist, bin_edges = np.histogram(data, bins=n_bins)
    expected = len(data) / n_bins

    # Chi-square test
    chi2_stat = np.sum((hist - expected) ** 2 / expected)
    p_value = 1 - stats.chi2.cdf(chi2_stat, n_bins - 1)

    return {
        "chi2_statistic": float(chi2_stat),
        "p_value": float(p_value),
        "is_uniform": p_value > 0.05,  # Fail to reject uniformity
        "histogram": hist.tolist(),
    }
