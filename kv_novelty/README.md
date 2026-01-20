# KV-Cache Novelty Analysis Framework

A research framework for studying the temporal dynamics of KV-cache informativeness during Transformer inference.

## Overview

This framework tests the hypothesis that **KV-cache entries are not uniformly informative** during autoregressive generation. We hypothesize that:

1. Most tokens correspond to **low-novelty KV updates** (model "coasting" in local continuation)
2. Occasionally, the model undergoes **"thinking turns"** with high-novelty spikes
3. Novelty is **sparse, bursty, and structured** in time

## Installation

The framework requires:
```bash
pip install torch transformers numpy scipy matplotlib
```

For vLLM backend (optional):
```bash
pip install vllm
```

## Quick Start

### Run a single experiment
```bash
cd /path/to/vllm
python -m kv_novelty.run_experiments --model gpt2 --max-tokens 100
```

### Run with Llama
```bash
python -m kv_novelty.run_experiments \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --max-tokens 512 \
    --temperature 0.7
```

### Run full experiment suite
```bash
python -m kv_novelty.run_experiments --suite --model meta-llama/Llama-3.1-8B-Instruct
```

## Module Structure

```
kv_novelty/
├── __init__.py          # Package exports
├── collector.py         # Data collection and storage
├── metrics.py           # Novelty metric implementations
├── analysis.py          # Analysis and visualization
├── hooks.py             # PyTorch hooks for KV capture
├── llama_hooks.py       # Specialized hooks for Llama models
├── experiments.py       # Experiment configuration and runner
└── run_experiments.py   # CLI entry point
```

## Key Classes

### KVNoveltyCollector
Collects KV-cache data during generation:
```python
from kv_novelty import KVNoveltyCollector

collector = KVNoveltyCollector(
    capture_kv=True,
    layers_to_capture=[0, 8, 16, 24, 31],  # Sample layers
)
```

### Metrics
```python
from kv_novelty.metrics import (
    cosine_novelty,
    projection_residual_novelty,
    compute_all_novelty_metrics,
)

# Compute all metrics
metrics = compute_all_novelty_metrics(
    keys=captured_keys,      # [num_positions, num_heads, head_dim]
    logprobs=token_logprobs,
    entropies=token_entropies,
    tokens=token_strings,
)
```

### Analysis
```python
from kv_novelty.analysis import analyze_collection, generate_text_report

result = analyze_collection(collector)
report = generate_text_report(result)
print(report)
```

## Metrics Implemented

### Geometric Metrics
- **Cosine Novelty**: Cosine distance from recent KV mean
- **L2 Novelty**: Euclidean distance from recent mean
- **Projection Residual**: Fraction outside recent history subspace
- **Mahalanobis Novelty**: Statistical distance with covariance

### Predictive Metrics
- **Token Log-Probability**: Model surprise (lower = more novel)
- **Token Entropy**: Output distribution entropy
- **Entropy Gradient**: Change in entropy

### Temporal Metrics
- **Burstiness Index**: Clustered vs. uniform spike pattern
- **Autocorrelation**: Temporal persistence
- **Heavy-tail Tests**: Kurtosis, skewness

## Experiment Prompts

The framework includes prompts designed to test different content types:

| Prompt Type | Description |
|-------------|-------------|
| `prose_explanation` | Pure explanatory text |
| `code_generation` | Pure code output |
| `mixed_content` | Explanation with code examples |
| `reasoning_chain` | Step-by-step reasoning |
| `transition_heavy` | Multiple semantic transitions |

## Output Structure

```
experiments/
├── custom/
│   ├── collected_data/
│   │   ├── metadata.json
│   │   └── kv_layer_*.npz
│   ├── analysis.json
│   ├── report.txt
│   ├── generated.txt
│   └── plots/
│       ├── novelty_timeseries.png
│       ├── novelty_distribution.png
│       └── spike_analysis.png
└── comparison.json
```

## Interpreting Results

### Supporting the Hypothesis
- **Heavy-tailed distribution** (kurtosis > 0, positive skew)
- **Burstiness index > 0** (clustered spikes)
- **Spikes at semantic transitions** (code↔prose, topic shifts)
- **High autocorrelation** (structured, not random)

### Weakening the Hypothesis
- **Uniform distribution** (p > 0.05 in uniformity test)
- **Burstiness ≈ 0** (Poisson-like, random)
- **Spikes dominated by trivial tokens** (punctuation, whitespace)

## Example Analysis Output

```
RESULTS SUMMARY
----------------
Tokens: 512
Spikes detected: 23
Spike rate: 0.045
Burstiness index: 0.342
Heavy-tailed: True
Kurtosis: 2.847
```

## Documentation

See `.docs/` directory for detailed documentation:
- `01_research_design.md` - Research hypothesis and experimental design
- `02_kv_cache_internals.md` - vLLM KV-cache implementation details
- `03_metric_definitions.md` - Formal metric definitions

## Limitations

1. **Hidden states as proxy**: Uses hidden states, not actual K/V tensors
2. **HuggingFace only**: Full capture requires transformers (not vLLM)
3. **Memory**: Large models with full capture require significant memory
4. **Causal metrics**: Ablation studies not yet implemented

## Future Work

- [ ] True KV tensor capture via model patching
- [ ] Causal ablation experiments
- [ ] Cross-model comparison
- [ ] Temperature sensitivity analysis
- [ ] Attention pattern analysis
- [ ] vLLM native integration
