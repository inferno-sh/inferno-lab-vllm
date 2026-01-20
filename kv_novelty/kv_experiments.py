"""
Experiment runner for actual KV cache novelty analysis.

Runs generation with direct KV capture, computes online novelty,
and measures causal impact of spike tokens.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer

from .kv_capture import (
    OnlineKVNoveltyTracker,
    KVCaptureHooks,
    TokenNoveltyRecord,
    KVSnapshot,
    compute_kv_ablation_impact,
)


def convert_numpy(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    return obj


@dataclass
class KVExperimentConfig:
    """Configuration for a KV novelty experiment."""
    model_name: str
    prompt: str
    max_new_tokens: int = 200
    temperature: float = 0.7
    do_sample: bool = True

    # Novelty tracking
    ema_alpha: float = 0.1
    spike_threshold_std: float = 2.0
    top_k_heads: int = 3

    # Snapshot settings
    snapshot_window: int = 2
    snapshot_sample_rate: int = 50  # 1/N for non-spikes

    # Ablation settings
    ablation_sample_rate: int = 10  # Test impact on 1/N spikes
    ablation_type: str = "zero"

    # Layers to capture (None = all)
    layers_to_capture: list[int] | None = None


@dataclass
class KVExperimentResult:
    """Results from a KV novelty experiment."""
    config: dict
    num_tokens: int
    total_time: float

    # Aggregate statistics
    mean_novelty: float
    std_novelty: float
    max_novelty: float
    num_spikes: int
    spike_rate: float

    # Per-layer statistics
    layer_mean_novelty: dict[int, float]
    layer_spike_counts: dict[int, int]

    # Head-level statistics (which heads spike most)
    head_spike_counts: dict[str, int]  # "layer_head" -> count

    # Causal impact statistics
    num_ablations: int
    mean_delta_logprob: float
    mean_kl_divergence: float
    max_delta_logprob: float

    # Token records (compact)
    spike_tokens: list[dict]

    # Snapshot count
    num_snapshots: int


def run_kv_experiment(
    config: KVExperimentConfig,
    output_dir: Path | None = None,
    device: str = "cuda",
) -> KVExperimentResult:
    """
    Run a KV novelty experiment with actual K/V capture.

    Args:
        config: Experiment configuration
        output_dir: Optional directory to save results
        device: Device to use

    Returns:
        Experiment results
    """
    print(f"Loading model: {config.model_name}")
    start_time = time.time()

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Get model config
    model_config = model.config
    num_layers = getattr(model_config, "num_hidden_layers", 32)
    num_heads = getattr(model_config, "num_attention_heads", 32)
    num_kv_heads = getattr(model_config, "num_key_value_heads", num_heads)
    head_dim = getattr(model_config, "head_dim",
                       model_config.hidden_size // num_heads)

    print(f"Model: {num_layers} layers, {num_heads} heads ({num_kv_heads} KV heads), {head_dim} head_dim")

    # Determine layers to capture
    if config.layers_to_capture is None:
        # Sample layers: first, 1/4, 1/2, 3/4, last
        layers = [0, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1]
        layers = sorted(set(layers))
    else:
        layers = config.layers_to_capture

    print(f"Capturing layers: {layers}")

    # Initialize tracker
    tracker = OnlineKVNoveltyTracker(
        num_layers=len(layers),
        num_heads=num_kv_heads,
        head_dim=head_dim,
        ema_alpha=config.ema_alpha,
        spike_threshold_std=config.spike_threshold_std,
        top_k_heads=config.top_k_heads,
        device=device,
    )

    # Tokenize prompt
    inputs = tokenizer(config.prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    print(f"Generating {config.max_new_tokens} tokens...")

    # We'll use a manual generation loop to capture K/V at each step
    token_records: list[TokenNoveltyRecord] = []
    snapshots: list[KVSnapshot] = []
    ablation_results: list[dict] = []

    generated_ids = inputs["input_ids"].clone()
    past_key_values = None

    # Layer index mapping (since we're only capturing some layers)
    layer_map = {l: i for i, l in enumerate(layers)}

    for step in range(config.max_new_tokens):
        with torch.no_grad():
            # Forward pass
            outputs = model(
                generated_ids if past_key_values is None else generated_ids[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                output_hidden_states=False,
            )

            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

            # Sample next token
            if config.do_sample:
                probs = F.softmax(logits / config.temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            # Extract K/V from cache for captured layers
            if past_key_values is not None:
                k_list = []
                v_list = []

                for layer_idx in layers:
                    if layer_idx < len(past_key_values):
                        key, value = past_key_values[layer_idx]
                        # key/value: [batch, num_kv_heads, seq_len, head_dim]
                        k_last = key[0, :, -1, :].detach()  # [num_kv_heads, head_dim]
                        v_last = value[0, :, -1, :].detach()
                        k_list.append(k_last)
                        v_list.append(v_last)

                if k_list:
                    k_stack = torch.stack(k_list)  # [num_captured_layers, num_kv_heads, head_dim]
                    v_stack = torch.stack(v_list)

                    # Compute novelty
                    k_novelty, v_novelty = tracker.compute_novelty(k_stack, v_stack)
                    combined_novelty = torch.maximum(k_novelty, v_novelty)

                    # Decode token
                    token_id = next_token[0, 0].item()
                    token_text = tokenizer.decode([token_id])

                    # Build record
                    layer_novelty = {}
                    layer_top_heads = {}

                    for i, layer_idx in enumerate(layers):
                        layer_nov = combined_novelty[i]
                        layer_novelty[layer_idx] = layer_nov.mean().item()

                        # Top-k heads
                        top_k = min(config.top_k_heads, num_kv_heads)
                        top_vals, top_idxs = torch.topk(layer_nov, top_k)
                        layer_top_heads[layer_idx] = [
                            {"head_idx": idx.item(), "novelty": val.item()}
                            for val, idx in zip(top_vals, top_idxs)
                        ]

                    global_max = combined_novelty.max().item()
                    global_mean = combined_novelty.mean().item()

                    # Check spike
                    is_spike = tracker.is_spike(global_max)
                    tracker.update_novelty_history(global_max)

                    # Buffer for snapshots
                    tracker.buffer_kv(step, k_stack, v_stack)

                    record = TokenNoveltyRecord(
                        position=step,
                        token_id=token_id,
                        token_text=token_text,
                        layer_novelty=layer_novelty,
                        layer_top_heads=layer_top_heads,
                        global_max_novelty=global_max,
                        global_mean_novelty=global_mean,
                        is_spike=is_spike,
                    )
                    token_records.append(record)

                    # Capture snapshot if spike or sampled
                    if is_spike or (step > 0 and step % config.snapshot_sample_rate == 0):
                        snapshot = _create_snapshot(
                            record, tracker, layers, config.top_k_heads
                        )
                        if snapshot:
                            snapshots.append(snapshot)

                    # Ablation test on sample of spikes
                    if is_spike and len([r for r in token_records if r.is_spike]) % config.ablation_sample_rate == 0:
                        # Find top spiking layer/head
                        top_layer = max(layer_novelty.keys(), key=lambda l: layer_novelty[l])
                        top_heads_for_layer = [h["head_idx"] for h in layer_top_heads[top_layer][:1]]

                        impact = compute_kv_ablation_impact(
                            model=model,
                            tokenizer=tokenizer,
                            input_ids=generated_ids,
                            target_position=step + input_len - 1,  # Account for prompt
                            target_layer=top_layer,
                            target_heads=top_heads_for_layer,
                            ablation_type=config.ablation_type,
                            device=device,
                        )
                        record.impact_delta_logprob = impact["delta_logprob"]
                        record.impact_kl_divergence = impact["kl_divergence"]
                        ablation_results.append({
                            "position": step,
                            "token": token_text,
                            "layer": top_layer,
                            "heads": top_heads_for_layer,
                            **impact,
                        })

            # Append token
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            # Check for EOS
            if next_token[0, 0].item() == tokenizer.eos_token_id:
                break

        # Progress
        if (step + 1) % 50 == 0:
            print(f"  Generated {step + 1} tokens...")

    total_time = time.time() - start_time
    print(f"Generation complete: {len(token_records)} tokens in {total_time:.2f}s")

    # Compute aggregate statistics
    novelty_values = [r.global_max_novelty for r in token_records]
    spikes = [r for r in token_records if r.is_spike]

    # Per-layer stats
    layer_mean_novelty = {}
    layer_spike_counts = {}
    for layer_idx in layers:
        layer_vals = [r.layer_novelty.get(layer_idx, 0) for r in token_records]
        layer_mean_novelty[layer_idx] = float(np.mean(layer_vals)) if layer_vals else 0
        layer_spike_counts[layer_idx] = sum(
            1 for r in spikes if r.layer_novelty.get(layer_idx, 0) > np.mean(layer_vals)
        )

    # Head-level spike counts
    head_spike_counts = {}
    for record in spikes:
        for layer_idx, heads in record.layer_top_heads.items():
            for head_info in heads:
                if isinstance(head_info, dict):
                    head_idx = head_info["head_idx"]
                else:
                    head_idx = head_info.head_idx
                key = f"L{layer_idx}_H{head_idx}"
                head_spike_counts[key] = head_spike_counts.get(key, 0) + 1

    # Ablation stats
    if ablation_results:
        mean_delta = np.mean([r["delta_logprob"] for r in ablation_results])
        mean_kl = np.mean([r["kl_divergence"] for r in ablation_results])
        max_delta = max(r["delta_logprob"] for r in ablation_results)
    else:
        mean_delta = mean_kl = max_delta = 0.0

    result = KVExperimentResult(
        config=asdict(config),
        num_tokens=len(token_records),
        total_time=total_time,
        mean_novelty=float(np.mean(novelty_values)) if novelty_values else 0,
        std_novelty=float(np.std(novelty_values)) if novelty_values else 0,
        max_novelty=float(max(novelty_values)) if novelty_values else 0,
        num_spikes=len(spikes),
        spike_rate=len(spikes) / len(token_records) if token_records else 0,
        layer_mean_novelty=layer_mean_novelty,
        layer_spike_counts=layer_spike_counts,
        head_spike_counts=head_spike_counts,
        num_ablations=len(ablation_results),
        mean_delta_logprob=mean_delta,
        mean_kl_divergence=mean_kl,
        max_delta_logprob=max_delta,
        spike_tokens=[
            {
                "position": r.position,
                "token": r.token_text,
                "novelty": r.global_max_novelty,
                "top_layer": max(r.layer_novelty.keys(), key=lambda l: r.layer_novelty[l]),
                "impact_delta": r.impact_delta_logprob,
            }
            for r in spikes
        ],
        num_snapshots=len(snapshots),
    )

    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save summary
        with open(output_dir / "results.json", "w") as f:
            json.dump(asdict(result), f, indent=2, default=str)

        # Save detailed token records
        records_data = [
            {
                "position": r.position,
                "token_id": r.token_id,
                "token_text": r.token_text,
                "layer_novelty": convert_numpy(r.layer_novelty),
                "global_max_novelty": float(r.global_max_novelty),
                "global_mean_novelty": float(r.global_mean_novelty),
                "is_spike": bool(r.is_spike),
                "impact_delta_logprob": float(r.impact_delta_logprob) if r.impact_delta_logprob is not None else None,
                "impact_kl_divergence": float(r.impact_kl_divergence) if r.impact_kl_divergence is not None else None,
            }
            for r in token_records
        ]
        with open(output_dir / "token_records.json", "w") as f:
            json.dump(records_data, f, indent=2)

        # Save ablation results
        if ablation_results:
            with open(output_dir / "ablation_results.json", "w") as f:
                json.dump(convert_numpy(ablation_results), f, indent=2)

        # Save snapshots (numpy format for raw K/V)
        if snapshots:
            snapshots_dir = output_dir / "snapshots"
            snapshots_dir.mkdir(exist_ok=True)

            for i, snap in enumerate(snapshots):
                snap_data = {
                    "spike_position": snap.spike_position,
                    "spike_token": snap.spike_token,
                    "spike_novelty": snap.spike_novelty,
                    "captured_layers": snap.captured_layers,
                    "captured_heads": snap.captured_heads,
                }
                with open(snapshots_dir / f"snapshot_{i}_meta.json", "w") as f:
                    json.dump(snap_data, f, indent=2)

                # Save raw K/V as npz
                np.savez_compressed(
                    snapshots_dir / f"snapshot_{i}_kv.npz",
                    **{f"k_{k[0]}_{k[1]}_{k[2]}": v for k, v in snap.keys.items()},
                    **{f"v_{k[0]}_{k[1]}_{k[2]}": v for k, v in snap.values.items()},
                )

        print(f"Results saved to {output_dir}")

    return result


def _create_snapshot(
    record: TokenNoveltyRecord,
    tracker: OnlineKVNoveltyTracker,
    layers: list[int],
    top_k: int,
) -> KVSnapshot | None:
    """Create a K/V snapshot from buffered data."""
    buffered = tracker.get_buffered_kv()
    if not buffered:
        return None

    snapshot = KVSnapshot(
        spike_position=record.position,
        spike_token=record.token_text,
        spike_novelty=record.global_max_novelty,
    )

    # Get top spiking layers
    top_layers = sorted(
        record.layer_novelty.keys(),
        key=lambda l: record.layer_novelty[l],
        reverse=True,
    )[:3]

    for buf in buffered:
        pos = buf["position"]
        k = buf["k"].numpy()  # [num_layers, num_heads, head_dim]
        v = buf["v"].numpy()

        rel_pos = pos - record.position

        for i, layer_idx in enumerate(layers):
            if layer_idx not in top_layers:
                continue
            if i >= k.shape[0]:
                continue

            # Get top heads
            if layer_idx in record.layer_top_heads:
                heads_info = record.layer_top_heads[layer_idx]
                if heads_info and isinstance(heads_info[0], dict):
                    top_heads = [h["head_idx"] for h in heads_info[:top_k]]
                else:
                    top_heads = [h.head_idx for h in heads_info[:top_k]]
            else:
                top_heads = list(range(min(top_k, k.shape[1])))

            for head_idx in top_heads:
                if head_idx >= k.shape[1]:
                    continue

                key = (layer_idx, head_idx, rel_pos)
                snapshot.keys[key] = k[i, head_idx]
                snapshot.values[key] = v[i, head_idx]

    snapshot.captured_layers = top_layers
    snapshot.captured_heads = {
        l: [h["head_idx"] if isinstance(h, dict) else h.head_idx
            for h in record.layer_top_heads.get(l, [])[:top_k]]
        for l in top_layers
    }

    return snapshot


def print_experiment_summary(result: KVExperimentResult):
    """Print a summary of experiment results."""
    print("\n" + "=" * 70)
    print("KV NOVELTY EXPERIMENT RESULTS")
    print("=" * 70)

    print(f"\nModel: {result.config['model_name']}")
    print(f"Tokens: {result.num_tokens}")
    print(f"Time: {result.total_time:.2f}s")

    print(f"\n--- Novelty Statistics ---")
    print(f"Mean novelty: {result.mean_novelty:.4f}")
    print(f"Std novelty: {result.std_novelty:.4f}")
    print(f"Max novelty: {result.max_novelty:.4f}")
    print(f"Spikes: {result.num_spikes} ({result.spike_rate*100:.1f}%)")

    print(f"\n--- Per-Layer Statistics ---")
    for layer, mean_nov in sorted(result.layer_mean_novelty.items()):
        spike_count = result.layer_spike_counts.get(layer, 0)
        print(f"  Layer {layer}: mean={mean_nov:.4f}, spikes={spike_count}")

    print(f"\n--- Top Spiking Heads ---")
    sorted_heads = sorted(result.head_spike_counts.items(), key=lambda x: -x[1])[:10]
    for head, count in sorted_heads:
        print(f"  {head}: {count} spikes")

    print(f"\n--- Causal Impact (Ablation) ---")
    print(f"Ablations performed: {result.num_ablations}")
    if result.num_ablations > 0:
        print(f"Mean Δlogprob: {result.mean_delta_logprob:.4f}")
        print(f"Mean KL divergence: {result.mean_kl_divergence:.4f}")
        print(f"Max Δlogprob: {result.max_delta_logprob:.4f}")

    print(f"\n--- Spike Tokens (first 10) ---")
    for spike in result.spike_tokens[:10]:
        impact_str = f", Δlp={spike['impact_delta']:.3f}" if spike.get('impact_delta') else ""
        print(f"  [{spike['position']}] {repr(spike['token'])}: nov={spike['novelty']:.3f}, L{spike['top_layer']}{impact_str}")

    print(f"\nSnapshots captured: {result.num_snapshots}")
    print("=" * 70)
