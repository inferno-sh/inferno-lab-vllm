#!/usr/bin/env python3
"""
Main script to run KV-cache novelty experiments.

This script runs controlled experiments to test the hypothesis that
KV-cache entries exhibit sparse, bursty novelty patterns corresponding
to "thinking turns" in transformer inference.

Usage:
    # Run with default settings (requires GPU and transformers)
    python -m kv_novelty.run_experiments

    # Run a quick test with a small model
    python -m kv_novelty.run_experiments --model gpt2 --max-tokens 100

    # Run full experiment suite
    python -m kv_novelty.run_experiments --suite --model meta-llama/Llama-3.1-8B-Instruct
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def check_requirements():
    """Check that required packages are installed."""
    missing = []
    try:
        import torch
    except ImportError:
        missing.append("torch")

    try:
        import transformers
    except ImportError:
        missing.append("transformers")

    if missing:
        print(f"Missing required packages: {missing}")
        print("Install with: pip install torch transformers")
        return False
    return True


def run_single_experiment(
    model_name: str,
    prompt: str,
    prompt_name: str,
    output_dir: Path,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    do_sample: bool = True,
    layers_to_capture: list[int] | None = None,
    device: str = "cuda",
):
    """Run a single experiment and save results."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .collector import KVNoveltyCollector
    from .llama_hooks import run_with_kv_capture
    from .analysis import analyze_collection, generate_text_report, generate_matplotlib_plots

    print(f"\n{'='*60}")
    print(f"Experiment: {prompt_name}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    # Create output directory
    exp_dir = output_dir / prompt_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Load model and tokenizer
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device if device == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()

    # Determine layers to capture (sample if too many)
    num_layers = len(model.model.layers) if hasattr(model, "model") and hasattr(model.model, "layers") else 0
    if layers_to_capture is None and num_layers > 8:
        # Sample layers across the model
        layers_to_capture = [0, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1]
        print(f"Capturing layers: {layers_to_capture} (sampled from {num_layers} total)")

    # Create collector
    collector = KVNoveltyCollector(
        capture_kv=True,
        layers_to_capture=layers_to_capture,
    )

    # Run generation with capture
    print("Generating...")
    generated_text, collector = run_with_kv_capture(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        collector=collector,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        layers_to_capture=layers_to_capture,
        device=device,
    )

    print(f"Generated {collector.num_positions} tokens")

    # Analyze
    print("Analyzing...")
    result = analyze_collection(collector)

    # Print summary
    print("\n" + "-" * 40)
    print("RESULTS SUMMARY")
    print("-" * 40)
    print(f"Tokens: {result.num_tokens}")
    print(f"Spikes detected: {len(result.spike_positions)}")
    print(f"Spike rate: {result.aggregate_stats.get('spike_rate', 0):.3f}")
    print(f"Burstiness index: {result.aggregate_stats.get('burstiness_index', 0):.3f}")
    print(f"Heavy-tailed: {result.heavy_tail_test.get('is_heavy_tailed', 'N/A')}")
    print(f"Kurtosis: {result.heavy_tail_test.get('kurtosis', 0):.3f}")

    # Save results
    print(f"\nSaving to {exp_dir}...")

    # Save collector data
    collector.save(exp_dir / "collected_data")

    # Save analysis
    with open(exp_dir / "analysis.json", "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    # Save report
    report = generate_text_report(result)
    with open(exp_dir / "report.txt", "w") as f:
        f.write(report)

    # Save generated text
    with open(exp_dir / "generated.txt", "w") as f:
        f.write(f"PROMPT:\n{prompt}\n\n")
        f.write(f"GENERATED:\n{generated_text}")

    # Generate plots
    try:
        plots = generate_matplotlib_plots(result, exp_dir / "plots")
        print(f"Generated {len(plots)} plots")
    except Exception as e:
        print(f"Could not generate plots: {e}")

    # Clean up GPU memory
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def run_experiment_suite(
    model_name: str,
    output_dir: Path,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
):
    """Run the full experiment suite."""
    from .experiments import EXPERIMENT_PROMPTS
    from .analysis import compare_experiments

    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_dir = output_dir / f"suite_{timestamp}"
    suite_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("KV-CACHE NOVELTY EXPERIMENT SUITE")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Output: {suite_dir}")
    print(f"Prompts: {len(EXPERIMENT_PROMPTS)}")

    for prompt_name, prompt in EXPERIMENT_PROMPTS.items():
        try:
            result = run_single_experiment(
                model_name=model_name,
                prompt=prompt,
                prompt_name=prompt_name,
                output_dir=suite_dir,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            results[prompt_name] = result
        except Exception as e:
            print(f"ERROR in {prompt_name}: {e}")
            import traceback
            traceback.print_exc()

    # Generate comparison
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("SUITE COMPARISON")
        print("=" * 60)

        comparison = compare_experiments(
            list(results.values()),
            list(results.keys()),
        )

        print(f"\n{'Prompt':<25} {'Tokens':<8} {'Spikes':<8} {'Rate':<8} {'Burst':<8} {'Heavy':<8}")
        print("-" * 73)
        for i, name in enumerate(comparison["labels"]):
            print(
                f"{name:<25} "
                f"{comparison['num_tokens'][i]:<8} "
                f"{comparison['num_spikes'][i]:<8} "
                f"{comparison['spike_rate'][i]:<8.3f} "
                f"{comparison['burstiness'][i]:<8.3f} "
                f"{str(comparison['is_heavy_tailed'][i]):<8}"
            )

        # Save comparison
        with open(suite_dir / "comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)

        # Generate summary report
        with open(suite_dir / "summary.md", "w") as f:
            f.write("# KV-Cache Novelty Experiment Suite Results\n\n")
            f.write(f"**Model:** {model_name}\n")
            f.write(f"**Date:** {timestamp}\n\n")

            f.write("## Overview\n\n")
            f.write(f"- Total experiments: {len(results)}\n")
            f.write(f"- Average spike rate: {np.mean(comparison['spike_rate']):.3f}\n")
            f.write(f"- Average burstiness: {np.mean(comparison['burstiness']):.3f}\n")
            f.write(f"- Heavy-tailed distributions: {sum(comparison['is_heavy_tailed'])}/{len(comparison['is_heavy_tailed'])}\n\n")

            f.write("## Results Table\n\n")
            f.write("| Prompt | Tokens | Spikes | Rate | Burstiness | Heavy-tailed |\n")
            f.write("|--------|--------|--------|------|------------|-------------|\n")
            for i, name in enumerate(comparison["labels"]):
                f.write(
                    f"| {name} | {comparison['num_tokens'][i]} | "
                    f"{comparison['num_spikes'][i]} | {comparison['spike_rate'][i]:.3f} | "
                    f"{comparison['burstiness'][i]:.3f} | {comparison['is_heavy_tailed'][i]} |\n"
                )

            f.write("\n## Interpretation\n\n")

            # Basic interpretation
            avg_burstiness = np.mean(comparison['burstiness'])
            heavy_tailed_frac = sum(comparison['is_heavy_tailed']) / len(comparison['is_heavy_tailed'])

            if avg_burstiness > 0.1:
                f.write("- **Burstiness index > 0** indicates novelty spikes are clustered, not uniform.\n")
            else:
                f.write("- **Burstiness index near 0** suggests novelty is more uniformly distributed.\n")

            if heavy_tailed_frac > 0.5:
                f.write("- **Most distributions are heavy-tailed**, suggesting sparse, extreme novelty events.\n")
            else:
                f.write("- **Most distributions are not heavy-tailed**, suggesting more uniform novelty.\n")

            f.write("\nSee individual experiment directories for detailed analysis.\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="KV-Cache Novelty Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with GPT-2
  python -m kv_novelty.run_experiments --model gpt2 --max-tokens 100

  # Run with Llama
  python -m kv_novelty.run_experiments --model meta-llama/Llama-3.1-8B-Instruct

  # Full experiment suite
  python -m kv_novelty.run_experiments --suite

  # Custom prompt
  python -m kv_novelty.run_experiments --prompt "Write a poem about AI"
        """,
    )

    parser.add_argument(
        "--model", "-m",
        default="gpt2",
        help="Model name or path (default: gpt2 for testing)",
    )
    parser.add_argument(
        "--prompt", "-p",
        default=None,
        help="Custom prompt for single experiment",
    )
    parser.add_argument(
        "--suite", "-s",
        action="store_true",
        help="Run full experiment suite",
    )
    parser.add_argument(
        "--output", "-o",
        default="kv_novelty_experiments",
        help="Output directory",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--device",
        default="cuda" if __import__("torch").cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layer indices to capture (e.g., '0,8,16,24,31')",
    )

    args = parser.parse_args()

    if not check_requirements():
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    layers = None
    if args.layers:
        layers = [int(x.strip()) for x in args.layers.split(",")]

    if args.suite:
        run_experiment_suite(
            model_name=args.model,
            output_dir=output_dir,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    else:
        prompt = args.prompt or "Explain how neural networks learn. First describe the intuition, then the mathematics."
        run_single_experiment(
            model_name=args.model,
            prompt=prompt,
            prompt_name="custom",
            output_dir=output_dir,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            layers_to_capture=layers,
            device=args.device,
        )


if __name__ == "__main__":
    main()
