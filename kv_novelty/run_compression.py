#!/usr/bin/env python3
"""
CLI for KV-cache compression experiments.

Usage:
    python -m kv_novelty.run_compression --model Qwen/Qwen3-8B --strategy k_only
    python -m kv_novelty.run_compression --model Qwen/Qwen3-8B --strategy conservative --percentile 30
"""

import argparse
from pathlib import Path
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .kv_compression import (
    CompressionConfig,
    run_compression_experiment,
    print_compression_results,
)


PROMPTS = {
    "explanation": (
        "Explain how attention mechanisms work in transformers. "
        "Start with the intuition, then describe Q, K, V projections."
    ),
    "reasoning": (
        "Solve step by step: A train leaves station A at 9am traveling at 60mph. "
        "Another train leaves station B (100 miles away) at 10am traveling at 80mph "
        "toward station A. When do they meet?"
    ),
    "code": (
        "Write a Python function to find the longest common subsequence of two strings. "
        "Include comments explaining the dynamic programming approach."
    ),
}


def main():
    parser = argparse.ArgumentParser(description="KV-cache compression experiments")

    parser.add_argument(
        "--model", "-m",
        default="Qwen/Qwen3-8B",
        help="Model name or path",
    )
    parser.add_argument(
        "--strategy", "-s",
        choices=["k_only", "full", "conservative"],
        default="k_only",
        help="Compression strategy (default: k_only)",
    )
    parser.add_argument(
        "--percentile", "-p",
        type=float,
        default=50.0,
        help="Novelty percentile threshold - entries below this are compressed (default: 50)",
    )
    parser.add_argument(
        "--drop-method",
        choices=["zero", "mean", "first"],
        default="zero",
        help="How to handle compressed entries (default: zero)",
    )
    parser.add_argument(
        "--spike-window",
        type=int,
        default=2,
        help="For conservative strategy: tokens to preserve around spikes (default: 2)",
    )
    parser.add_argument(
        "--max-tokens", "-n",
        type=int,
        default=100,
        help="Max tokens to generate (default: 100)",
    )
    parser.add_argument(
        "--prompt-type",
        choices=list(PROMPTS.keys()),
        default="explanation",
        help="Type of prompt to use",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Custom prompt (overrides --prompt-type)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output directory for results",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run sweep over percentiles (10, 30, 50, 70, 90)",
    )

    args = parser.parse_args()

    # Load model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Get prompt
    prompt = args.prompt if args.prompt else PROMPTS[args.prompt_type]

    if args.sweep:
        # Run sweep over percentiles
        percentiles = [10, 30, 50, 70, 90]
        results = []

        print("\n" + "=" * 70)
        print("COMPRESSION SWEEP")
        print("=" * 70)
        print(f"Model: {args.model}")
        print(f"Strategy: {args.strategy}")
        print(f"Percentiles: {percentiles}")
        print("=" * 70)

        for pct in percentiles:
            print(f"\n--- Percentile {pct}% ---")

            config = CompressionConfig(
                novelty_percentile=pct,
                strategy=args.strategy,
                spike_window=args.spike_window,
                drop_method=args.drop_method,
            )

            result = run_compression_experiment(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                config=config,
                max_new_tokens=args.max_tokens,
            )

            results.append({
                "percentile": pct,
                "compression_ratio": result.compression_ratio,
                "original_loss": result.original_loss,
                "compressed_loss": result.compressed_loss,
                "loss_increase_pct": result.loss_increase_pct,
                "mean_kl": result.mean_kl_divergence,
            })

            print(f"  Compression: {result.compression_ratio*100:.1f}%")
            print(f"  Loss increase: {result.loss_increase_pct:.2f}%")

        # Summary table
        print("\n" + "=" * 70)
        print("SWEEP SUMMARY")
        print("=" * 70)
        print(f"{'Percentile':<12} {'Compression':<12} {'Loss Δ%':<12} {'Mean |Δlp|':<12}")
        print("-" * 50)
        for r in results:
            print(f"{r['percentile']:<12} {r['compression_ratio']*100:<12.1f} "
                  f"{r['loss_increase_pct']:<12.2f} {r['mean_kl']:<12.4f}")

        if args.output:
            args.output.mkdir(parents=True, exist_ok=True)
            with open(args.output / "sweep_results.json", "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output / 'sweep_results.json'}")

    else:
        # Single run
        config = CompressionConfig(
            novelty_percentile=args.percentile,
            strategy=args.strategy,
            spike_window=args.spike_window,
            drop_method=args.drop_method,
        )

        print("\n" + "=" * 70)
        print("KV-CACHE COMPRESSION EXPERIMENT")
        print("=" * 70)
        print(f"Model: {args.model}")
        print(f"Strategy: {args.strategy}")
        print(f"Percentile: {args.percentile}%")
        print(f"Drop method: {args.drop_method}")
        print("=" * 70)

        result = run_compression_experiment(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            config=config,
            max_new_tokens=args.max_tokens,
        )

        print_compression_results(result)

        if args.output:
            args.output.mkdir(parents=True, exist_ok=True)
            with open(args.output / "compression_result.json", "w") as f:
                json.dump({
                    "config": vars(config),
                    "compression_ratio": result.compression_ratio,
                    "original_loss": result.original_loss,
                    "compressed_loss": result.compressed_loss,
                    "loss_increase_pct": result.loss_increase_pct,
                    "mean_kl": result.mean_kl_divergence,
                    "max_kl": result.max_kl_divergence,
                    "worst_tokens": result.worst_affected_tokens,
                }, f, indent=2)
            print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
