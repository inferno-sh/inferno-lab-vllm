#!/usr/bin/env python3
"""
CLI for window-based KV-cache compression experiments.

Usage:
    python -m kv_novelty.run_window_compression --model Qwen/Qwen3-8B
    python -m kv_novelty.run_window_compression --model Qwen/Qwen3-8B --rank-ratio 0.25
"""

import argparse
from pathlib import Path
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .kv_window_compression import (
    WindowCompressionConfig,
    CompressionMethod,
    run_window_compression_experiment,
    print_window_compression_results,
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
    parser = argparse.ArgumentParser(description="Window-based KV compression")

    parser.add_argument(
        "--model", "-m",
        default="Qwen/Qwen3-8B",
        help="Model name or path",
    )
    parser.add_argument(
        "--method",
        choices=["low_rank", "cluster"],
        default="low_rank",
        help="Compression method",
    )
    parser.add_argument(
        "--rank-ratio",
        type=float,
        default=0.5,
        help="For low_rank: fraction of singular values to keep (default: 0.5)",
    )
    parser.add_argument(
        "--novelty-threshold",
        type=float,
        default=50.0,
        help="Percentile threshold for low-novelty (default: 50)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=16,
        help="Tokens per compression window (default: 16)",
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
        "--sweep",
        action="store_true",
        help="Run sweep over rank ratios (0.25, 0.5, 0.75)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output directory for results",
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

    prompt = args.prompt if args.prompt else PROMPTS[args.prompt_type]

    if args.sweep:
        rank_ratios = [0.25, 0.5, 0.75]
        results = []

        print("\n" + "=" * 70)
        print("WINDOW COMPRESSION SWEEP")
        print("=" * 70)
        print(f"Model: {args.model}")
        print(f"Method: {args.method}")
        print(f"Rank ratios: {rank_ratios}")
        print("=" * 70)

        for ratio in rank_ratios:
            print(f"\n--- Rank ratio {ratio} ---")

            config = WindowCompressionConfig(
                method=CompressionMethod(args.method),
                rank_ratio=ratio,
                novelty_threshold=args.novelty_threshold,
                window_size=args.window_size,
            )

            result = run_window_compression_experiment(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                config=config,
                max_new_tokens=args.max_tokens,
            )

            results.append({
                "rank_ratio": ratio,
                "windows_compressed": result.windows_compressed,
                "compression_ratio": result.compression_ratio,
                "loss_increase_pct": result.loss_increase_pct,
                "mean_logprob_delta": result.mean_logprob_delta,
                "reconstruction_error": result.mean_reconstruction_error,
            })

            print(f"  Windows: {result.windows_compressed}")
            print(f"  Loss increase: {result.loss_increase_pct:.2f}%")
            print(f"  Reconstruction error: {result.mean_reconstruction_error:.4f}")

        # Summary
        print("\n" + "=" * 70)
        print("SWEEP SUMMARY")
        print("=" * 70)
        print(f"{'Rank':<8} {'Windows':<10} {'Loss Δ%':<12} {'Mean |Δlp|':<12} {'Recon Err':<12}")
        print("-" * 55)
        for r in results:
            print(f"{r['rank_ratio']:<8} {r['windows_compressed']:<10} "
                  f"{r['loss_increase_pct']:<12.2f} {r['mean_logprob_delta']:<12.4f} "
                  f"{r['reconstruction_error']:<12.4f}")

        if args.output:
            args.output.mkdir(parents=True, exist_ok=True)
            with open(args.output / "window_sweep_results.json", "w") as f:
                json.dump(results, f, indent=2)

    else:
        config = WindowCompressionConfig(
            method=CompressionMethod(args.method),
            rank_ratio=args.rank_ratio,
            novelty_threshold=args.novelty_threshold,
            window_size=args.window_size,
        )

        print("\n" + "=" * 70)
        print("WINDOW-BASED KV COMPRESSION")
        print("=" * 70)
        print(f"Model: {args.model}")
        print(f"Method: {args.method}")
        print(f"Rank ratio: {args.rank_ratio}")
        print(f"Novelty threshold: {args.novelty_threshold}%")
        print("=" * 70)

        result = run_window_compression_experiment(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            config=config,
            max_new_tokens=args.max_tokens,
        )

        print_window_compression_results(result)

        if args.output:
            args.output.mkdir(parents=True, exist_ok=True)
            with open(args.output / "window_result.json", "w") as f:
                json.dump({
                    "config": result.config,
                    "compression_ratio": result.compression_ratio,
                    "loss_increase_pct": result.loss_increase_pct,
                    "mean_logprob_delta": result.mean_logprob_delta,
                    "reconstruction_error": result.mean_reconstruction_error,
                }, f, indent=2)


if __name__ == "__main__":
    main()
