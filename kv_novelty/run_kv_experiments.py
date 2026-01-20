#!/usr/bin/env python3
"""
Command-line runner for actual KV cache novelty experiments.

Usage:
    python -m kv_novelty.run_kv_experiments --model Qwen/Qwen3-8B --max-tokens 200
    python -m kv_novelty.run_kv_experiments --model meta-llama/Llama-3.1-8B-Instruct --suite
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from .kv_experiments import (
    KVExperimentConfig,
    run_kv_experiment,
    print_experiment_summary,
)


# Standard prompts for experiments
EXPERIMENT_PROMPTS = {
    "prose_explanation": (
        "Explain how transformers work in deep learning. Start with the high-level "
        "intuition of attention, then describe the mathematical formulation including "
        "Q, K, V projections and scaled dot-product attention."
    ),
    "code_generation": (
        "Write a Python function that implements a binary search tree with insert, "
        "search, and delete operations. Include docstrings and handle edge cases."
    ),
    "reasoning_chain": (
        "Solve this step by step: A farmer has 17 sheep. All but 9 run away. "
        "How many sheep does the farmer have left? Explain your reasoning carefully."
    ),
    "creative_writing": (
        "Write the opening paragraph of a mystery novel set in a small coastal town "
        "where strange lights have been appearing over the ocean at night."
    ),
    "technical_documentation": (
        "Write API documentation for a REST endpoint that handles user authentication. "
        "Include request/response formats, error codes, and example usage."
    ),
}


def main():
    parser = argparse.ArgumentParser(
        description="Run KV cache novelty experiments with actual K/V capture"
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Model name or path (e.g., Qwen/Qwen3-8B)",
    )
    parser.add_argument(
        "--max-tokens", "-n",
        type=int,
        default=200,
        help="Maximum tokens to generate (default: 200)",
    )
    parser.add_argument(
        "--prompt", "-p",
        help="Custom prompt (default: uses prose_explanation)",
    )
    parser.add_argument(
        "--prompt-type",
        choices=list(EXPERIMENT_PROMPTS.keys()),
        default="prose_explanation",
        help="Type of prompt to use (default: prose_explanation)",
    )
    parser.add_argument(
        "--suite",
        action="store_true",
        help="Run full experiment suite (all prompt types)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output directory (default: kv_novelty_experiments/kv_<model>_<timestamp>)",
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--spike-threshold",
        type=float,
        default=2.0,
        help="Spike detection threshold in std deviations (default: 2.0)",
    )
    parser.add_argument(
        "--ablation-rate",
        type=int,
        default=5,
        help="Test causal impact on 1/N spikes (default: 5)",
    )
    parser.add_argument(
        "--no-ablation",
        action="store_true",
        help="Skip ablation tests (faster)",
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output:
        output_base = args.output
    else:
        model_short = args.model.split("/")[-1].replace("-", "_").lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = Path(f"kv_novelty_experiments/kv_{model_short}_{timestamp}")

    # Determine prompts to run
    if args.suite:
        prompts = EXPERIMENT_PROMPTS
    else:
        prompt = args.prompt if args.prompt else EXPERIMENT_PROMPTS[args.prompt_type]
        prompts = {args.prompt_type: prompt}

    print("=" * 70)
    print("KV CACHE NOVELTY EXPERIMENTS")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Experiments: {list(prompts.keys())}")
    print(f"Output: {output_base}")
    print("=" * 70)

    all_results = {}

    for exp_name, prompt in prompts.items():
        print(f"\n{'='*70}")
        print(f"EXPERIMENT: {exp_name}")
        print(f"{'='*70}")

        config = KVExperimentConfig(
            model_name=args.model,
            prompt=prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            spike_threshold_std=args.spike_threshold,
            ablation_sample_rate=args.ablation_rate if not args.no_ablation else 999999,
        )

        output_dir = output_base / exp_name

        try:
            result = run_kv_experiment(
                config=config,
                output_dir=output_dir,
            )

            print_experiment_summary(result)
            all_results[exp_name] = {
                "num_tokens": result.num_tokens,
                "spike_rate": result.spike_rate,
                "mean_novelty": result.mean_novelty,
                "mean_delta_logprob": result.mean_delta_logprob,
                "num_spikes": result.num_spikes,
            }

        except Exception as e:
            print(f"ERROR in {exp_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[exp_name] = {"error": str(e)}

    # Save comparison
    if len(prompts) > 1:
        print("\n" + "=" * 70)
        print("EXPERIMENT COMPARISON")
        print("=" * 70)

        print(f"\n{'Experiment':<25} {'Tokens':<8} {'Spikes':<8} {'Rate':<8} {'Mean Nov':<10} {'Δlogprob':<10}")
        print("-" * 80)

        for exp_name, res in all_results.items():
            if "error" in res:
                print(f"{exp_name:<25} ERROR: {res['error'][:40]}")
            else:
                print(f"{exp_name:<25} {res['num_tokens']:<8} {res['num_spikes']:<8} "
                      f"{res['spike_rate']:.3f}    {res['mean_novelty']:.4f}     "
                      f"{res['mean_delta_logprob']:.4f}")

        output_base.mkdir(parents=True, exist_ok=True)
        with open(output_base / "comparison.json", "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"\nComparison saved to {output_base / 'comparison.json'}")


if __name__ == "__main__":
    main()
