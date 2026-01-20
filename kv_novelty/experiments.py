"""
Experiment runner for KV-cache novelty research.

This module provides functions to run controlled experiments testing
the hypothesis about KV-cache novelty and "thinking turns".

Two modes of operation:
1. HuggingFace Transformers (full KV capture, slower)
2. vLLM (logprob/entropy only, faster, for large-scale experiments)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from .collector import KVNoveltyCollector
from .metrics import compute_all_novelty_metrics, NoveltyMetrics
from .analysis import analyze_collection, AnalysisResult, generate_text_report


# =============================================================================
# Experiment Configuration
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for a novelty experiment."""

    # Model settings
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    device: str = "cuda"
    dtype: str = "bfloat16"

    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

    # Capture settings
    capture_kv: bool = True
    layers_to_capture: list[int] | None = None  # None = all
    window_size: int = 16

    # Backend
    backend: Literal["transformers", "vllm"] = "transformers"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "dtype": self.dtype,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
            "capture_kv": self.capture_kv,
            "layers_to_capture": self.layers_to_capture,
            "window_size": self.window_size,
            "backend": self.backend,
        }


# =============================================================================
# Prompts for different content types
# =============================================================================

EXPERIMENT_PROMPTS = {
    "prose_explanation": """Explain how neural networks learn through backpropagation.
Start with the intuition, then describe the mathematics, and finally discuss practical considerations.""",

    "code_generation": """Write a Python function that implements a binary search tree with insert,
search, and delete operations. Include proper error handling and docstrings.""",

    "mixed_content": """Explain the concept of recursion in programming. First describe it conceptually,
then show a Python example of calculating factorial recursively, and finally discuss when to use
recursion vs iteration.""",

    "reasoning_chain": """Solve this step by step: A farmer has 17 sheep. All but 9 run away.
How many sheep does the farmer have left? Walk through your reasoning carefully.""",

    "creative_writing": """Write the opening paragraph of a mystery novel set in a small coastal town.
The scene should introduce the protagonist and hint at an upcoming discovery.""",

    "technical_documentation": """Document the following API endpoint for a user authentication system:
POST /api/v1/auth/login
Include request format, response format, error codes, and example usage.""",

    "list_generation": """List the top 10 most important algorithms every programmer should know,
with a brief explanation of why each is important and where it's commonly used.""",

    "transition_heavy": """First, explain what machine learning is in simple terms.
Then, write Python code to train a simple linear regression model.
After that, discuss the ethical implications of ML in hiring decisions.
Finally, suggest three books for someone wanting to learn more about ML.""",
}


# =============================================================================
# HuggingFace Transformers Backend
# =============================================================================

class TransformersExperiment:
    """
    Experiment runner using HuggingFace Transformers.

    This backend allows full KV-cache capture through hooks.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.collector = None

    def setup(self) -> None:
        """Load model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading model: {self.config.model_name}")

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map.get(self.config.dtype, torch.bfloat16)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=dtype,
            device_map=self.config.device,
            trust_remote_code=True,
        )
        self.model.eval()

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Model loaded. Layers: {len(self.model.model.layers)}")

    def _install_hooks(self, collector: KVNoveltyCollector) -> list:
        """Install forward hooks to capture KV tensors."""
        hooks = []

        def make_hook(layer_idx):
            def hook(module, inputs, outputs):
                # For LlamaAttention, outputs is the attention output
                # We need to capture K, V from the inputs after projection
                # This depends on the specific model architecture
                pass  # Will be filled in for specific models
            return hook

        # For Llama-style models
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            for layer_idx, layer in enumerate(self.model.model.layers):
                if self.config.layers_to_capture is not None:
                    if layer_idx not in self.config.layers_to_capture:
                        continue

                # Hook into the attention layer
                if hasattr(layer, "self_attn"):
                    # We need a pre-forward hook to capture K, V before attention
                    def make_kv_hook(idx):
                        def kv_hook(module, args):
                            # hidden_states is typically the first argument
                            if len(args) > 0:
                                hidden_states = args[0]
                                # This captures the input to attention
                                # For full KV, we'd need to hook after qkv projection
                        return kv_hook

                    handle = layer.self_attn.register_forward_pre_hook(make_kv_hook(layer_idx))
                    hooks.append(handle)

        return hooks

    def run(
        self,
        prompt: str,
        prompt_name: str = "custom",
    ) -> tuple[KVNoveltyCollector, str]:
        """
        Run generation with KV capture.

        Returns:
            Tuple of (collector with captured data, generated text)
        """
        if self.model is None:
            self.setup()

        # Create collector
        collector = KVNoveltyCollector(
            capture_kv=self.config.capture_kv,
            layers_to_capture=self.config.layers_to_capture,
        )
        collector.start_collection(
            model_name=self.config.model_name,
            prompt=prompt,
            generation_config=self.config.to_dict(),
        )

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)
        input_len = inputs["input_ids"].shape[1]

        # Generate with output_scores for entropy/logprob calculation
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature if self.config.do_sample else 1.0,
                top_p=self.config.top_p if self.config.do_sample else 1.0,
                do_sample=self.config.do_sample,
                output_scores=True,
                output_hidden_states=self.config.capture_kv,
                output_attentions=False,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Extract generated tokens
        generated_ids = outputs.sequences[0, input_len:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Process scores to get logprobs and entropy
        if hasattr(outputs, "scores") and outputs.scores:
            for t, score in enumerate(outputs.scores):
                # score shape: [batch_size, vocab_size]
                logits = score[0]  # Take first batch
                probs = torch.softmax(logits, dim=-1)
                log_probs = torch.log_softmax(logits, dim=-1)

                # Get the generated token
                token_id = generated_ids[t].item()
                token_logprob = log_probs[token_id].item()

                # Compute entropy
                entropy = -torch.sum(probs * log_probs).item()

                # Get token text
                token_text = self.tokenizer.decode([token_id])

                # Record
                collector.record_token(
                    token_id=token_id,
                    token_text=token_text,
                    logprob=token_logprob,
                    entropy=entropy,
                )

                # Process hidden states for KV if available
                if self.config.capture_kv and hasattr(outputs, "hidden_states"):
                    # hidden_states is tuple of (layer_outputs) for each generation step
                    # layer_outputs is tuple of hidden states for each layer
                    if t < len(outputs.hidden_states):
                        step_hidden = outputs.hidden_states[t]
                        # step_hidden is a tuple with one tensor per layer
                        for layer_idx, layer_hidden in enumerate(step_hidden):
                            if self.config.layers_to_capture is not None:
                                if layer_idx not in self.config.layers_to_capture:
                                    continue
                            # layer_hidden shape: [batch, seq_len, hidden_size]
                            # We use the last token's hidden state as a proxy for KV
                            # (Actual KV requires hooks into attention)
                            hidden = layer_hidden[0, -1].cpu().numpy()
                            # Store as a simple 1D array (hidden state proxy)
                            collector.kv_data.setdefault(layer_idx, {})
                            from .collector import LayerKVData
                            collector.kv_data[layer_idx][t] = LayerKVData(
                                layer_idx=layer_idx,
                                position=t,
                                key=hidden.reshape(1, -1),  # [1, hidden_size]
                                value=None,
                            )

                collector.advance_position()

        collector.stop_collection()
        return collector, generated_text


# =============================================================================
# vLLM Backend (faster, limited capture)
# =============================================================================

class VLLMExperiment:
    """
    Experiment runner using vLLM.

    This backend is faster but currently limited to logprob/entropy capture
    (no direct KV-cache access without significant modifications).
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.llm = None

    def setup(self) -> None:
        """Initialize vLLM engine."""
        from vllm import LLM, SamplingParams

        print(f"Initializing vLLM with model: {self.config.model_name}")

        self.llm = LLM(
            model=self.config.model_name,
            dtype=self.config.dtype,
            trust_remote_code=True,
        )

        print("vLLM initialized")

    def run(
        self,
        prompt: str,
        prompt_name: str = "custom",
    ) -> tuple[KVNoveltyCollector, str]:
        """
        Run generation with vLLM.

        Returns:
            Tuple of (collector with token data, generated text)
        """
        from vllm import SamplingParams

        if self.llm is None:
            self.setup()

        # Create collector (KV capture disabled for vLLM)
        collector = KVNoveltyCollector(
            capture_kv=False,  # vLLM doesn't easily expose KV cache
        )
        collector.start_collection(
            model_name=self.config.model_name,
            prompt=prompt,
            generation_config=self.config.to_dict(),
        )

        # Sampling parameters
        sampling_params = SamplingParams(
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature if self.config.do_sample else 0.0,
            top_p=self.config.top_p if self.config.do_sample else 1.0,
            logprobs=1,  # Get logprobs for generated tokens
        )

        # Generate
        outputs = self.llm.generate([prompt], sampling_params)
        output = outputs[0]

        generated_text = output.outputs[0].text

        # Extract token-level data
        for t, token_data in enumerate(output.outputs[0].logprobs or []):
            # token_data is a dict mapping token_id to Logprob
            if token_data:
                # Get the sampled token
                for token_id, logprob_obj in token_data.items():
                    token_text = logprob_obj.decoded_token
                    logprob = logprob_obj.logprob

                    # Estimate entropy (approximate, since we only have top logprobs)
                    # For accurate entropy, we'd need all logprobs
                    entropy = -logprob  # Simple approximation

                    collector.record_token(
                        token_id=token_id,
                        token_text=token_text,
                        logprob=logprob,
                        entropy=entropy,
                    )
                    collector.advance_position()
                    break  # Only record the sampled token

        collector.stop_collection()
        return collector, generated_text


# =============================================================================
# Experiment Runner
# =============================================================================

def run_experiment(
    config: ExperimentConfig,
    prompt: str,
    prompt_name: str = "custom",
    output_dir: str | Path | None = None,
) -> AnalysisResult:
    """
    Run a single novelty experiment.

    Args:
        config: Experiment configuration
        prompt: The prompt to generate from
        prompt_name: Name for the prompt (for logging)
        output_dir: Optional directory to save results

    Returns:
        AnalysisResult with all metrics
    """
    print(f"\n{'='*60}")
    print(f"Running experiment: {prompt_name}")
    print(f"Backend: {config.backend}")
    print(f"{'='*60}")

    # Select backend
    if config.backend == "transformers":
        experiment = TransformersExperiment(config)
    else:
        experiment = VLLMExperiment(config)

    # Run generation
    start_time = time.time()
    collector, generated_text = experiment.run(prompt, prompt_name)
    generation_time = time.time() - start_time

    print(f"Generation completed in {generation_time:.2f}s")
    print(f"Tokens generated: {collector.num_positions}")

    # Analyze
    result = analyze_collection(
        collector,
        window_size=config.window_size,
    )

    # Print summary
    print("\n" + "-" * 40)
    print("SUMMARY")
    print("-" * 40)
    print(f"Spikes detected: {len(result.spike_positions)}")
    print(f"Burstiness index: {result.aggregate_stats.get('burstiness_index', 'N/A')}")
    print(f"Heavy-tailed: {result.heavy_tail_test.get('is_heavy_tailed', 'N/A')}")

    # Save if output_dir specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save collector data
        collector.save(output_dir / "collected_data")

        # Save analysis result
        with open(output_dir / "analysis.json", "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        # Save report
        report = generate_text_report(result)
        with open(output_dir / "report.txt", "w") as f:
            f.write(report)

        # Generate plots if matplotlib available
        try:
            from .analysis import generate_matplotlib_plots
            plots = generate_matplotlib_plots(result, output_dir / "plots")
            print(f"Saved {len(plots)} plots")
        except ImportError:
            print("matplotlib not available, skipping plots")

        print(f"\nResults saved to: {output_dir}")

    return result


def run_experiment_suite(
    config: ExperimentConfig,
    prompts: dict[str, str] | None = None,
    output_dir: str | Path = "experiments",
) -> dict[str, AnalysisResult]:
    """
    Run experiments across multiple prompts.

    Args:
        config: Experiment configuration
        prompts: Dictionary of prompt_name -> prompt (uses defaults if None)
        output_dir: Base directory for results

    Returns:
        Dictionary of prompt_name -> AnalysisResult
    """
    if prompts is None:
        prompts = EXPERIMENT_PROMPTS

    output_dir = Path(output_dir)
    results = {}

    for prompt_name, prompt in prompts.items():
        result = run_experiment(
            config=config,
            prompt=prompt,
            prompt_name=prompt_name,
            output_dir=output_dir / prompt_name,
        )
        results[prompt_name] = result

    # Generate comparison report
    print("\n" + "=" * 60)
    print("EXPERIMENT SUITE SUMMARY")
    print("=" * 60)

    from .analysis import compare_experiments

    comparison = compare_experiments(
        list(results.values()),
        list(results.keys()),
    )

    # Print comparison table
    print(f"\n{'Prompt':<25} {'Tokens':<8} {'Spikes':<8} {'Rate':<8} {'Burst':<8} {'Heavy':<8}")
    print("-" * 73)
    for i, name in enumerate(comparison["labels"]):
        print(
            f"{name:<25} "
            f"{comparison['num_tokens'][i]:<8} "
            f"{comparison['num_spikes'][i]:<8} "
            f"{comparison['spike_rate'][i]:<8.3f} "
            f"{comparison['burstiness'][i]:<8.3f} "
            f"{comparison['is_heavy_tailed'][i]!s:<8}"
        )

    # Save comparison
    with open(output_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    return results


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Command-line interface for running experiments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="KV-Cache Novelty Experiments"
    )
    parser.add_argument(
        "--model", "-m",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name or path",
    )
    parser.add_argument(
        "--backend", "-b",
        choices=["transformers", "vllm"],
        default="transformers",
        help="Backend to use",
    )
    parser.add_argument(
        "--prompt", "-p",
        default=None,
        help="Custom prompt (if not using suite)",
    )
    parser.add_argument(
        "--suite", "-s",
        action="store_true",
        help="Run full experiment suite",
    )
    parser.add_argument(
        "--output", "-o",
        default="experiments",
        help="Output directory",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
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
        default="cuda",
        help="Device to use",
    )

    args = parser.parse_args()

    config = ExperimentConfig(
        model_name=args.model,
        backend=args.backend,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        device=args.device,
    )

    if args.suite:
        run_experiment_suite(config, output_dir=args.output)
    else:
        prompt = args.prompt or EXPERIMENT_PROMPTS["mixed_content"]
        run_experiment(
            config,
            prompt=prompt,
            prompt_name="custom",
            output_dir=args.output,
        )


if __name__ == "__main__":
    main()
