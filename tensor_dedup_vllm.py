#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Nitin Gupta (ngupta@inferno.sh)
"""
Tensor Dedup Multi-Model Demo

Loads multiple models simultaneously, keeps them in the same process, and
demonstrates that identical tensors automatically share storage via the global
dedup registry.

Usage:
    VLLM_ENABLE_V1_MULTIPROCESSING=0 python tensor_dedup_vllm.py
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple

import logging

import torch
from vllm import LLM, SamplingParams
from vllm.config.load import TensorDedupConfig
from vllm.model_executor.model_loader.tensor_dedup import TensorDedupRegistry

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROMPT = "Tell me a story about dinosaurs."
MAX_TOKENS = 80
TEMPERATURE = 0.6
MODEL_IDS: List[str] = [
    "Qwen/Qwen2-0.5B",
    "Qwen/Qwen2-0.5B-Instruct",
]
GPU_MEMORY_UTILIZATION = 0.45
ENABLE_TENSOR_DEDUP = os.environ.get("ENABLE_TENSOR_DEDUP", "1") == "1"
DEDUP_HASH = os.environ.get("DEDUP_HASH", "blake2b")
DEDUP_VERIFY = os.environ.get("DEDUP_VERIFY", "0") == "1"
DEDUP_MIN_BYTES = int(os.environ.get("DEDUP_MIN_BYTES", str(4 * 1024 * 1024)))
os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")
logging.getLogger("vllm.model_executor.model_loader.default_loader").setLevel(
    logging.INFO
)
logging.getLogger("vllm.model_executor.model_loader.gguf_loader").setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format="%(message)s")

VRAM_USAGE: list[tuple[str, float, float]] = []


def _require_single_process() -> None:
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING", "1") != "0":
        print(
            "\nERROR: This demo requires single-process mode.\n"
            "Set VLLM_ENABLE_V1_MULTIPROCESSING=0 before running:\n"
            "    VLLM_ENABLE_V1_MULTIPROCESSING=0 python tensor_dedup_vllm.py\n"
        )
        sys.exit(1)


def _print_section(title: str) -> None:
    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'=' * 80}")


def _report_vram(label: str, model_id: str | None = None) -> tuple[float, float]:
    if not torch.cuda.is_available():
        return (0.0, 0.0)
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    print(f"{label}: allocated={allocated:.2f} GB, reserved={reserved:.2f} GB")
    if model_id is not None:
        VRAM_USAGE.append((model_id, allocated, reserved))
    return allocated, reserved


def _tensor_dedup_config() -> TensorDedupConfig | None:
    if not ENABLE_TENSOR_DEDUP:
        return None
    return TensorDedupConfig(
        enabled=True,
        hash_algorithm=DEDUP_HASH,
        verify_bytes=DEDUP_VERIFY,
        min_tensor_bytes=DEDUP_MIN_BYTES,
    )


def load_model(model_id: str) -> Tuple[str, LLM]:
    _print_section(f"Loading model: {model_id}")
    dedup_cfg = _tensor_dedup_config()
    llm = LLM(
        model=model_id,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        disable_log_stats=True,
        tensor_dedup=dedup_cfg,
    )
    _report_vram(f"After loading {model_id}", model_id=model_id)
    return model_id, llm


def report_vram_summary() -> None:
    if not VRAM_USAGE:
        return
    _print_section("Per-model VRAM usage")
    for model_id, allocated, reserved in VRAM_USAGE:
        print(
            f"{model_id}: allocated={allocated:.2f} GB, reserved={reserved:.2f} GB"
        )


def report_dedup_summary(num_models: int) -> None:
    if not ENABLE_TENSOR_DEDUP:
        return
    _print_section("Tensor dedup registry summary")
    stats = TensorDedupRegistry.get_stats()
    entries = int(stats.get("entries", 0))
    reused = int(stats.get("reused", 0))
    if entries == 0:
        print("No deduplicated tensors were registered.")
        return
    shared_bytes = stats.get("total_bytes", 0.0)
    shared_mb = shared_bytes / (1024**2)
    shared_gb = shared_mb / 1024
    saved_bytes = stats.get("saved_bytes", 0.0)
    saved_mb = saved_bytes / (1024**2)
    saved_gb = saved_mb / 1024
    # Percentage: savings relative to what would have been loaded without dedup
    # i.e. saved_bytes / (shared_bytes * num_models) * 100
    percent = (
        (saved_bytes / (shared_bytes * num_models)) * 100
        if shared_bytes > 0 and num_models > 0
        else 0.0
    )
    print(f"Unique tensors registered: {entries}")
    print(f"Tensors reused (duplicates avoided): {reused}")
    print(f"Total unique tensor size: {shared_mb:.2f} MB ({shared_gb:.2f} GB)")
    print(f"Total savings: {saved_mb:.2f} MB ({saved_gb:.2f} GB) ≈ {percent:.1f}%")


def generate_from_models(models: List[Tuple[str, LLM]]) -> None:
    sampling = SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
    for name, llm in models:
        _print_section(f"Generating with {name}")
        outputs = llm.generate(PROMPT, sampling)
        print(outputs[0].outputs[0].text.strip())


def main() -> None:
    _require_single_process()
    TensorDedupRegistry.clear()

    loaded_models: List[Tuple[str, LLM]] = [
        load_model(model_id) for model_id in MODEL_IDS
    ]

    report_vram_summary()
    report_dedup_summary(num_models=len(loaded_models))
    generate_from_models(loaded_models)


if __name__ == "__main__":
    main()
