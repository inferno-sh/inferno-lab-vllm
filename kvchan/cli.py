from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

from .control import CompressionConfig
from .hf_backend import HFBackend
from .prompts import PROMPT_ORDER, PROMPTS
from .vllm_backend import VLLMBackend

LOG = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dynamic KV-channel compression runner"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run generation with dynamic compression.")
    run_p.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    run_p.add_argument("--mode", choices=["full", "dynamic"], default="dynamic")
    run_p.add_argument("--W", type=int, default=128, help="Uncompressed window size.")
    run_p.add_argument("--rK", type=int, default=32, help="Top-r dims for keys.")
    run_p.add_argument("--rV", type=int, default=128, help="Top-r dims for values.")
    run_p.add_argument("--disable_k_compress", action="store_true")
    run_p.add_argument("--disable_v_compress", action="store_true")
    run_p.add_argument("--max_tokens", type=int, default=64)
    run_p.add_argument("--output", type=str, default="kvchan/outputs/run.jsonl")
    run_p.add_argument("--seed", type=int, default=0)
    run_p.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    run_p.add_argument(
        "--prompt_file",
        type=str,
        help="Optional JSONL file with custom prompts (fields: name, prompt).",
    )
    run_p.add_argument(
        "--check_every", type=int, default=64, help="Fidelity check interval."
    )
    run_p.add_argument(
        "--update_interval", type=int, default=32, help="Importance update interval."
    )
    run_p.add_argument(
        "--n_stable",
        type=int,
        default=20,
        help="Consecutive stable intervals required before compressing.",
    )
    run_p.add_argument(
        "--retention_threshold",
        type=float,
        default=0.9,
        help="Retention threshold for stability.",
    )
    run_p.add_argument(
        "--min_k_keep_ratio",
        type=float,
        default=0.75,
        help="Minimum ratio of head_dim to keep for keys (guardrail).",
    )
    run_p.add_argument(
        "--debug_force_all_ones_mask",
        action="store_true",
        help="Force masks to all-ones for debugging masking correctness.",
    )
    run_p.add_argument(
        "--debug_skip_masks",
        action="store_true",
        help="Skip applying masks but still use step loop (debug correctness).",
    )
    run_p.add_argument(
        "--debug_step_baseline",
        action="store_true",
        help="Run FULL baseline via step loop (no masks) for apples-to-apples comparison.",
    )
    run_p.add_argument(
        "--stable_mask_enable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable stable per-head masks with hysteresis.",
    )
    run_p.add_argument(
        "--stable_mask_update_interval",
        type=int,
        default=64,
        help="Update cadence (tokens) for stable mask hysteresis.",
    )
    run_p.add_argument(
        "--stable_mask_overlap_threshold",
        type=float,
        default=0.85,
        help="Jaccard overlap threshold for accepting new masks.",
    )
    run_p.add_argument(
        "--use_importance",
        action="store_true",
        help="Use importance-based channel selection (EMA of magnitudes) instead of naive first-N.",
    )
    run_p.add_argument(
        "--importance_beta",
        type=float,
        default=0.98,
        help="EMA decay factor for importance tracking (higher = slower adaptation).",
    )
    run_p.add_argument(
        "--importance_warmup_tokens",
        type=int,
        default=64,
        help="Tokens to wait before enabling compression (warmup period).",
    )
    probe_p = sub.add_parser(
        "train-probe", help="Collect importance stats without compression."
    )
    probe_p.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    probe_p.add_argument("--rK", type=int, default=32)
    probe_p.add_argument("--rV", type=int, default=128)
    probe_p.add_argument("--output", type=str, default="kvchan/outputs/probe.jsonl")
    probe_p.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    probe_p.add_argument(
        "--prompt_file",
        type=str,
        help="Optional JSONL file with custom prompts (fields: name, prompt).",
    )
    return parser.parse_args()


def build_backend(name: str, model_name: str):
    if name == "hf":
        return HFBackend(model_name=model_name)
    return VLLMBackend(model_name=model_name)


def load_prompts(prompt_file: str | None) -> Tuple[list[str], Dict[str, str]]:
    if not prompt_file:
        return list(PROMPT_ORDER), PROMPTS
    path = Path(prompt_file)
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    prompts: Dict[str, str] = {}
    order: list[str] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            name = row.get("name")
            prompt = row.get("prompt")
            if name is None or prompt is None:
                raise ValueError(f"Invalid row in prompt file {prompt_file}: {row}")
            prompts[name] = prompt
            order.append(name)
    LOG.info("Loaded %d prompts from %s", len(order), prompt_file)
    return order, prompts


def run():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    if args.command == "run":
        cfg = CompressionConfig(
            r_k=args.rK,
            r_v=args.rV,
            window=args.W,
            disable_k_compress=args.disable_k_compress,
            disable_v_compress=args.disable_v_compress,
            fidelity_every=args.check_every,
            update_interval=args.update_interval,
            n_stable=args.n_stable,
            retention_threshold=args.retention_threshold,
            stable_mask_enable=args.stable_mask_enable,
            stable_mask_update_interval=args.stable_mask_update_interval,
            stable_mask_overlap_threshold=args.stable_mask_overlap_threshold,
            min_k_keep_ratio=args.min_k_keep_ratio,
            debug_force_all_ones_mask=args.debug_force_all_ones_mask,
            debug_skip_masks=args.debug_skip_masks,
            debug_step_baseline=args.debug_step_baseline,
            use_importance=args.use_importance,
            importance_beta=args.importance_beta,
            importance_warmup_tokens=args.importance_warmup_tokens,
        )
        backend = build_backend(args.backend, args.model)
        results = []
        prompt_order, prompts = load_prompts(args.prompt_file)
        for name in prompt_order:
            prompt = prompts[name]
            LOG.info("Running prompt %s", name)
            res = backend.run_prompt(
                prompt=prompt,
                mode=args.mode,
                cfg=cfg,
                max_new_tokens=args.max_tokens,
                seed=args.seed,
            )
            res["prompt_name"] = name
            results.append(res)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for row in results:
                f.write(json.dumps(row) + "\n")
        LOG.info("Wrote results to %s", out_path)
        return

    if args.command == "train-probe":
        cfg = CompressionConfig(
            r_k=args.rK,
            r_v=args.rV,
            window=0,
        )
        backend = build_backend(args.backend, args.model)
        prompt_order, prompts = load_prompts(args.prompt_file)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for name in prompt_order:
                prompt = prompts[name]
                res = backend.run_prompt(
                    prompt=prompt,
                    mode="full",
                    cfg=cfg,
                    max_new_tokens=0,
                    seed=args.seed,
                )
                res["prompt_name"] = name
                f.write(json.dumps(res) + "\n")
        LOG.info("Probe stats written to %s", out_path)


if __name__ == "__main__":
    run()
