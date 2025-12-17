from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

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

    probe_p = sub.add_parser(
        "train-probe", help="Collect importance stats without compression."
    )
    probe_p.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    probe_p.add_argument("--rK", type=int, default=32)
    probe_p.add_argument("--rV", type=int, default=128)
    probe_p.add_argument("--output", type=str, default="kvchan/outputs/probe.jsonl")
    return parser.parse_args()


def build_backend(name: str):
    if name == "hf":
        return HFBackend()
    return VLLMBackend()


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
        )
        backend = build_backend(args.backend)
        results = []
        for name in PROMPT_ORDER:
            prompt = PROMPTS[name]
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
        backend = build_backend(args.backend)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for name in PROMPT_ORDER:
                prompt = PROMPTS[name]
                res = backend.run_prompt(
                    prompt=prompt,
                    mode="full",
                    cfg=cfg,
                    max_new_tokens=0,
                )
                res["prompt_name"] = name
                f.write(json.dumps(res) + "\n")
        LOG.info("Probe stats written to %s", out_path)


if __name__ == "__main__":
    run()
