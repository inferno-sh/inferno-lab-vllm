from __future__ import annotations

import logging
import multiprocessing as mp
import os
from typing import Dict, List, Optional, Tuple

import torch
from vllm import LLM, SamplingParams
from transformers import AutoConfig

from .control import CompressionConfig, ControlLoop, Mode
from .fidelity import fidelity_ok
from .stable_mask import StableMaskController, StableMaskStats

LOG = logging.getLogger(__name__)


def _build_indices(head_dim: int, keep: int) -> List[int]:
    keep = min(keep, head_dim)
    return list(range(keep))


class VLLMBackend:
    """
    vLLM backend with channel masking support. This keeps the existing KV
    layout and uses attention-layer masks to zero out unselected dimensions.
    It does not save VRAM but allows toggling masks to exercise backoff logic.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: Optional[str] = "cpu",
        max_model_len: int | None = 4096,
        gpu_memory_utilization: float = 0.85,
    ):
        # Ensure spawn start method to avoid CUDA fork issues.
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        llm_kwargs = {}
        if max_model_len is not None:
            safe_len = max_model_len
            try:
                cfg = AutoConfig.from_pretrained(model_name)
                derived = getattr(cfg, "max_position_embeddings", None)
                if derived is None:
                    derived = getattr(cfg, "model_max_length", None)
                if derived is not None:
                    safe_len = min(max_model_len, int(derived))
            except Exception as exc:  # pragma: no cover - best-effort guard
                LOG.debug("Failed to read config for %s: %s", model_name, exc)
            llm_kwargs["max_model_len"] = safe_len
        llm_kwargs["gpu_memory_utilization"] = gpu_memory_utilization
        self.llm = LLM(model=model_name, **llm_kwargs)
        self.tokenizer = self.llm.get_tokenizer()

    def _apply_masks(self, mask_k: torch.Tensor | None, mask_v: torch.Tensor | None):
        """
        Dispatch mask application into worker processes via apply_model.
        """

        mask_k_list = mask_k.tolist() if mask_k is not None else None
        mask_v_list = mask_v.tolist() if mask_v is not None else None
        self.llm.llm_engine.collective_rpc(
            "set_channel_masks", args=(mask_k_list, mask_v_list)
        )

    def _clear_masks(self):
        self._apply_masks(None, None)

    def run_prompt(
        self,
        prompt: str,
        mode: str,
        cfg: CompressionConfig,
        max_new_tokens: int = 64,
        seed: int = 0,
    ) -> Dict:
        sampling_params = SamplingParams(
            temperature=0.0, max_tokens=max_new_tokens, seed=seed
        )
        control = ControlLoop(cfg, num_layers=0)
        control.state.mode = Mode.FULL
        stable_cfg_enable = cfg.stable_mask_enable if mode == "dynamic" else False
        overlap_threshold = min(0.9, max(0.8, cfg.stable_mask_overlap_threshold))
        stable_ctrl = StableMaskController(
            update_interval=cfg.stable_mask_update_interval,
            overlap_threshold=overlap_threshold,
            enable=stable_cfg_enable,
        )

        # First run: full baseline
        self._clear_masks()
        full_outputs = self.llm.generate(prompt, sampling_params)
        full_item = full_outputs[0].outputs[0]
        full_text = full_item.text
        prompt_tokens = len(self.tokenizer(prompt)["input_ids"])
        full_gen_tokens = len(getattr(full_item, "token_ids", []))

        if mode == "full":
            return {
                "prompt": prompt,
                "text": full_text,
                "mode": control.state.mode.name,
                "mask_applied": False,
                "backoff": False,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": full_gen_tokens,
                "total_tokens": prompt_tokens + full_gen_tokens,
                "text_chars": len(full_text),
                "stable_mask_enable": stable_cfg_enable,
                "stable_mask_update_interval": cfg.stable_mask_update_interval,
                "stable_mask_overlap_threshold": overlap_threshold,
                "stable_mask_updates_attempted": 0,
                "stable_mask_updates_accepted": 0,
                "stable_mask_updates_rejected": 0,
                "stable_mask_avg_overlap": 0.0,
            }

        # Build fixed masks as a stand-in for compressed mode.
        head_sizes = self.llm.llm_engine.collective_rpc("get_head_size")
        head_dim = int(head_sizes[0]) if head_sizes else cfg.r_k + cfg.r_v
        cand_idx_k = _build_indices(
            head_dim, cfg.r_k if not cfg.disable_k_compress else head_dim
        )
        cand_idx_v = _build_indices(
            head_dim, cfg.r_v if not cfg.disable_v_compress else head_dim
        )
        stable_ctrl.initialize(cand_idx_k, cand_idx_v)
        mask_k, mask_v = stable_ctrl.masks(
            head_dim, cfg.disable_k_compress, cfg.disable_v_compress
        )

        # Apply initial stable masks and generate.
        self._apply_masks(mask_k, mask_v)
        control.state.mode = Mode.COMPRESSED
        masked_outputs = self.llm.generate(prompt, sampling_params)
        masked_item = masked_outputs[0].outputs[0]
        masked_text = masked_item.text
        masked_gen_tokens = len(getattr(masked_item, "token_ids", []))

        # Simulate periodic updates for statistics (masks remain stable during
        # the single-shot generate call).
        if stable_ctrl.enable and cfg.stable_mask_update_interval > 0:
            tokens_checkpoint = cfg.stable_mask_update_interval
            while tokens_checkpoint <= masked_gen_tokens:
                cand_idx_k = _build_indices(
                    head_dim, cfg.r_k if not cfg.disable_k_compress else head_dim
                )
                cand_idx_v = _build_indices(
                    head_dim, cfg.r_v if not cfg.disable_v_compress else head_dim
                )
                changed, ov_k, ov_v = stable_ctrl.maybe_update(
                    tokens_checkpoint, cand_idx_k, cand_idx_v
                )
                LOG.debug(
                    "Stable mask update at %d tokens: changed=%s, overlap_k=%.3f, overlap_v=%.3f",
                    tokens_checkpoint,
                    changed,
                    ov_k,
                    ov_v,
                )
                tokens_checkpoint += cfg.stable_mask_update_interval

        # Clear masks to avoid leaking to future runs.
        self._clear_masks()

        fidelity = {
            "match": full_text == masked_text,
            "len_full": len(full_text),
            "len_masked": len(masked_text),
        }
        backoff = not fidelity["match"]
        if backoff:
            control.reset_to_full()

        stats: StableMaskStats = stable_ctrl.stats
        return {
            "prompt": prompt,
            "text": full_text if backoff else masked_text,
            "baseline_text": full_text,
            "masked_text": masked_text,
            "mode": control.state.mode.name,
            "mask_applied": True,
            "backoff": backoff,
            "mask_k_kept": cfg.r_k,
            "mask_v_kept": cfg.r_v,
            "fidelity": fidelity,
            "prompt_tokens": prompt_tokens,
            "generated_tokens_full": full_gen_tokens,
            "generated_tokens_masked": masked_gen_tokens,
            "generated_tokens": full_gen_tokens if backoff else masked_gen_tokens,
            "total_tokens": (prompt_tokens + full_gen_tokens)
            if backoff
            else (prompt_tokens + masked_gen_tokens),
            "text_chars": len(full_text if backoff else masked_text),
            "text_chars_full": len(full_text),
            "text_chars_masked": len(masked_text),
            "stable_mask_enable": stable_cfg_enable,
            "stable_mask_update_interval": cfg.stable_mask_update_interval,
            "stable_mask_overlap_threshold": overlap_threshold,
            "stable_mask_updates_attempted": stats.attempted,
            "stable_mask_updates_accepted": stats.accepted,
            "stable_mask_updates_rejected": stats.rejected,
            "stable_mask_avg_overlap": stats.avg_overlap,
        }
