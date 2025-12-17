from __future__ import annotations

import logging
import multiprocessing as mp
import os
from typing import Dict, List, Optional

import torch
from vllm import LLM, SamplingParams

from .control import CompressionConfig, ControlLoop, Mode
from .fidelity import fidelity_ok

LOG = logging.getLogger(__name__)


def _find_attention_modules(model) -> List[torch.nn.Module]:
    attn = []
    for mod in model.modules():
        if hasattr(mod, "set_channel_masks"):
            attn.append(mod)
    return attn


def _build_mask(head_dim: int, keep: int) -> torch.Tensor:
    mask = torch.zeros(head_dim, dtype=torch.float32)
    keep = min(keep, head_dim)
    if keep > 0:
        mask[:keep] = 1.0
    return mask


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
    ):
        # Ensure spawn start method to avoid CUDA fork issues.
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        self.llm = LLM(model=model_name)

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

        # First run: full baseline
        self._clear_masks()
        full_outputs = self.llm.generate(prompt, sampling_params)
        full_text = full_outputs[0].outputs[0].text

        if mode == "full":
            return {
                "prompt": prompt,
                "text": full_text,
                "mode": control.state.mode.name,
                "mask_applied": False,
                "backoff": False,
            }

        # Build fixed masks as a stand-in for compressed mode.
        head_sizes = self.llm.llm_engine.collective_rpc("get_head_size")
        head_dim = int(head_sizes[0]) if head_sizes else cfg.r_k + cfg.r_v
        mask_k = _build_mask(
            head_dim, cfg.r_k if not cfg.disable_k_compress else head_dim
        )
        mask_v = _build_mask(
            head_dim, cfg.r_v if not cfg.disable_v_compress else head_dim
        )

        # Apply mask and generate masked text.
        self._apply_masks(mask_k, mask_v)
        control.state.mode = Mode.COMPRESSED
        masked_outputs = self.llm.generate(prompt, sampling_params)
        masked_text = masked_outputs[0].outputs[0].text

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
        }
