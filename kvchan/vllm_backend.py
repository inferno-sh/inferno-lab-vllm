from __future__ import annotations

import logging
import multiprocessing as mp
import os
from typing import Dict, List, Optional, Tuple

import torch
from vllm import LLM, SamplingParams
from vllm.sampling_params import RequestOutputKind
from transformers import AutoConfig

from .control import CompressionConfig, ControlLoop, Mode
from .fidelity import fidelity_ok
from .stable_mask import StableMaskController, StableMaskStats

LOG = logging.getLogger(__name__)


def _build_indices(head_dim: int, keep: int) -> List[int]:
    """Build naive indices (first N channels). Used as fallback."""
    keep = min(keep, head_dim)
    return list(range(keep))


def _indices_to_mask(indices: List[int], head_dim: int) -> torch.Tensor:
    """Convert channel indices to a binary mask tensor."""
    mask = torch.zeros(head_dim, dtype=torch.float32)
    for idx in indices:
        if 0 <= idx < head_dim:
            mask[idx] = 1.0
    return mask


def _first_mismatch(a: List[int], b: List[int]) -> Optional[int]:
    limit = min(len(a), len(b))
    for i in range(limit):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return limit
    return None


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
        # Disable prefix caching to avoid cross-request nondeterminism during
        # step-driven runs and mask debugging.
        llm_kwargs["enable_prefix_caching"] = False
        # Disable torch.compile to allow runtime mask application.
        # The unified_attention custom op is compiled away otherwise,
        # preventing dynamic mask updates during inference.
        llm_kwargs["enforce_eager"] = True
        self.llm = LLM(model=model_name, **llm_kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        self.model_name = model_name

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

    def _configure_importance(
        self,
        r_k: int,
        r_v: int,
        beta: float = 0.98,
        update_interval: int = 32,
        warmup_tokens: int = 64,
    ) -> None:
        """Configure importance tracking on workers."""
        self.llm.llm_engine.collective_rpc(
            "configure_channel_importance",
            args=(r_k, r_v, beta, update_interval, warmup_tokens),
        )

    def _step_importance(self) -> List[Dict]:
        """Step importance tracking and return summaries from all workers."""
        return self.llm.llm_engine.collective_rpc("step_channel_importance")

    def _get_importance_masks(self) -> Optional[Tuple[List[float], List[float]]]:
        """Get current importance-based masks from workers."""
        results = self.llm.llm_engine.collective_rpc("get_channel_importance_masks")
        if results and results[0] is not None:
            return results[0]
        return None

    def _disable_importance(self) -> None:
        """Disable importance tracking on workers."""
        self.llm.llm_engine.collective_rpc("disable_channel_importance")

    def run_prompt(
        self,
        prompt: str,
        mode: str,
        cfg: CompressionConfig,
        max_new_tokens: int = 64,
        seed: int = 0,
    ) -> Dict:
        baseline_params = SamplingParams(
            temperature=0.0, max_tokens=max_new_tokens, seed=seed
        )
        masked_params = SamplingParams(
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
        head_sizes = self.llm.llm_engine.collective_rpc("get_head_size")
        head_dim = int(head_sizes[0]) if head_sizes else cfg.r_k + cfg.r_v
        min_k_keep = int(max(1, head_dim * cfg.min_k_keep_ratio))
        force_full_k = head_dim <= 64
        k_keep = head_dim if cfg.disable_k_compress else cfg.r_k
        k_keep = min(k_keep, head_dim)
        if not cfg.disable_k_compress:
            k_keep = max(k_keep, min_k_keep)
        if force_full_k:
            k_keep = head_dim
        v_keep = head_dim if cfg.disable_v_compress else cfg.r_v
        v_keep = min(v_keep, head_dim)

        # First run: full baseline
        prompt_tokens = len(self.tokenizer(prompt)["input_ids"])

        def run_step_no_masks(sp: SamplingParams) -> Tuple[str, int, List[int]]:
            sp.output_kind = RequestOutputKind.FINAL_ONLY
            self.llm._validate_and_add_requests(
                prompts=[prompt],
                params=[sp],
                use_tqdm=False,
                lora_request=None,
                priority=None,
                tokenization_kwargs=None,
            )
            text_out = ""
            gen_tokens = 0
            token_ids: List[int] = []
            while self.llm.llm_engine.has_unfinished_requests():
                step_outputs = self.llm.llm_engine.step()
                for out in step_outputs:
                    if not out.outputs:
                        continue
                    item = out.outputs[0]
                    text_out = item.text
                    gen_tokens = len(getattr(item, "token_ids", []))
                    token_ids = getattr(item, "token_ids", []) or token_ids
            return text_out, gen_tokens, token_ids

        self._clear_masks()
        baseline_token_ids: List[int] = []
        if cfg.debug_step_baseline:
            full_text, full_gen_tokens, baseline_token_ids = run_step_no_masks(
                baseline_params
            )
        else:
            full_outputs = self.llm.generate(prompt, baseline_params)
            full_item = full_outputs[0].outputs[0]
            full_text = full_item.text
            baseline_token_ids = getattr(full_item, "token_ids", []) or []
            full_gen_tokens = len(baseline_token_ids)

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
                "head_dim": head_dim,
                "mask_k_effective": len(
                    _build_indices(
                        head_dim, head_dim if cfg.disable_k_compress else k_keep
                    )
                ),
                "mask_v_effective": len(
                    _build_indices(
                        head_dim, head_dim if cfg.disable_v_compress else v_keep
                    )
                ),
                "first_mismatch_token_idx": None,
                "backoff_token_idx": None,
            }

        # Configure importance-based or naive channel selection
        use_importance = cfg.use_importance and not force_full_k
        importance_stats: Dict = {}

        if use_importance:
            # Configure importance tracking with warmup
            LOG.info(
                "Configuring importance tracking: r_k=%d, r_v=%d, warmup=%d",
                k_keep,
                v_keep,
                cfg.importance_warmup_tokens,
            )
            self._configure_importance(
                r_k=k_keep,
                r_v=v_keep,
                beta=cfg.importance_beta,
                update_interval=cfg.update_interval,
                warmup_tokens=cfg.importance_warmup_tokens,
            )
            # Start with all-ones masks during warmup phase
            mask_k = torch.ones(head_dim, dtype=torch.float32)
            mask_v = torch.ones(head_dim, dtype=torch.float32)
            cand_idx_k = list(range(head_dim))
            cand_idx_v = list(range(head_dim))
        else:
            cand_idx_k = _build_indices(head_dim, k_keep)
            cand_idx_v = _build_indices(head_dim, v_keep)

        stable_ctrl.initialize(cand_idx_k, cand_idx_v)

        if not use_importance:
            # Build masks from naive indices
            mask_k, mask_v = stable_ctrl.masks(
                head_dim, cfg.disable_k_compress, cfg.disable_v_compress
            )
            if force_full_k and v_keep == head_dim:
                mask_k = torch.ones_like(mask_k) if mask_k is not None else None
                mask_v = torch.ones_like(mask_v) if mask_v is not None else None

        if cfg.debug_force_all_ones_mask:
            mask_k = torch.ones_like(mask_k) if mask_k is not None else None
            mask_v = torch.ones_like(mask_v) if mask_v is not None else None
        apply_masks = (mask_k is not None or mask_v is not None) and (
            not cfg.debug_skip_masks
        )

        # Apply initial masks and drive the engine manually for mid-gen updates.
        if apply_masks:
            self._apply_masks(mask_k, mask_v)
        else:
            self._clear_masks()
        control.state.mode = Mode.COMPRESSED
        masked_params.output_kind = (
            RequestOutputKind.FINAL_ONLY
            if cfg.debug_skip_masks
            else RequestOutputKind.CUMULATIVE
        )
        self.llm._validate_and_add_requests(
            prompts=[prompt],
            params=[masked_params],
            use_tqdm=False,
            lora_request=None,
            priority=None,
            tokenization_kwargs=None,
        )

        masked_text = ""
        masked_gen_tokens = 0
        prompt_token_count = prompt_tokens
        masked_token_ids: List[int] = []
        while self.llm.llm_engine.has_unfinished_requests():
            step_outputs = self.llm.llm_engine.step()

            # Step importance tracking (triggers mask updates via callback)
            if use_importance:
                step_stats = self._step_importance()
                if step_stats:
                    importance_stats = step_stats[0]  # From first worker

            for out in step_outputs:
                if not out.outputs:
                    continue
                masked_item = out.outputs[0]
                masked_text = masked_item.text
                masked_gen_tokens = len(getattr(masked_item, "token_ids", []))
                masked_token_ids = (
                    getattr(masked_item, "token_ids", []) or masked_token_ids
                )
                if masked_item.finished():
                    prompt_token_count = (
                        len(out.prompt_token_ids)
                        if out.prompt_token_ids
                        else prompt_tokens
                    )
                # Only use stable_ctrl for non-importance mode
                if not use_importance and stable_ctrl.should_update(masked_gen_tokens):
                    cand_idx_k = _build_indices(head_dim, k_keep)
                    cand_idx_v = _build_indices(head_dim, v_keep)
                    changed, ov_k, ov_v = stable_ctrl.maybe_update(
                        masked_gen_tokens, cand_idx_k, cand_idx_v
                    )
                    LOG.debug(
                        "Stable mask update at %d tokens: changed=%s, overlap_k=%.3f, overlap_v=%.3f",
                        masked_gen_tokens,
                        changed,
                        ov_k,
                        ov_v,
                    )
                    if changed and apply_masks:
                        mask_k, mask_v = stable_ctrl.masks(
                            head_dim, cfg.disable_k_compress, cfg.disable_v_compress
                        )
                        self._apply_masks(mask_k, mask_v)

        # Clean up
        self._clear_masks()
        if use_importance:
            self._disable_importance()

        fidelity = {
            "match": full_text == masked_text,
            "len_full": len(full_text),
            "len_masked": len(masked_text),
        }
        backoff = not fidelity["match"]
        first_mismatch = None
        backoff_token_idx = None
        if backoff:
            first_mismatch = _first_mismatch(baseline_token_ids, masked_token_ids)
            backoff_token_idx = first_mismatch
        if backoff:
            control.reset_to_full()

        stats: StableMaskStats = stable_ctrl.stats
        return {
            "prompt": prompt,
            "text": full_text if backoff else masked_text,
            "baseline_text": full_text,
            "masked_text": masked_text,
            "mode": control.state.mode.name,
            "mask_applied": apply_masks,
            "backoff": backoff,
            "mask_k_kept": k_keep,
            "mask_v_kept": v_keep,
            "fidelity": fidelity,
            "head_dim": head_dim,
            "mask_k_effective": k_keep if use_importance else len(cand_idx_k),
            "mask_v_effective": v_keep if use_importance else len(cand_idx_v),
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
            "first_mismatch_token_idx": first_mismatch,
            "backoff_token_idx": backoff_token_idx,
            "use_importance": use_importance,
            "importance_stats": importance_stats,
        }
