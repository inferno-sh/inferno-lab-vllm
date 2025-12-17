from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from .control import CompressionConfig, ControlLoop, Mode
from .fidelity import fidelity_ok, logits_metrics
from .importance import EMAImportance
from .kvcache_full import FullKVCache
from .kvcache_packed import PackedKVCache

LOG = logging.getLogger(__name__)


def _dtype_bytes(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


class HFBackend:
    """
    HF-eager backend used as a correctness-first reference.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        ).to(self.device)
        if hasattr(self.model.config, "enforce_eager"):
            self.model.config.enforce_eager = True
        self.model.eval()

    def _prefill(
        self, input_ids: torch.Tensor
    ) -> Tuple[FullKVCache, torch.Tensor, Dict]:
        """
        Run a full prefill to build a baseline cache and return logits for the prompt.
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
        past = outputs.past_key_values
        past_seq: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx, (k, v) in enumerate(past):
            # past may be Cache; __iter__ yields (keys, values)
            past_seq.append((k, v))
        num_layers = len(past_seq)
        num_heads = past_seq[0][0].shape[1]
        head_dim = past_seq[0][0].shape[-1]

        cache = FullKVCache(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=past[0][0].dtype,
            device=past[0][0].device,
        )
        # Populate cache token-by-token for importance tracking.
        seq_len = input_ids.shape[1]
        for layer in range(num_layers):
            k = past_seq[layer][0][0]  # [num_heads, seq, head_dim]
            v = past_seq[layer][1][0]
            cache.extend_prefill(layer, k, v)
        return (
            cache,
            outputs.logits[:, -1, :],
            {
                "num_layers": num_layers,
                "num_heads": num_heads,
                "head_dim": head_dim,
            },
        )

    def _materialize_past(self, cache) -> DynamicCache:
        mats = cache.materialize()
        return DynamicCache.from_legacy_cache(tuple((k, v) for k, v in mats))

    def _update_importance_prefill(self, cache: FullKVCache, importance: EMAImportance):
        seq_len = len(cache)
        for t in range(seq_len):
            for layer in range(cache.num_layers):
                k_t = cache.keys[layer][t]
                v_t = cache.vals[layer][t]
                importance.update(
                    layer, head_metric_k=k_t.abs(), head_metric_v=v_t.abs()
                )

    def _cache_to_packed(
        self,
        cache: FullKVCache,
        idx_k: List[torch.Tensor],
        idx_v: List[torch.Tensor],
        cfg: CompressionConfig,
    ) -> PackedKVCache:
        packed = PackedKVCache(
            num_layers=cache.num_layers,
            num_heads=cache.num_heads,
            head_dim=cache.head_dim,
            idx_k=idx_k,
            idx_v=idx_v,
            window=cfg.window,
            dtype=cache.dtype,
            device=cache.device,
            store_full_v=cfg.disable_v_compress,
        )
        seq_len = len(cache)
        for pos in range(seq_len):
            keys = [cache.keys[layer][pos] for layer in range(cache.num_layers)]
            vals = [cache.vals[layer][pos] for layer in range(cache.num_layers)]
            packed.append(pos, keys, vals)
        return packed

    def run_prompt(
        self,
        prompt: str,
        mode: str,
        cfg: CompressionConfig,
        max_new_tokens: int = 64,
        seed: int = 0,
        fidelity_threshold_top1: float = 0.6,
    ) -> Dict:
        torch.manual_seed(seed)
        tokens = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = tokens["input_ids"]
        # prefill baseline
        cache_full, prefill_logits, model_info = self._prefill(input_ids)
        importance = EMAImportance(
            num_layers=model_info["num_layers"],
            num_heads=model_info["num_heads"],
            head_dim=model_info["head_dim"],
            bucket_per_layer=cfg.bucket_per_layer,
            device=self.device,
        )
        self._update_importance_prefill(cache_full, importance)

        control = ControlLoop(cfg, num_layers=model_info["num_layers"])
        if mode == "dynamic":
            control.state.mode = Mode.WARMUP
        else:
            control.state.mode = Mode.FULL

        r_k_effective = (
            cfg.r_k if not cfg.disable_k_compress else model_info["head_dim"]
        )
        r_v_effective = (
            cfg.r_v if not cfg.disable_v_compress else model_info["head_dim"]
        )

        cache = cache_full
        output_tokens: List[int] = []
        fidelity_events: List[Dict] = []

        last_token = input_ids[:, -1:]
        # generation loop
        for step in range(max_new_tokens):
            past = self._materialize_past(cache)
            with torch.no_grad():
                out = self.model(
                    input_ids=last_token,
                    past_key_values=past,
                    use_cache=True,
                    output_attentions=False,
                    return_dict=True,
                )
            logits = out.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            output_tokens.append(int(next_token.item()))

            pkv = out.past_key_values
            keys_new: List[torch.Tensor] = []
            vals_new: List[torch.Tensor] = []
            for layer in range(model_info["num_layers"]):
                k_new = pkv[layer][0][:, :, -1, :].detach().squeeze(0)
                v_new = pkv[layer][1][:, :, -1, :].detach().squeeze(0)
                keys_new.append(k_new)
                vals_new.append(v_new)
                importance.update(
                    layer, head_metric_k=k_new.abs(), head_metric_v=v_new.abs()
                )
            pos = len(cache)
            cache.append_step(pos, keys_new, vals_new)

            global_step = len(cache)
            if mode == "dynamic" and control.should_update_selection(global_step):
                idx_k, idx_v = importance.compute_topk(r_k_effective, r_v_effective)
                stability = importance.update_stability(
                    idx_k,
                    idx_v,
                    r_k_effective,
                    r_v_effective,
                    retention_threshold=cfg.retention_threshold,
                    n_stable=cfg.n_stable,
                )
                control.state.stability = stability
                control.attach_indices(idx_k, idx_v)
                if control.maybe_switch_to_compressed(stability):
                    LOG.info("Switching to compressed cache at step %d", global_step)
                    cache = self._cache_to_packed(cache, idx_k, idx_v, cfg)
            if (
                mode == "dynamic"
                and control.should_check_fidelity(global_step)
                and isinstance(cache, PackedKVCache)
            ):
                # normal mode logits already computed (logits)
                baseline_past = self._materialize_past(cache)
                with torch.no_grad():
                    baseline_out = self.model(
                        input_ids=last_token,
                        past_key_values=baseline_past,
                        use_cache=True,
                        output_attentions=False,
                        return_dict=True,
                    )
                baseline_logits = baseline_out.logits[:, -1, :]
                metrics = logits_metrics(logits, baseline_logits)
                ok = fidelity_ok(metrics, top1_threshold=fidelity_threshold_top1)
                metrics["ok"] = ok
                fidelity_events.append(metrics)
                if not ok:
                    LOG.warning("Fidelity check failed, backing off to full mode.")
                    control.reset_to_full()
                    cache = cache.to_full()  # back to full cache
            last_token = next_token

        text_out = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
        return {
            "prompt": prompt,
            "generated_text": text_out,
            "tokens": output_tokens,
            "mode": control.state.mode.name,
            "cache_bytes": cache.memory_bytes(),
            "cache_type": type(cache).__name__,
            "fidelity_events": fidelity_events,
            "importance": importance.summary(),
            "model_info": model_info,
        }
