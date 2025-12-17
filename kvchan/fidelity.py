from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def logits_metrics(
    logits_a: torch.Tensor, logits_b: torch.Tensor, top_k: int = 5
) -> Dict[str, float]:
    """
    Compute simple fidelity metrics between two logits tensors shaped [batch, vocab].
    """
    a_topk = torch.topk(logits_a, k=top_k, dim=-1)
    b_topk = torch.topk(logits_b, k=top_k, dim=-1)

    a_top1 = a_topk.indices[..., 0]
    b_top1 = b_topk.indices[..., 0]
    top1_match = (a_top1 == b_top1).float().mean().item()
    # top-1 of baseline inside top-k of normal
    baseline_in_a = (
        (b_top1.unsqueeze(-1) == a_topk.indices).any(dim=-1).float().mean().item()
    )

    # margin drift
    def margin(x):
        top2 = torch.topk(x, k=2, dim=-1)
        return (top2.values[..., 0] - top2.values[..., 1]).mean().item()

    margin_a = margin(logits_a)
    margin_b = margin(logits_b)
    # optional: KL over top-k
    probs_a = F.softmax(logits_a, dim=-1)
    probs_b = F.softmax(logits_b, dim=-1)
    kl_ab = F.kl_div(probs_a.log(), probs_b, reduction="batchmean").item()
    return {
        "top1_match": top1_match,
        "baseline_in_topk": baseline_in_a,
        "margin_a": margin_a,
        "margin_b": margin_b,
        "kl_ab": kl_ab,
    }


def fidelity_ok(
    metrics: Dict[str, float], top1_threshold: float = 0.6, margin_drop: float = 0.5
) -> bool:
    if metrics["baseline_in_topk"] < top1_threshold:
        return False
    if metrics["margin_a"] + margin_drop < metrics["margin_b"]:
        return False
    return True
