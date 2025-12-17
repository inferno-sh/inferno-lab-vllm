from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch


def _to_sorted_unique(idx: Iterable[int]) -> List[int]:
    return sorted(set(int(i) for i in idx))


def jaccard_overlap(a: Sequence[int], b: Sequence[int]) -> float:
    """
    Compute Jaccard overlap between two index sets. Returns 1.0 when both are
    empty to avoid divide-by-zero.
    """
    set_a = set(int(x) for x in a)
    set_b = set(int(x) for x in b)
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    inter = set_a & set_b
    return len(inter) / len(union)


@dataclass
class StableMaskStats:
    attempted: int = 0
    accepted: int = 0
    rejected: int = 0
    overlap_sum: float = 0.0

    @property
    def avg_overlap(self) -> float:
        return self.overlap_sum / self.attempted if self.attempted else 0.0

    def to_dict(self) -> dict:
        return {
            "attempted": self.attempted,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "avg_overlap": self.avg_overlap,
        }


class StableMaskController:
    """
    Maintains stable channel index selections with hysteresis. Designed to be
    called at fixed token intervals. K and V are tracked separately.
    """

    def __init__(
        self,
        update_interval: int = 64,
        overlap_threshold: float = 0.85,
        enable: bool = True,
    ) -> None:
        self.update_interval = update_interval
        self.overlap_threshold = max(0.0, min(1.0, overlap_threshold))
        self.enable = enable
        self.stable_k: List[int] | None = None
        self.stable_v: List[int] | None = None
        self.last_update_token: int = 0
        self.stats = StableMaskStats()

    def initialize(self, idx_k: Sequence[int], idx_v: Sequence[int]) -> None:
        self.stable_k = _to_sorted_unique(idx_k)
        self.stable_v = _to_sorted_unique(idx_v)

    def should_update(self, generated_tokens: int) -> bool:
        if not self.enable:
            return False
        if generated_tokens <= 0:
            return False
        return generated_tokens - self.last_update_token >= self.update_interval

    def _update_single(
        self, stable: List[int] | None, candidate: Sequence[int]
    ) -> Tuple[List[int], bool, float]:
        if stable is None:
            return _to_sorted_unique(candidate), True, 1.0
        overlap = jaccard_overlap(stable, candidate)
        self.stats.attempted += 1
        self.stats.overlap_sum += overlap
        if overlap >= self.overlap_threshold:
            self.stats.accepted += 1
            return _to_sorted_unique(candidate), True, overlap
        self.stats.rejected += 1
        return stable, False, overlap

    def maybe_update(
        self, generated_tokens: int, idx_k: Sequence[int], idx_v: Sequence[int]
    ) -> Tuple[bool, float, float]:
        """
        Attempt to update stable masks given new candidates. Returns
        (changed_any, overlap_k, overlap_v).
        """
        if not self.enable:
            return False, 0.0, 0.0
        changed_any = False
        self.stable_k, changed_k, overlap_k = self._update_single(self.stable_k, idx_k)
        self.stable_v, changed_v, overlap_v = self._update_single(self.stable_v, idx_v)
        self.last_update_token = generated_tokens
        changed_any = changed_k or changed_v
        return changed_any, overlap_k, overlap_v

    def masks(
        self, head_dim: int, disable_k: bool, disable_v: bool
    ) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
        mask_k = None
        mask_v = None
        if not disable_k and self.stable_k is not None:
            mask_k = torch.zeros(head_dim, dtype=torch.float32)
            mask_k[self.stable_k] = 1.0
        if not disable_v and self.stable_v is not None:
            mask_v = torch.zeros(head_dim, dtype=torch.float32)
            mask_v[self.stable_v] = 1.0
        return mask_k, mask_v


def _run_unit_checks() -> None:
    # Identical masks -> overlap 1.0 accepted.
    ctrl = StableMaskController(overlap_threshold=0.8)
    ctrl.initialize([0, 1], [0, 1])
    changed, ok_k, ok_v = ctrl.maybe_update(64, [0, 1], [0, 1])
    assert changed and ok_k == 1.0 and ok_v == 1.0

    # Disjoint masks rejected.
    ctrl = StableMaskController(overlap_threshold=0.8)
    ctrl.initialize([0, 1], [0, 1])
    changed, ok_k, ok_v = ctrl.maybe_update(64, [2, 3], [2, 3])
    assert not changed and ok_k == 0.0 and ok_v == 0.0

    # Partial overlap at boundary.
    ctrl = StableMaskController(overlap_threshold=0.3)
    ctrl.initialize([0, 1], [0, 1])
    changed, ok_k, ok_v = ctrl.maybe_update(64, [1, 2], [1, 2])
    assert changed and round(ok_k, 3) == 0.333 and round(ok_v, 3) == 0.333


if __name__ == "__main__":
    _run_unit_checks()
    print("stable_mask sanity checks passed.")
