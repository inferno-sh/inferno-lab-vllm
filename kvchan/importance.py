import dataclasses
from typing import Dict, List, Tuple

import torch


@dataclasses.dataclass
class TopKSelection:
    idx_k: torch.Tensor
    idx_v: torch.Tensor
    retention_k: float
    retention_v: float


@dataclasses.dataclass
class StabilityState:
    counter: int = 0
    stable: bool = False
    last_idx_k: torch.Tensor | None = None
    last_idx_v: torch.Tensor | None = None


class EMAImportance:
    """
    Tracks per-dimension importance using an exponential moving average.
    The tracker is bucketed by layer (shared across heads by default).
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        beta: float = 0.98,
        device: str | torch.device = "cpu",
        bucket_per_layer: bool = True,
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.beta = beta
        self.device = device
        self.bucket_per_layer = bucket_per_layer

        self.imp_k = torch.zeros(
            num_layers, num_heads, head_dim, device=device, dtype=torch.float32
        )
        self.imp_v = torch.zeros_like(self.imp_k)
        # shared stability state per bucket id
        num_buckets = num_layers if bucket_per_layer else 1
        self.stability: List[StabilityState] = [
            StabilityState() for _ in range(num_buckets)
        ]

    def bucket_id(self, layer_id: int) -> int:
        return layer_id if self.bucket_per_layer else 0

    @torch.no_grad()
    def update(
        self, layer: int, head_metric_k: torch.Tensor, head_metric_v: torch.Tensor
    ):
        """
        Update EMA importance scores for a single token.
        head_metric_* shape: [num_heads, head_dim]
        """
        self.imp_k[layer] = (
            self.beta * self.imp_k[layer] + (1 - self.beta) * head_metric_k
        )
        self.imp_v[layer] = (
            self.beta * self.imp_v[layer] + (1 - self.beta) * head_metric_v
        )

    def _top_indices(self, tensor: torch.Tensor, r: int) -> torch.Tensor:
        if r >= tensor.numel():
            return torch.arange(tensor.numel(), device=tensor.device)
        values, idx = torch.topk(tensor, k=r, dim=-1)
        return idx

    def _flatten_and_select(self, layer_tensor: torch.Tensor, r: int) -> torch.Tensor:
        # layer_tensor shape: [num_heads, head_dim]; flatten heads together
        flat = layer_tensor.reshape(-1, layer_tensor.shape[-1])
        # average across heads to get a single importance per dimension
        mean_imp = flat.mean(0)
        return self._top_indices(mean_imp, r)

    def compute_topk(
        self, r_k: int, r_v: int
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        idx_k: List[torch.Tensor] = []
        idx_v: List[torch.Tensor] = []
        for layer in range(self.num_layers):
            idx_k.append(self._flatten_and_select(self.imp_k[layer], r_k))
            idx_v.append(self._flatten_and_select(self.imp_v[layer], r_v))
        return idx_k, idx_v

    def update_stability(
        self,
        idx_k: List[torch.Tensor],
        idx_v: List[torch.Tensor],
        r_k: int,
        r_v: int,
        retention_threshold: float = 0.9,
        n_stable: int = 20,
    ) -> List[StabilityState]:
        """
        Track stability of selections per bucket. Returns the updated stability list.
        """
        for layer in range(self.num_layers):
            bucket = self.bucket_id(layer)
            state = self.stability[bucket]
            new_k = idx_k[layer]
            new_v = idx_v[layer]
            # Compute retention
            if state.last_idx_k is None:
                retention_k = 0.0
                retention_v = 0.0
            else:
                retention_k = len(
                    set(state.last_idx_k.tolist()) & set(new_k.tolist())
                ) / max(1, r_k)
                retention_v = len(
                    set(state.last_idx_v.tolist()) & set(new_v.tolist())
                ) / max(1, r_v)
            stable_now = (
                retention_k >= retention_threshold
                and retention_v >= retention_threshold
            )
            if stable_now:
                state.counter += 1
            else:
                state.counter = 0
                state.stable = False
            if state.counter >= n_stable:
                state.stable = True
            state.last_idx_k = new_k
            state.last_idx_v = new_v
        return self.stability

    def summary(self) -> Dict[str, float]:
        return {
            "imp_k_mean": float(self.imp_k.mean().item()),
            "imp_v_mean": float(self.imp_v.mean().item()),
        }
