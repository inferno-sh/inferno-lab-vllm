import dataclasses
from enum import Enum, auto
from typing import Dict, List

import torch


class Mode(Enum):
    FULL = auto()
    WARMUP = auto()
    COMPRESSED = auto()


@dataclasses.dataclass
class CompressionConfig:
    r_k: int
    r_v: int
    window: int
    disable_k_compress: bool = False
    disable_v_compress: bool = False
    update_interval: int = 32
    fidelity_every: int = 64
    retention_threshold: float = 0.9
    n_stable: int = 20
    bucket_per_layer: bool = True
    stable_mask_enable: bool = True
    stable_mask_update_interval: int = 64
    stable_mask_overlap_threshold: float = 0.85


@dataclasses.dataclass
class SequenceState:
    mode: Mode = Mode.FULL
    last_switch_step: int = 0
    stability: List[int] | None = None
    idx_k: List[torch.Tensor] | None = None
    idx_v: List[torch.Tensor] | None = None
    backoff_events: int = 0


class ControlLoop:
    """
    Light-weight control state machine managing mode transitions and backoff.
    """

    def __init__(self, cfg: CompressionConfig, num_layers: int):
        self.cfg = cfg
        self.state = SequenceState()
        self.num_layers = num_layers

    def should_update_selection(self, step: int) -> bool:
        return step % self.cfg.update_interval == 0 and step > 0

    def should_check_fidelity(self, step: int) -> bool:
        return step % self.cfg.fidelity_every == 0 and step > 0

    def maybe_switch_to_compressed(self, stability_states) -> bool:
        stable = all(s.stable for s in stability_states)
        if stable and self.state.mode != Mode.COMPRESSED:
            self.state.mode = Mode.COMPRESSED
            self.state.last_switch_step = 0
            return True
        return False

    def reset_to_full(self):
        self.state.mode = Mode.FULL
        self.state.idx_k = None
        self.state.idx_v = None
        self.state.stability = None
        self.state.backoff_events += 1

    def attach_indices(self, idx_k, idx_v):
        self.state.idx_k = idx_k
        self.state.idx_v = idx_v

    def as_dict(self) -> Dict:
        return {
            "mode": self.state.mode.name,
            "backoff_events": self.state.backoff_events,
        }
