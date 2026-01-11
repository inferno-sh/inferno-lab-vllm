import logging

import torch

from vllm.config.load import TensorDedupConfig
from vllm.model_executor.model_loader.tensor_dedup import (
    TensorDedupRegistry,
    wrap_tensor_dedup,
)


def _iter_shared_tensors():
    base = [
        torch.arange(8, dtype=torch.float32).reshape(2, 4),
        torch.arange(8, dtype=torch.float32).reshape(2, 4) + 5,
    ]
    for idx, tensor in enumerate(base):
        yield f"model.layers.{idx}.weight", tensor.clone()


def _iter_mixed_tensors(offset: float):
    yield from _iter_shared_tensors()
    yield "head.weight", torch.arange(4, dtype=torch.float32) + offset


def test_tensor_dedup_reuses_identical_layers():
    TensorDedupRegistry.clear()
    logger = logging.getLogger("test_tensor_dedup_reuses_identical_layers")
    dedup_cfg = TensorDedupConfig(enabled=True, min_tensor_bytes=0)

    first = list(
        wrap_tensor_dedup(
            _iter_shared_tensors(),
            logger=logger,
            tensor_dedup=dedup_cfg,
        )
    )
    second = list(
        wrap_tensor_dedup(
            _iter_shared_tensors(),
            logger=logger,
            tensor_dedup=dedup_cfg,
        )
    )

    assert len(first) == len(second) == 2
    assert first[0][1].data_ptr() == second[0][1].data_ptr()
    assert first[1][1].data_ptr() == second[1][1].data_ptr()


def test_tensor_dedup_respects_min_tensor_bytes():
    TensorDedupRegistry.clear()
    logger = logging.getLogger("test_tensor_dedup_respects_min_tensor_bytes")
    dedup_cfg = TensorDedupConfig(enabled=True, min_tensor_bytes=1 << 20)

    weights_a = list(
        wrap_tensor_dedup(
            _iter_shared_tensors(),
            logger=logger,
            tensor_dedup=dedup_cfg,
        )
    )
    weights_b = list(
        wrap_tensor_dedup(
            _iter_shared_tensors(),
            logger=logger,
            tensor_dedup=dedup_cfg,
        )
    )

    assert weights_a[0][1].data_ptr() != weights_b[0][1].data_ptr()
    assert weights_a[1][1].data_ptr() != weights_b[1][1].data_ptr()


def test_tensor_dedup_reuses_shared_but_not_unique_layers():
    TensorDedupRegistry.clear()
    logger = logging.getLogger("test_tensor_dedup_reuses_shared_but_not_unique_layers")
    dedup_cfg = TensorDedupConfig(enabled=True, min_tensor_bytes=0)

    base = list(
        wrap_tensor_dedup(
            _iter_mixed_tensors(offset=0.0),
            logger=logger,
            tensor_dedup=dedup_cfg,
        )
    )
    derived = list(
        wrap_tensor_dedup(
            _iter_mixed_tensors(offset=1.0),
            logger=logger,
            tensor_dedup=dedup_cfg,
        )
    )

    # The shared tensors should reuse storage.
    assert base[0][1].data_ptr() == derived[0][1].data_ptr()
    assert base[1][1].data_ptr() == derived[1][1].data_ptr()
    # The unique tensor differs, so it should not be reused.
    assert base[2][1].data_ptr() != derived[2][1].data_ptr()

