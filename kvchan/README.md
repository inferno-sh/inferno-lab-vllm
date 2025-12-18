# kvchan: quality-first dynamic KV channel compression

This prototype implements a channel-selection KV cache with a full-fidelity sliding window and a conservative control loop. It is written to be easy to inspect and debug in HF eager first, with scaffolding for vLLM integration.

## Design
- **Priority:** fidelity first. The system always starts in full mode, gathers importance statistics online, and only enables packing when selection stability is high.
- **Importance:** exponential moving average of per-dimension magnitudes for keys and values. Values use the last-step value vectors as a proxy for attention outputs. Keys use the same proxy; K compression can be disabled.
- **Selection & stability:** every `T_update` tokens compute top-`rK`/`rV` sets per layer. Retention ≥0.9 for `N_stable` consecutive updates marks the set as stable, which triggers compression.
- **Window W:** a ring buffer of full-dimension K/V for the most recent W tokens is always kept. Tokens older than W are stored in packed form.
- **Packed storage:** `PackedKVCache` stores only the selected dimensions (for K and optionally V). Persistent allocation shrinks to `rK`/`rV` per head; missing dims are reconstructed on the fly (slow scatter path) when feeding the model. This is where VRAM is saved versus masking-only approaches.
- **Fidelity checks:** optional periodic recomputation using the windowed baseline (full window + compressed tail). If metrics fall below thresholds, the system immediately backs off to full mode and resets stability counters.
- **Backoff:** on any fidelity failure, caches are reconstructed to full tensors and dynamic probing restarts.

## Layouts
- `FullKVCache`: stores per-layer lists of full [num_heads, head_dim] tensors.
- `PackedKVCache`: keeps a deque for the full window and packed tensors for older tokens. Packing gathers selected dims into tight buffers; scatter rebuilds the full shape only for compute. `memory_bytes()` reports allocated bytes to confirm savings.

## CLI
```
# Baseline full mode (HF eager)
python -m kvchan.cli run --backend hf --mode full --max_tokens 64

# Dynamic compression with window W (HF)
python -m kvchan.cli run --backend hf --mode dynamic --W 128 --rK 32 --rV 128 --check_every 16

# Baseline full mode (vLLM)
python -m kvchan.cli run --backend vllm --mode full --max_tokens 64

# Dynamic compression with masking (vLLM)
python -m kvchan.cli run --backend vllm --mode dynamic --rK 32 --rV 128

# Probe-only importance collection
python -m kvchan.cli train-probe --backend hf
```
Results are written as JSONL under `kvchan/outputs/`.

## Backends
- **HF backend (`hf_backend.py`)**: correctness-first, stepwise decoding with custom KV stores and stability gating. Uses a scatter slow path when compressed to keep behavior correct. This path is the reference for debugging.
- **vLLM backend (`vllm_backend.py`)**: independent implementation using vLLM's native engine with channel masking support. Uses attention-layer masks to zero out unselected dimensions and compares baseline vs masked outputs for fidelity. The intended hook points inside vLLM are the KV cache allocator and attention backends; implement `PackedKVCache` as a KVStore and scatter packed K/V into temporary full tensors for attention reads. The packed buffers should be allocated with last-dim = `rK`/`rV` to reduce persistent KV allocation.

## How VRAM is saved
Packed storage reduces the physical last dimension of the KV buffers (rK/rV instead of d_head) for all tokens outside the uncompressed window. The sliding window guarantees fidelity for recent context, while older tokens consume less memory. `memory_bytes()` in both caches reports the total allocated bytes so runs can log savings.

## Fidelity guardrails
`fidelity.py` computes top-k overlap, margin drift, and KL between the current logits and a windowed baseline. Failures trigger immediate backoff to full mode. Thresholds are conservative by default to favor quality over compression.

## Prompt suite
`prompts.py` contains the fixed 3-prompt suite: merge sort, BST API, and a Sudoku grid. These are used by both `run` and `train-probe` commands for repeatability.
