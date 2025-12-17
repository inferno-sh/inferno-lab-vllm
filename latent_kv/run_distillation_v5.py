#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Latent KV Distillation v5 - Staged Compression

Key changes from v4:
1. STAGED COMPRESSION: Train at moderate ranks, then anneal to target ranks
2. Rank masking: Parameters sized to max, effective ranks controlled via masking
3. Fixed loss weights (NO schedule) - CE=0.6, KL=0.4, UL=1.0
4. Digit-run penalty in unlikelihood loss
5. JSONL logging for eval metrics
6. Hard early stopping on repetition/entropy

Rank Schedule:
- Stage A (0-70%): Train at moderate ranks (stable)
- Stage B (70-100%): Linear anneal to aggressive target ranks

Hardware: RTX 3090 (24GB), optimized for FP32 stability
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from collections import deque
from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from latent_kv.convert import convert_qwen2_to_latent_kv
from latent_kv.config import get_default_config, StagedCompressionConfig, LayerLatentConfig, LatentKVConfig
from latent_kv.attention import LatentKVAttention


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_OUTPUT_DIR = Path(os.environ.get("HOME", "/tmp")) / "temp" / "latent_kv_distill_v5"
DEFAULT_DATASET = "togethercomputer/RedPajama-Data-1T-Sample"
DEFAULT_TARGET_TOKENS = 50_000_000

# Staged compression defaults
DEFAULT_RK_MAX = 32      # Max K rank (parameter size)
DEFAULT_RV_MAX = 64      # Max V rank (parameter size)
DEFAULT_RK_START = 32    # Starting effective K rank (Stage A)
DEFAULT_RV_START = 64    # Starting effective V rank (Stage A)
DEFAULT_RK_TARGET = 16   # Target K rank after annealing (Stage B end)
DEFAULT_RV_TARGET = 32   # Target V rank after annealing (Stage B end)
DEFAULT_ANNEAL_START = 0.7  # Fraction when annealing starts

# Context length curriculum
CONTEXT_LEN_PHASE1 = 512   # Steps 0-70%
CONTEXT_LEN_PHASE2 = 1024  # Steps 70-100%
CURRICULUM_TRANSITION = 0.7

# Batch size
BATCH_SIZE = 1
GRAD_ACCUM = 64

# Training
TOTAL_STEPS = 8000
EVAL_EVERY = 250
SAVE_EVERY = 1000
LOG_EVERY = 25
WARMUP_RATIO = 0.05
LEARNING_RATE = 3e-4
LR_FLOOR = 1e-4
GRAD_CLIP = 1.0
EMA_DECAY = 0.999

# =============================================================================
# LOSS WEIGHTS - FIXED (NO SCHEDULE!)
# =============================================================================
CE_WEIGHT = 0.6
KL_WEIGHT = 0.4
UNLIKELIHOOD_WEIGHT = 1.0
ATTN_OUT_WEIGHT = 0.5
HIDDEN_WEIGHT = 0.1

# Loss hyperparameters
LABEL_SMOOTHING = 0.1
TEMPERATURE = 3.0
TOP_K_KL = 32

# Unlikelihood settings
NGRAM_SIZE = 3
MIN_PREFIX_FOR_UL = 32
DIGIT_RUN_THRESHOLD = 6  # Penalize if last N tokens are all digits

# Early stopping
MAX_REPETITION_RATE = 0.05
MAX_ENTROPY_DROP = 0.25
ENTROPY_BASELINE_WINDOW = 10


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Latent KV Distillation v5 - Staged Compression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default configuration (moderate -> aggressive compression)
  python run_distillation_v5.py

  # Custom rank schedule
  python run_distillation_v5.py --rk-max 32 --rv-max 64 \\
      --rk-start 32 --rv-start 64 \\
      --rk-target 12 --rv-target 24

  # More aggressive compression
  python run_distillation_v5.py --rk-target 8 --rv-target 16
        """
    )

    # Model and output
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="HuggingFace model name")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                        help="HuggingFace dataset name")
    parser.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET_TOKENS,
                        help="Target number of tokens to train on")

    # Staged compression ranks
    parser.add_argument("--rk-max", type=int, default=DEFAULT_RK_MAX,
                        help="Maximum K rank (parameter size)")
    parser.add_argument("--rv-max", type=int, default=DEFAULT_RV_MAX,
                        help="Maximum V rank (parameter size)")
    parser.add_argument("--rk-start", type=int, default=DEFAULT_RK_START,
                        help="Starting effective K rank (Stage A)")
    parser.add_argument("--rv-start", type=int, default=DEFAULT_RV_START,
                        help="Starting effective V rank (Stage A)")
    parser.add_argument("--rk-target", type=int, default=DEFAULT_RK_TARGET,
                        help="Target K rank after annealing (Stage B end)")
    parser.add_argument("--rv-target", type=int, default=DEFAULT_RV_TARGET,
                        help="Target V rank after annealing (Stage B end)")
    parser.add_argument("--rank-anneal-start-frac", type=float, default=DEFAULT_ANNEAL_START,
                        help="Fraction of training when rank annealing starts")

    # Training
    parser.add_argument("--total-steps", type=int, default=TOTAL_STEPS,
                        help="Total training steps")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help="Learning rate")
    parser.add_argument("--grad-accum", type=int, default=GRAD_ACCUM,
                        help="Gradient accumulation steps")

    return parser.parse_args()


class StreamingTextDataset(IterableDataset):
    """Streaming dataset from HuggingFace datasets."""

    def __init__(self, tokenizer, max_length, target_tokens, dataset_name):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_tokens = target_tokens
        self.dataset_name = dataset_name
        self.tokens_seen = 0

    def __iter__(self):
        try:
            dataset = load_dataset(self.dataset_name, split="train", streaming=True)
        except Exception as e:
            print(f"Failed to load {self.dataset_name}, falling back to C4")
            dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)

        buffer = []

        for item in dataset:
            text = item.get("text", item.get("content", ""))
            if not text or len(text.strip()) < 100:
                continue

            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)

            while len(buffer) >= self.max_length:
                sequence = buffer[:self.max_length]
                buffer = buffer[self.max_length:]

                self.tokens_seen += len(sequence)

                yield {
                    "input_ids": torch.tensor(sequence, dtype=torch.long),
                    "attention_mask": torch.ones(len(sequence), dtype=torch.long),
                }

                if self.tokens_seen >= self.target_tokens:
                    return


class EMAModel:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


class AttentionOutputCapture:
    """Hook to capture attention outputs."""

    def __init__(self, model):
        self.outputs = []
        self.hooks = []
        self._register_hooks(model)

    def _register_hooks(self, model):
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "layers"):
            layers = model.layers
        else:
            return

        for layer in layers:
            if hasattr(layer, "self_attn"):
                hook = layer.self_attn.register_forward_hook(self._hook_fn)
                self.hooks.append(hook)

    def _hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            self.outputs.append(output[0].detach())
        else:
            self.outputs.append(output.detach())

    def clear(self):
        self.outputs = []

    def get_outputs(self):
        outputs = self.outputs
        self.outputs = []
        return outputs

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()


def get_digit_tokens(tokenizer):
    """Get token IDs that represent digits 0-9."""
    digit_tokens = set()
    for digit in "0123456789":
        # Get all tokens that are just this digit
        tokens = tokenizer.encode(digit, add_special_tokens=False)
        digit_tokens.update(tokens)
    return digit_tokens


def compute_ngram_repetition_mask(input_ids, n=3, min_prefix=32):
    """Find positions where tokens create repeated n-grams."""
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    repetition_mask = torch.zeros(batch_size, seq_len, device=device)

    for b in range(batch_size):
        seen_ngrams = set()
        for i in range(seq_len - n + 1):
            if i < min_prefix:
                ngram = tuple(input_ids[b, i:i+n].tolist())
                seen_ngrams.add(ngram)
                continue

            ngram = tuple(input_ids[b, i:i+n].tolist())
            if ngram in seen_ngrams:
                repetition_mask[b, i:i+n] = 1.0
            seen_ngrams.add(ngram)

    return repetition_mask


def compute_digit_run_mask(input_ids, digit_tokens, run_threshold=6):
    """
    Find positions after a run of digit tokens.

    If the last N tokens (N >= run_threshold) are all digits,
    penalize the next token if it's also a digit.
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    digit_run_mask = torch.zeros(batch_size, seq_len, device=device)
    digit_set = digit_tokens

    for b in range(batch_size):
        run_length = 0
        for i in range(seq_len):
            token = input_ids[b, i].item()
            if token in digit_set:
                run_length += 1
                if run_length >= run_threshold:
                    # Mark this position for penalty if next is digit
                    if i + 1 < seq_len and input_ids[b, i + 1].item() in digit_set:
                        digit_run_mask[b, i + 1] = 1.0
            else:
                run_length = 0

    return digit_run_mask


def compute_unlikelihood_loss(logits, input_ids, repetition_mask, digit_run_mask=None, digit_tokens=None):
    """
    Unlikelihood loss: penalize probability of repeated tokens and digit runs.
    """
    batch_size, seq_len, vocab_size = logits.shape
    device = logits.device

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_rep_mask = repetition_mask[:, 1:]

    # Combine masks
    combined_mask = shift_rep_mask.clone()
    if digit_run_mask is not None:
        shift_digit_mask = digit_run_mask[:, 1:]
        combined_mask = torch.max(combined_mask, shift_digit_mask)

    probs = F.softmax(shift_logits, dim=-1)
    next_token_probs = probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    unlikelihood = -torch.log(torch.clamp(1 - next_token_probs, min=1e-8))
    masked_loss = unlikelihood * combined_mask

    if combined_mask.sum() > 0:
        return masked_loss.sum() / (combined_mask.sum() + 1e-8)
    else:
        return torch.tensor(0.0, device=device)


def compute_entropy(logits):
    """Compute average entropy of predictions."""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy.mean().item()


def compute_repetition_score(text, n=3):
    """Compute fraction of repeated n-grams in generated text."""
    tokens = text.split()
    if len(tokens) < n:
        return 0.0

    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0

    unique = set(ngrams)
    return 1 - len(unique) / len(ngrams)


def compute_loss(student_out, teacher_out, input_ids, attention_mask,
                 attn_outputs_student=None, attn_outputs_teacher=None,
                 digit_tokens=None):
    """V5 loss function with FIXED weights (no schedule)."""
    device = student_out.logits.device
    batch_size, seq_len, vocab_size = student_out.logits.shape

    student_logits = student_out.logits.float()
    teacher_logits = teacher_out.logits.float()

    student_logits = student_logits.clamp(-100, 100)
    teacher_logits = teacher_logits.clamp(-100, 100)

    shift_student = student_logits[:, :-1, :].contiguous()
    shift_teacher = teacher_logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous().float()

    # 1. Cross-Entropy Loss
    ce_loss = F.cross_entropy(
        shift_student.view(-1, vocab_size),
        shift_labels.view(-1),
        reduction='none',
        label_smoothing=LABEL_SMOOTHING
    ).view(batch_size, seq_len - 1)
    ce_loss = (ce_loss * shift_mask).sum() / (shift_mask.sum() + 1e-8)

    # 2. Top-K KL Divergence
    with torch.no_grad():
        teacher_probs = F.softmax(shift_teacher / TEMPERATURE, dim=-1)
        top_k_probs, top_k_indices = teacher_probs.topk(TOP_K_KL, dim=-1)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)

    student_log_probs = F.log_softmax(shift_student / TEMPERATURE, dim=-1)
    student_top_k = student_log_probs.gather(2, top_k_indices)

    kl_per_token = -(top_k_probs * student_top_k).sum(dim=-1).clamp(max=100.0)
    kl_loss = (kl_per_token * shift_mask).sum() / (shift_mask.sum() + 1e-8)
    kl_loss = kl_loss * (TEMPERATURE ** 2)

    # 3. Unlikelihood Loss (with digit-run penalty)
    rep_mask = compute_ngram_repetition_mask(input_ids, n=NGRAM_SIZE, min_prefix=MIN_PREFIX_FOR_UL)
    digit_run_mask = None
    if digit_tokens:
        digit_run_mask = compute_digit_run_mask(input_ids, digit_tokens, DIGIT_RUN_THRESHOLD)
    ul_loss = compute_unlikelihood_loss(student_logits, input_ids, rep_mask, digit_run_mask, digit_tokens)

    # 4. Attention Output Matching
    attn_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    if attn_outputs_student and attn_outputs_teacher:
        num_layers = len(attn_outputs_student)
        match_layers = [num_layers // 4, num_layers // 2, 3 * num_layers // 4]
        attn_mask_float = attention_mask.float().unsqueeze(-1)

        count = 0
        for idx in match_layers:
            if idx < len(attn_outputs_student) and idx < len(attn_outputs_teacher):
                s_attn = attn_outputs_student[idx].float()
                t_attn = attn_outputs_teacher[idx].float().detach()

                if not torch.isnan(s_attn).any():
                    s_masked = s_attn * attn_mask_float
                    t_masked = t_attn * attn_mask_float
                    mse = ((s_masked - t_masked).pow(2).sum()) / (attn_mask_float.sum() * s_attn.shape[-1] + 1e-8)
                    attn_loss += mse
                    count += 1

        if count > 0:
            attn_loss = attn_loss / count

    # 5. Hidden State MSE
    hidden_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    if student_out.hidden_states and teacher_out.hidden_states:
        num_hidden = len(student_out.hidden_states)
        stride = max(1, num_hidden // 4)
        count = 0
        for i in range(0, num_hidden, stride):
            if i < len(teacher_out.hidden_states):
                s_h = student_out.hidden_states[i].float()
                t_h = teacher_out.hidden_states[i].float().detach()

                if not torch.isnan(s_h).any():
                    s_norm = s_h / (s_h.norm(dim=-1, keepdim=True) + 1e-8)
                    t_norm = t_h / (t_h.norm(dim=-1, keepdim=True) + 1e-8)
                    hidden_loss += F.mse_loss(s_norm, t_norm)
                    count += 1
        if count > 0:
            hidden_loss = hidden_loss / count

    # Total Loss (FIXED WEIGHTS)
    total_loss = (
        CE_WEIGHT * ce_loss +
        KL_WEIGHT * kl_loss +
        UNLIKELIHOOD_WEIGHT * ul_loss +
        ATTN_OUT_WEIGHT * attn_loss +
        HIDDEN_WEIGHT * hidden_loss
    )

    entropy = compute_entropy(shift_student)

    if torch.isnan(total_loss):
        print(f"WARNING: NaN loss! CE={ce_loss.item():.4f}, KL={kl_loss.item():.4f}")
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

    return {
        "total": total_loss,
        "ce": ce_loss,
        "kl": kl_loss,
        "ul": ul_loss,
        "attn": attn_loss,
        "hidden": hidden_loss,
        "entropy": entropy,
    }


def evaluate_generation(model, tokenizer, prompts, device):
    """Evaluate generation quality and compute repetition rate."""
    model.eval()
    results = []
    total_rep = 0
    total_entropy = 0

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )

            logits = model(output).logits
            entropy = compute_entropy(logits[:, -30:, :])

        text = tokenizer.decode(output[0], skip_special_tokens=True)
        rep_score = compute_repetition_score(text)

        results.append({
            "prompt": prompt,
            "text": text,
            "repetition": rep_score,
            "entropy": entropy,
        })

        total_rep += rep_score
        total_entropy += entropy

    model.train()

    avg_rep = total_rep / len(prompts) if prompts else 0
    avg_entropy = total_entropy / len(prompts) if prompts else 0

    return results, avg_rep, avg_entropy


def set_model_effective_ranks(model, r_k_eff: int, r_v_eff: int):
    """Set effective ranks for all LatentKVAttention layers in the model."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        return 0

    count = 0
    for layer in layers:
        if hasattr(layer, "self_attn"):
            attn = layer.self_attn
            if isinstance(attn, LatentKVAttention):
                # Clamp to layer's max
                actual_r_k = min(r_k_eff, attn.r_k_max)
                actual_r_v = min(r_v_eff, attn.r_v_max)
                attn.set_effective_ranks(actual_r_k, actual_r_v)
                count += 1

    return count


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create staged compression config
    staged_config = StagedCompressionConfig(
        r_k_max=args.rk_max,
        r_v_max=args.rv_max,
        r_k_start=args.rk_start,
        r_v_start=args.rv_start,
        r_k_target=args.rk_target,
        r_v_target=args.rv_target,
        anneal_start_frac=args.rank_anneal_start_frac,
    )

    print("=" * 70)
    print("Latent KV Distillation v5 - Staged Compression")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Output: {output_dir}")
    print(f"Dataset: {args.dataset}")
    print(f"Target tokens: {args.target_tokens:,}")
    print()
    print("Staged Compression Configuration:")
    print(f"  Max ranks: r_k={args.rk_max}, r_v={args.rv_max}")
    print(f"  Start ranks: r_k={args.rk_start}, r_v={args.rv_start}")
    print(f"  Target ranks: r_k={args.rk_target}, r_v={args.rv_target}")
    print(f"  Anneal start: {args.rank_anneal_start_frac:.0%} of training")
    print()
    print("Loss weights (FIXED - no schedule):")
    print(f"  CE={CE_WEIGHT}, KL={KL_WEIGHT}, UL={UNLIKELIHOOD_WEIGHT}")
    print(f"  Attn={ATTN_OUT_WEIGHT}, Hidden={HIDDEN_WEIGHT}")
    print()
    print(f"Early stopping: rep>{MAX_REPETITION_RATE:.0%} OR entropy_drop>{MAX_ENTROPY_DROP:.0%}")
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get digit tokens for digit-run penalty
    digit_tokens = get_digit_tokens(tokenizer)
    print(f"  Digit tokens identified: {len(digit_tokens)}")

    # Load teacher
    print(f"Loading teacher model...")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
    ).to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    print(f"  Teacher parameters: {sum(p.numel() for p in teacher.parameters()):,}")

    # Create student with MAX ranks (for staged compression)
    print(f"\nCreating latent KV student model with max ranks...")

    # Get model config
    hf_config = teacher.config
    d_model = hf_config.hidden_size
    n_heads = hf_config.num_attention_heads
    n_kv_heads = getattr(hf_config, "num_key_value_heads", n_heads)
    d_head = d_model // n_heads
    num_layers = hf_config.num_hidden_layers

    # Create layer configs with MAX ranks
    layer_configs = []
    for layer_idx in range(num_layers):
        layer_configs.append(LayerLatentConfig(
            r_k=args.rk_max,
            r_v=args.rv_max,
            use_k_anchor=False,
        ))

    latent_config = LatentKVConfig(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        d_head=d_head,
        num_layers=num_layers,
        layer_configs=layer_configs,
        rope_theta=getattr(hf_config, "rope_theta", 10000.0),
        rope_scaling=getattr(hf_config, "rope_scaling", None),
    )
    latent_config.print_summary()

    student = convert_qwen2_to_latent_kv(
        teacher,
        config=latent_config,
        init_method="svd",
        copy_model=True,
    )
    student = student.float()

    # Set initial effective ranks
    r_k_eff, r_v_eff = staged_config.get_effective_ranks(0, args.total_steps)
    num_updated = set_model_effective_ranks(student, r_k_eff, r_v_eff)
    print(f"\nInitial effective ranks: r_k={r_k_eff}, r_v={r_v_eff} (updated {num_updated} layers)")

    # Freeze non-bottleneck params
    latent_kv_params = []
    frozen_count = 0
    trainable_count = 0

    for name, param in student.named_parameters():
        if any(x in name for x in ['k_down', 'k_up', 'v_down', 'v_up', 'k_anchor']):
            param.requires_grad = True
            latent_kv_params.append(param)
            trainable_count += param.numel()
        else:
            param.requires_grad = False
            frozen_count += param.numel()

    print(f"  Frozen: {frozen_count:,}, Trainable: {trainable_count:,}")

    # Test prompts
    test_prompts = [
        "The capital of France is",
        "In the year 2024, artificial intelligence",
        "The meaning of life is",
        "Once upon a time in a small village,",
        "The best way to learn programming is",
    ]

    # Initial generation test
    print("\n" + "=" * 70)
    print("Pre-training Generation Test")
    print("=" * 70)

    print("\nTeacher outputs:")
    for prompt in test_prompts[:3]:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = teacher.generate(**inputs, max_new_tokens=30, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        print(f"  {prompt!r} -> {tokenizer.decode(out[0], skip_special_tokens=True)}")

    # Setup training
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)

    optimizer = torch.optim.AdamW(latent_kv_params, lr=args.lr, weight_decay=0.01)

    warmup_steps = int(args.total_steps * WARMUP_RATIO)
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, args.total_steps - warmup_steps)
        decay = 0.5 * (1 + math.cos(math.pi * progress))
        return max(LR_FLOOR / args.lr, decay)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Setup hooks and EMA
    teacher_attn_capture = AttentionOutputCapture(teacher)
    student_attn_capture = AttentionOutputCapture(student)
    ema = EMAModel(student, decay=EMA_DECAY)

    print(f"  Teacher hooks: {len(teacher_attn_capture.hooks)}")
    print(f"  Student hooks: {len(student_attn_capture.hooks)}")

    # Training state
    student.train()
    global_step = 0
    running_loss = 0.0
    running_ce = 0.0
    running_kl = 0.0
    running_ul = 0.0
    running_entropy = 0.0
    start_time = time.time()

    # Early stopping state
    entropy_baseline = None
    entropy_history = deque(maxlen=ENTROPY_BASELINE_WINDOW)
    best_checkpoint_step = 0
    best_rep_score = float('inf')
    early_stopped = False
    stop_reason = None

    # Log files
    log_file = output_dir / "training_log.txt"
    jsonl_file = output_dir / "eval_metrics.jsonl"

    with open(log_file, "w") as f:
        f.write(f"Training started at {datetime.now()}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Staged Compression: max({args.rk_max},{args.rv_max}) -> start({args.rk_start},{args.rv_start}) -> target({args.rk_target},{args.rv_target})\n")
        f.write(f"v5 Configuration: Fixed weights CE={CE_WEIGHT}, KL={KL_WEIGHT}, UL={UNLIKELIHOOD_WEIGHT}\n")
        f.write(f"Early stopping: rep>{MAX_REPETITION_RATE:.0%}, entropy_drop>{MAX_ENTROPY_DROP:.0%}\n\n")

    # Clear JSONL file
    open(jsonl_file, "w").close()

    # Context length helper
    def get_context_length(step):
        if step < args.total_steps * CURRICULUM_TRANSITION:
            return CONTEXT_LEN_PHASE1
        return CONTEXT_LEN_PHASE2

    # Create initial dataloader
    current_ctx_len = get_context_length(0)
    dataset = StreamingTextDataset(tokenizer, current_ctx_len, args.target_tokens, args.dataset)
    dataloader = iter(DataLoader(dataset, batch_size=BATCH_SIZE))

    print(f"\nStarting with context length: {current_ctx_len}")
    print(f"Stage: {staged_config.get_stage(0, args.total_steps)}")

    while global_step < args.total_steps and not early_stopped:
        # Update effective ranks based on schedule
        r_k_eff, r_v_eff = staged_config.get_effective_ranks(global_step, args.total_steps)
        set_model_effective_ranks(student, r_k_eff, r_v_eff)

        # Check if we need to update context length
        new_ctx_len = get_context_length(global_step)
        if new_ctx_len != current_ctx_len:
            print(f"\n*** Context length curriculum: {current_ctx_len} -> {new_ctx_len} ***")
            print(f"*** Stage: {staged_config.get_stage(global_step, args.total_steps)} ***")
            print(f"*** Effective ranks: r_k={r_k_eff}, r_v={r_v_eff} ***\n")
            current_ctx_len = new_ctx_len
            dataset = StreamingTextDataset(tokenizer, current_ctx_len, args.target_tokens, args.dataset)
            dataloader = iter(DataLoader(dataset, batch_size=BATCH_SIZE))

        # Get batch
        try:
            batch = next(dataloader)
        except StopIteration:
            dataset = StreamingTextDataset(tokenizer, current_ctx_len, args.target_tokens, args.dataset)
            dataloader = iter(DataLoader(dataset, batch_size=BATCH_SIZE))
            batch = next(dataloader)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Clear captures
        teacher_attn_capture.clear()
        student_attn_capture.clear()

        # Teacher forward
        with torch.no_grad():
            teacher_out = teacher(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            teacher_attn_outputs = teacher_attn_capture.get_outputs()

        # Student forward
        student_out = student(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        student_attn_outputs = student_attn_capture.get_outputs()

        # Compute loss
        losses = compute_loss(
            student_out, teacher_out, input_ids, attention_mask,
            attn_outputs_student=student_attn_outputs,
            attn_outputs_teacher=teacher_attn_outputs,
            digit_tokens=digit_tokens,
        )

        loss = losses["total"] / args.grad_accum
        loss.backward()

        running_loss += losses["total"].item()
        running_ce += losses["ce"].item()
        running_kl += losses["kl"].item()
        running_ul += losses["ul"].item()
        running_entropy += losses["entropy"]

        # Gradient accumulation step
        if (global_step + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(latent_kv_params, GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            ema.update(student)

        global_step += 1

        # Logging
        if global_step % LOG_EVERY == 0:
            avg_loss = running_loss / LOG_EVERY
            avg_ce = running_ce / LOG_EVERY
            avg_kl = running_kl / LOG_EVERY
            avg_ul = running_ul / LOG_EVERY
            avg_ent = running_entropy / LOG_EVERY
            lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - start_time
            stage = staged_config.get_stage(global_step, args.total_steps)

            log_msg = (
                f"Step {global_step:5d}/{args.total_steps} | "
                f"Loss: {avg_loss:.3f} | "
                f"CE: {avg_ce:.3f} | "
                f"KL: {avg_kl:.3f} | "
                f"UL: {avg_ul:.3f} | "
                f"Ent: {avg_ent:.2f} | "
                f"r_k={r_k_eff} r_v={r_v_eff} | "
                f"LR: {lr:.2e} | "
                f"{stage}"
            )
            print(log_msg)

            with open(log_file, "a") as f:
                f.write(log_msg + "\n")

            running_loss = 0.0
            running_ce = 0.0
            running_kl = 0.0
            running_ul = 0.0
            running_entropy = 0.0

        # Evaluation and early stopping check
        if global_step % EVAL_EVERY == 0:
            print("\n--- Evaluation ---")
            stage = staged_config.get_stage(global_step, args.total_steps)
            print(f"  Stage: {stage}, Effective ranks: r_k={r_k_eff}, r_v={r_v_eff}")

            ema.apply_shadow(student)
            gen_results, avg_rep, avg_entropy = evaluate_generation(
                student, tokenizer, test_prompts, device
            )
            ema.restore(student)

            # Update entropy baseline
            entropy_history.append(avg_entropy)
            if entropy_baseline is None and len(entropy_history) >= ENTROPY_BASELINE_WINDOW:
                entropy_baseline = sum(entropy_history) / len(entropy_history)
                print(f"  Entropy baseline established: {entropy_baseline:.2f}")

            # Print samples
            for r in gen_results[:3]:
                print(f"  {r['prompt']!r}")
                print(f"    -> {r['text'][:100]}...")
                print(f"    Rep: {r['repetition']:.1%}, Ent: {r['entropy']:.2f}")

            print(f"\n  Avg Repetition: {avg_rep:.1%}")
            print(f"  Avg Entropy: {avg_entropy:.2f}")

            # Log to JSONL
            eval_record = {
                "step": global_step,
                "timestamp": datetime.now().isoformat(),
                "stage": stage,
                "r_k_eff": r_k_eff,
                "r_v_eff": r_v_eff,
                "avg_repetition": avg_rep,
                "avg_entropy": avg_entropy,
                "entropy_baseline": entropy_baseline,
                "samples": [
                    {"prompt": r["prompt"], "text": r["text"][:200], "rep": r["repetition"], "ent": r["entropy"]}
                    for r in gen_results
                ]
            }
            with open(jsonl_file, "a") as f:
                f.write(json.dumps(eval_record) + "\n")

            # Track best checkpoint
            if avg_rep < best_rep_score:
                best_rep_score = avg_rep
                best_checkpoint_step = global_step
                print(f"  ** NEW BEST (rep={best_rep_score:.1%}) **")

            # Early stopping checks
            if avg_rep > MAX_REPETITION_RATE:
                early_stopped = True
                stop_reason = f"Repetition rate {avg_rep:.1%} > {MAX_REPETITION_RATE:.0%}"
                print(f"\n*** EARLY STOPPING: {stop_reason} ***")

            if entropy_baseline is not None:
                entropy_drop = (entropy_baseline - avg_entropy) / entropy_baseline
                if entropy_drop > MAX_ENTROPY_DROP:
                    early_stopped = True
                    stop_reason = f"Entropy drop {entropy_drop:.1%} > {MAX_ENTROPY_DROP:.0%}"
                    print(f"\n*** EARLY STOPPING: {stop_reason} ***")

            print()

            with open(log_file, "a") as f:
                f.write(f"\n--- Eval at step {global_step} ---\n")
                f.write(f"Stage: {stage}, Effective ranks: r_k={r_k_eff}, r_v={r_v_eff}\n")
                f.write(f"Avg Repetition: {avg_rep:.1%}\n")
                f.write(f"Avg Entropy: {avg_entropy:.2f}\n")
                for r in gen_results:
                    f.write(f"  {r['prompt']!r} -> {r['text'][:80]}... (rep={r['repetition']:.1%})\n")
                f.write("\n")

        # Save checkpoint
        if global_step % SAVE_EVERY == 0 or early_stopped:
            checkpoint_dir = output_dir / f"checkpoint-{global_step}"
            checkpoint_dir.mkdir(exist_ok=True)
            student.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)

            # Save staged config
            with open(checkpoint_dir / "staged_config.json", "w") as f:
                json.dump(asdict(staged_config), f, indent=2)

            print(f"  Saved checkpoint to {checkpoint_dir}")

    # Final summary
    print("\n" + "=" * 70)
    print("Training Complete")
    print("=" * 70)

    if early_stopped:
        print(f"Early stopped at step {global_step}: {stop_reason}")
    else:
        print(f"Completed all {args.total_steps} steps")

    print(f"\nBest checkpoint: step {best_checkpoint_step} (rep={best_rep_score:.1%})")

    # Final generation test
    print("\nFinal generations (EMA):")
    ema.apply_shadow(student)
    for prompt in test_prompts[:3]:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = student.generate(**inputs, max_new_tokens=40, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        print(f"  {prompt!r}")
        print(f"    -> {tokenizer.decode(out[0], skip_special_tokens=True)}")
    ema.restore(student)

    # Save final
    final_dir = output_dir / "final"
    final_dir.mkdir(exist_ok=True)
    student.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    with open(final_dir / "staged_config.json", "w") as f:
        json.dump(asdict(staged_config), f, indent=2)

    with open(log_file, "a") as f:
        f.write(f"\n{'=' * 70}\n")
        f.write("Training Complete\n")
        f.write(f"{'=' * 70}\n")
        f.write(f"Early stopped: {early_stopped}\n")
        if early_stopped:
            f.write(f"Stop reason: {stop_reason}\n")
        f.write(f"Best checkpoint: step {best_checkpoint_step} (rep={best_rep_score:.1%})\n")
        f.write(f"Total time: {time.time() - start_time:.1f}s\n")

    print(f"\nTotal training time: {time.time() - start_time:.1f}s")
    print(f"Training log: {log_file}")
    print(f"Eval metrics: {jsonl_file}")
    print("\nDone!")


if __name__ == "__main__":
    main()
