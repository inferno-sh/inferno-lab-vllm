#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Distillation training script for Latent KV attention.

Optimized for RTX 3090 (24GB VRAM) with Qwen2.5-0.5B-Instruct.

Usage:
    python -m latent_kv.run_distillation
"""

import math
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from latent_kv.convert import convert_qwen2_to_latent_kv
from latent_kv.config import get_default_config


# Configuration
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = Path(os.environ.get("HOME", "/tmp")) / "temp" / "latent_kv_distill_v4"
COMPRESSION = "conservative"  # Less aggressive compression for early layers

# Training hyperparameters (following ChatGPT recommendations v4)
BATCH_SIZE = 2
GRAD_ACCUM = 8  # Effective batch size = 16
MAX_LENGTH = 256
LEARNING_RATE = 3e-4  # Higher LR for bottleneck-only training
WARMUP_RATIO = 0.05   # 5% warmup
TOTAL_STEPS = 10000   # Longer training
EVAL_EVERY = 500
SAVE_EVERY = 1000
LOG_EVERY = 50
GRAD_CLIP = 1.0

# =============================================================================
# Loss weights - CRITICAL CHANGES from v3:
# 1. CE floor = 0.5 (NEVER go below this - it's the language anchor)
# 2. KL increases but CE stays meaningful
# 3. Top-k KL (only match teacher's top-k tokens, not full vocab)
# 4. Unlikelihood loss for repeated n-grams (breaks repetition loops)
# =============================================================================
CE_WEIGHT_INIT = 1.0
CE_WEIGHT_FINAL = 0.5   # CHANGED: Floor at 0.5 (was 0.2 - caused collapse!)
KL_WEIGHT_INIT = 0.2
KL_WEIGHT_FINAL = 0.7   # CHANGED: Don't go too high (was 1.0)
UNLIKELIHOOD_WEIGHT = 0.5  # NEW: Penalize repeated n-grams
ATTN_OUT_WEIGHT = 1.0
HIDDEN_WEIGHT = 0.1
LABEL_SMOOTHING = 0.1
TEMPERATURE = 3.0
SCHEDULE_TRANSITION_STEP = 2000
EMA_DECAY = 0.999

# Top-k KL distillation (NEW - prevents gaming via junk tokens)
TOP_K_KL = 32  # Only match teacher's top-32 tokens per position

# Unlikelihood settings (NEW - breaks repetition loops)
NGRAM_SIZE = 3  # Penalize repeated 3-grams


def get_scheduled_weights(step):
    """
    Get scheduled loss weights based on current step.

    Phase 1 (0 to SCHEDULE_TRANSITION_STEP): Focus on learning sane language (CE=1.0, KL=0.2)
    Phase 2 (transition to end): Gradual shift but CE NEVER below 0.5 (the language anchor!)
    """
    if step < SCHEDULE_TRANSITION_STEP:
        # Phase 1: Fixed initial weights
        return CE_WEIGHT_INIT, KL_WEIGHT_INIT
    else:
        # Phase 2: Linear interpolation but CE stays >= 0.5 (CRITICAL!)
        progress = min(1.0, (step - SCHEDULE_TRANSITION_STEP) / (TOTAL_STEPS - SCHEDULE_TRANSITION_STEP))
        ce_w = CE_WEIGHT_INIT + progress * (CE_WEIGHT_FINAL - CE_WEIGHT_INIT)
        kl_w = KL_WEIGHT_INIT + progress * (KL_WEIGHT_FINAL - KL_WEIGHT_INIT)
        return ce_w, kl_w


def compute_ngram_repetition_mask(input_ids, n=3):
    """
    Find positions where the next token would create a repeated n-gram.

    Returns a mask [B, S] where 1 = this position is part of a repeated n-gram.
    Used for unlikelihood training to penalize repetition.
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    # Build set of seen n-grams for each batch item
    repetition_mask = torch.zeros(batch_size, seq_len, device=device)

    for b in range(batch_size):
        seen_ngrams = set()
        for i in range(seq_len - n + 1):
            ngram = tuple(input_ids[b, i:i+n].tolist())
            if ngram in seen_ngrams:
                # Mark all positions in this repeated n-gram
                repetition_mask[b, i:i+n] = 1.0
            seen_ngrams.add(ngram)

    return repetition_mask


def compute_unlikelihood_loss(logits, input_ids, repetition_mask):
    """
    Unlikelihood loss: penalize probability assigned to tokens that create repeated n-grams.

    For positions marked as repetitions, we want to DECREASE the probability of the
    token that was actually generated (because it created a repetition).

    Loss = -log(1 - p(repeated_token))
    """
    batch_size, seq_len, vocab_size = logits.shape
    device = logits.device

    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :]  # [B, S-1, V]
    shift_labels = input_ids[:, 1:]   # [B, S-1]
    shift_mask = repetition_mask[:, 1:]  # [B, S-1]

    # Get probabilities
    probs = F.softmax(shift_logits, dim=-1)  # [B, S-1, V]

    # Get probability of the actual next token
    next_token_probs = probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)  # [B, S-1]

    # Unlikelihood: -log(1 - p) for positions that are repetitions
    # Clamp to avoid log(0)
    unlikelihood = -torch.log(torch.clamp(1 - next_token_probs, min=1e-8))

    # Only apply to positions marked as repetitions
    masked_loss = unlikelihood * shift_mask

    # Average over repetition positions (if any)
    if shift_mask.sum() > 0:
        return masked_loss.sum() / (shift_mask.sum() + 1e-8)
    else:
        return torch.tensor(0.0, device=device)


class EMAModel:
    """Exponential Moving Average of model parameters for stable evaluation."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        """Update EMA parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self, model):
        """Apply EMA parameters to model."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model):
        """Restore original parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


class TextDataset(Dataset):
    """Simple dataset for distillation."""

    def __init__(self, texts, tokenizer, max_length):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
        }


class AttentionOutputCapture:
    """
    Hook to capture post-attention outputs from decoder layers.

    This captures the output of self_attn (before MLP), which is key for
    distilling KV-compressed models.
    """

    def __init__(self, model):
        self.outputs = []
        self.hooks = []
        self._register_hooks(model)

    def _register_hooks(self, model):
        """Register forward hooks on attention modules."""
        # Find the layers container
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "layers"):
            layers = model.layers
        else:
            print("Warning: Could not find layers for attention hooks")
            return

        for layer in layers:
            if hasattr(layer, "self_attn"):
                hook = layer.self_attn.register_forward_hook(self._hook_fn)
                self.hooks.append(hook)

    def _hook_fn(self, module, input, output):
        """Hook function to capture attention output."""
        # output is (attn_output, attn_weights) or just attn_output
        if isinstance(output, tuple):
            self.outputs.append(output[0].detach())
        else:
            self.outputs.append(output.detach())

    def clear(self):
        """Clear captured outputs."""
        self.outputs = []

    def get_outputs(self):
        """Get captured outputs and clear."""
        outputs = self.outputs
        self.outputs = []
        return outputs

    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def load_data(tokenizer, num_samples=5000):
    """Load WikiText-2 dataset."""
    print("Loading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    texts = []
    for item in dataset:
        text = item["text"]
        if text and len(text.strip()) > 100:
            texts.append(text.strip())
            if len(texts) >= num_samples:
                break

    print(f"  Loaded {len(texts)} training samples")

    # Split into train/eval
    eval_size = min(200, len(texts) // 10)
    train_texts = texts[:-eval_size]
    eval_texts = texts[-eval_size:]

    train_dataset = TextDataset(train_texts, tokenizer, MAX_LENGTH)
    eval_dataset = TextDataset(eval_texts, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, eval_loader


def compute_loss(student_out, teacher_out, input_ids, attention_mask, step,
                 attn_outputs_student=None, attn_outputs_teacher=None):
    """
    Compute distillation loss with CE + Top-k KL + Unlikelihood + attention matching.

    Following ChatGPT recommendations v4:
    - CE loss with LABEL SMOOTHING (language anchor - NEVER let weight drop below 0.5!)
    - TOP-K KL distillation (only match teacher's top-k tokens, not full vocab)
    - UNLIKELIHOOD loss (directly penalize repeated n-grams)
    - Attention-block output matching with MSE
    - Stopgrad on teacher targets

    Uses FP32 for numerical stability.
    """
    device = student_out.logits.device
    batch_size, seq_len, vocab_size = student_out.logits.shape

    # Get scheduled weights for this step (CE floor = 0.5!)
    ce_weight, kl_weight = get_scheduled_weights(step)

    # Convert to FP32 for numerical stability
    student_logits = student_out.logits.float()
    teacher_logits = teacher_out.logits.float()

    # Check for NaN/Inf in inputs
    if torch.isnan(student_logits).any() or torch.isinf(student_logits).any():
        print("WARNING: NaN/Inf in student logits!")
        student_logits = torch.nan_to_num(student_logits, nan=0.0, posinf=100.0, neginf=-100.0)

    # Clamp logits for numerical stability
    student_logits = student_logits.clamp(-100, 100)
    teacher_logits = teacher_logits.clamp(-100, 100)

    # =========================================================================
    # 1. Cross-Entropy Loss with LABEL SMOOTHING (the language anchor!)
    # =========================================================================
    shift_logits = student_logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous().float()

    ce_loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        reduction='none',
        label_smoothing=LABEL_SMOOTHING
    ).view(batch_size, seq_len - 1)

    ce_loss = (ce_loss * shift_mask).sum() / (shift_mask.sum() + 1e-8)

    # =========================================================================
    # 2. TOP-K KL Divergence (NEW - only match teacher's important tokens!)
    # This prevents gaming via junk tokens in the tail of the distribution
    # =========================================================================
    shift_student_logits = student_logits[:, :-1, :]  # [B, S-1, V]
    shift_teacher_logits = teacher_logits[:, :-1, :]  # [B, S-1, V]

    with torch.no_grad():
        # Get teacher's top-k tokens per position
        teacher_probs_full = F.softmax(shift_teacher_logits / TEMPERATURE, dim=-1)
        top_k_probs, top_k_indices = teacher_probs_full.topk(TOP_K_KL, dim=-1)  # [B, S-1, k]

        # Renormalize top-k to sum to 1
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)

    # Get student's log-probs for the same top-k tokens
    student_log_probs_full = F.log_softmax(shift_student_logits / TEMPERATURE, dim=-1)
    student_top_k_log_probs = student_log_probs_full.gather(2, top_k_indices)  # [B, S-1, k]

    # KL on top-k only: sum(p_teacher * (log(p_teacher) - log(p_student)))
    # Since teacher probs are detached, we compute: -sum(p_teacher * log(p_student)) + const
    # The const (teacher entropy) doesn't affect gradients
    kl_per_token = -(top_k_probs * student_top_k_log_probs).sum(dim=-1)  # [B, S-1]

    # Clamp KL values
    kl_per_token = kl_per_token.clamp(max=100.0)

    # Apply mask
    kl_loss = (kl_per_token * shift_mask).sum() / (shift_mask.sum() + 1e-8)
    kl_loss = kl_loss * (TEMPERATURE ** 2)

    # =========================================================================
    # 3. UNLIKELIHOOD Loss (NEW - directly breaks repetition loops!)
    # =========================================================================
    repetition_mask = compute_ngram_repetition_mask(input_ids, n=NGRAM_SIZE)
    unlikelihood_loss = compute_unlikelihood_loss(student_logits, input_ids, repetition_mask)

    # =========================================================================
    # 4. Attention output matching with MSE (stopgrad on teacher)
    # =========================================================================
    attn_out_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    if attn_outputs_student is not None and attn_outputs_teacher is not None and len(attn_outputs_student) > 0:
        num_layers = len(attn_outputs_student)
        match_layers = [num_layers // 4, num_layers // 2, 3 * num_layers // 4]

        attn_mask_float = attention_mask.float().unsqueeze(-1)

        count = 0
        for layer_idx in match_layers:
            if layer_idx < len(attn_outputs_student) and layer_idx < len(attn_outputs_teacher):
                s_attn = attn_outputs_student[layer_idx].float()
                t_attn = attn_outputs_teacher[layer_idx].float().detach()

                if torch.isnan(s_attn).any():
                    continue

                s_masked = s_attn * attn_mask_float
                t_masked = t_attn * attn_mask_float

                diff = s_masked - t_masked
                mse = (diff.pow(2).sum()) / (attn_mask_float.sum() * s_attn.shape[-1] + 1e-8)
                attn_out_loss += mse
                count += 1

        if count > 0:
            attn_out_loss = attn_out_loss / count

    # =========================================================================
    # 5. Hidden state MSE (every Nth layer, stopgrad on teacher)
    # =========================================================================
    hidden_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    if student_out.hidden_states and teacher_out.hidden_states:
        num_hidden = len(student_out.hidden_states)
        stride = max(1, num_hidden // 4)
        count = 0
        for i in range(0, num_hidden, stride):
            if i < len(teacher_out.hidden_states):
                s_h = student_out.hidden_states[i].float()
                t_h = teacher_out.hidden_states[i].float().detach()

                if torch.isnan(s_h).any():
                    continue

                s_norm = s_h / (s_h.norm(dim=-1, keepdim=True) + 1e-8)
                t_norm = t_h / (t_h.norm(dim=-1, keepdim=True) + 1e-8)

                hidden_loss += F.mse_loss(s_norm, t_norm)
                count += 1
        if count > 0:
            hidden_loss = hidden_loss / count

    # =========================================================================
    # Total Loss (CE weight floor = 0.5 prevents collapse!)
    # =========================================================================
    total_loss = (
        ce_weight * ce_loss +
        kl_weight * kl_loss +
        UNLIKELIHOOD_WEIGHT * unlikelihood_loss +
        ATTN_OUT_WEIGHT * attn_out_loss +
        HIDDEN_WEIGHT * hidden_loss
    )

    # Final NaN check
    if torch.isnan(total_loss):
        print(f"WARNING: NaN loss! CE={ce_loss.item()}, KL={kl_loss.item()}, UL={unlikelihood_loss.item()}")
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

    return {
        "total": total_loss,
        "ce": ce_loss,
        "kl": kl_loss,
        "unlikelihood": unlikelihood_loss,
        "attn_out": attn_out_loss,
        "hidden": hidden_loss,
        "ce_weight": ce_weight,
        "kl_weight": kl_weight,
    }


def compute_repetition_score(text, n=3):
    """
    Compute repetition score for generated text.

    Returns the fraction of n-grams that are repeated.
    Lower is better (0 = no repetition).
    """
    tokens = text.split()
    if len(tokens) < n:
        return 0.0

    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append(tuple(tokens[i:i+n]))

    if not ngrams:
        return 0.0

    unique_ngrams = set(ngrams)
    repetition_rate = 1 - len(unique_ngrams) / len(ngrams)
    return repetition_rate


@torch.no_grad()
def evaluate(student, teacher, eval_loader, device, step=0):
    """Evaluate student model."""
    student.eval()
    total_loss = 0
    total_ce = 0
    total_kl = 0
    total_ul = 0
    num_batches = 0

    for batch in eval_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        teacher_out = teacher(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        student_out = student(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        losses = compute_loss(student_out, teacher_out, input_ids, attention_mask, step)
        total_loss += losses["total"].item()
        total_ce += losses["ce"].item()
        total_kl += losses["kl"].item()
        total_ul += losses["unlikelihood"].item()
        num_batches += 1

    student.train()
    return {
        "loss": total_loss / num_batches,
        "ce": total_ce / num_batches,
        "kl": total_kl / num_batches,
        "ul": total_ul / num_batches,
    }


def evaluate_generation_quality(model, tokenizer, prompts, device):
    """
    Evaluate generation quality with repetition scoring.

    Returns generated texts and average repetition score.
    Used for checkpoint selection - pick checkpoint with lowest repetition.
    """
    model.eval()
    results = []
    total_rep_score = 0

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=40,  # Longer to detect repetition
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        rep_score = compute_repetition_score(text)
        results.append({"text": text, "repetition": rep_score})
        total_rep_score += rep_score

    model.train()
    avg_rep = total_rep_score / len(prompts) if prompts else 0

    return results, avg_rep


def generate_samples(model, tokenizer, prompts, device):
    """Generate samples for qualitative evaluation."""
    model.eval()
    results = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        results.append(text)

    model.train()
    return results


def main():
    print("=" * 70)
    print("Latent KV Distillation Training")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Compression: {COMPRESSION}")
    print(f"Batch size: {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
    print(f"Total steps: {TOTAL_STEPS}")
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load teacher model (FP16 for memory efficiency)
    print(f"Loading teacher model...")
    teacher = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
    ).to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"  Teacher parameters: {teacher_params:,}")

    # Get config and convert
    print(f"\nCreating latent KV student model...")
    config = get_default_config(teacher.config, COMPRESSION)
    config.print_summary()

    # Create student in FP32 for stable training
    student = convert_qwen2_to_latent_kv(
        teacher,
        config=config,
        init_method="svd",
        copy_model=True,
    )
    # Convert student to FP32 for stable training (FP16 causes NaN with AdamW)
    student = student.float()
    print("  Student converted to FP32 for stable training")

    # CRITICAL: Freeze all params EXCEPT latent-KV projections
    # This focuses training on the bottleneck and allows higher LR
    latent_kv_params = []
    frozen_count = 0
    trainable_count = 0
    for name, param in student.named_parameters():
        # Only train k_down, k_up, v_down, v_up, and k_anchor
        if any(x in name for x in ['k_down', 'k_up', 'v_down', 'v_up', 'k_anchor']):
            param.requires_grad = True
            latent_kv_params.append(param)
            trainable_count += param.numel()
        else:
            param.requires_grad = False
            frozen_count += param.numel()

    print(f"  Frozen parameters: {frozen_count:,}")
    print(f"  Trainable (latent-KV) parameters: {trainable_count:,}")

    student_params = sum(p.numel() for p in student.parameters())
    print(f"  Student parameters: {student_params:,}")
    print(f"  Parameter diff: {student_params - teacher_params:+,}")

    # Load data
    train_loader, eval_loader = load_data(tokenizer)

    # Test prompts for qualitative evaluation
    test_prompts = [
        "The capital of France is",
        "In the year 2024,",
        "The meaning of life is",
    ]

    # Initial generation test
    print("\n" + "=" * 70)
    print("Pre-training Generation Test")
    print("=" * 70)
    print("\nTeacher outputs:")
    teacher_samples = generate_samples(teacher, tokenizer, test_prompts, device)
    for prompt, output in zip(test_prompts, teacher_samples):
        print(f"  {prompt!r}")
        print(f"    -> {output}")

    print("\nStudent outputs (before training):")
    student_samples = generate_samples(student, tokenizer, test_prompts, device)
    for prompt, output in zip(test_prompts, student_samples):
        print(f"  {prompt!r}")
        print(f"    -> {output}")

    # Create optimizer - ONLY for latent-KV params at higher LR
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)

    optimizer = torch.optim.AdamW(
        latent_kv_params,  # Only train latent-KV params!
        lr=LEARNING_RATE,  # 3e-4 (higher than before since only training bottleneck)
        weight_decay=0.01,
    )

    # Cosine decay scheduler with warmup (instead of linear decay to 0)
    warmup_steps = int(TOTAL_STEPS * WARMUP_RATIO)
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        # Cosine decay to 10% of max LR (not 0)
        progress = (step - warmup_steps) / max(1, TOTAL_STEPS - warmup_steps)
        return 0.1 + 0.9 * (0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    student.train()
    global_step = 0
    running_loss = 0.0
    running_ce = 0.0
    running_kl = 0.0
    running_attn = 0.0
    running_ul = 0.0  # Unlikelihood (replaces conf)
    start_time = time.time()
    data_iter = iter(train_loader)

    # Track best checkpoint by repetition score
    best_rep_score = float('inf')
    best_checkpoint_step = 0

    # Set up attention output capture hooks
    print("Setting up attention output capture hooks...")
    teacher_attn_capture = AttentionOutputCapture(teacher)
    student_attn_capture = AttentionOutputCapture(student)
    print(f"  Teacher hooks: {len(teacher_attn_capture.hooks)}")
    print(f"  Student hooks: {len(student_attn_capture.hooks)}")

    # EMA for stable evaluation
    print("Setting up EMA...")
    ema = EMAModel(student, decay=EMA_DECAY)
    print(f"  EMA decay: {EMA_DECAY}")

    log_file = OUTPUT_DIR / "training_log.txt"
    with open(log_file, "w") as f:
        f.write(f"Training started at {datetime.now()}\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Compression: {COMPRESSION}\n")
        f.write(f"Cache reduction: {config.cache_size_reduction():.1%}\n")
        f.write(f"LR: {LEARNING_RATE}, Warmup: {WARMUP_RATIO*100:.0f}%, Steps: {TOTAL_STEPS}\n")
        f.write(f"Loss weights (scheduled): CE={CE_WEIGHT_INIT}→{CE_WEIGHT_FINAL}, KL={KL_WEIGHT_INIT}→{KL_WEIGHT_FINAL}\n")
        f.write(f"Fixed weights: AttnOut={ATTN_OUT_WEIGHT}, Hidden={HIDDEN_WEIGHT}, LabelSmooth={LABEL_SMOOTHING}\n\n")

    while global_step < TOTAL_STEPS:
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Clear attention captures
        teacher_attn_capture.clear()
        student_attn_capture.clear()

        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_out = teacher(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            teacher_attn_outputs = teacher_attn_capture.get_outputs()

        # Student forward
        student_out = student(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        student_attn_outputs = student_attn_capture.get_outputs()

        # Compute loss with attention output matching (pass step for scheduled weights)
        losses = compute_loss(
            student_out, teacher_out, input_ids, attention_mask, global_step,
            attn_outputs_student=student_attn_outputs,
            attn_outputs_teacher=teacher_attn_outputs,
        )
        loss = losses["total"] / GRAD_ACCUM
        loss.backward()

        running_loss += losses["total"].item()
        running_ce += losses["ce"].item()
        running_kl += losses["kl"].item()
        running_attn += losses["attn_out"].item()
        running_ul += losses["unlikelihood"].item()

        # Gradient accumulation step
        if (global_step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(latent_kv_params, GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            # Update EMA
            ema.update(student)

        global_step += 1

        # Logging
        if global_step % LOG_EVERY == 0:
            avg_loss = running_loss / LOG_EVERY
            avg_ce = running_ce / LOG_EVERY
            avg_kl = running_kl / LOG_EVERY
            avg_attn = running_attn / LOG_EVERY
            avg_ul = running_ul / LOG_EVERY
            lr = scheduler.get_last_lr()[0]
            ce_w, kl_w = get_scheduled_weights(global_step)
            elapsed = time.time() - start_time
            steps_per_sec = global_step / elapsed

            log_msg = (
                f"Step {global_step:5d}/{TOTAL_STEPS} | "
                f"Loss: {avg_loss:.4f} | "
                f"CE: {avg_ce:.4f} (w={ce_w:.2f}) | "
                f"KL: {avg_kl:.4f} (w={kl_w:.2f}) | "
                f"Attn: {avg_attn:.4f} | "
                f"UL: {avg_ul:.4f} | "
                f"LR: {lr:.2e} | "
                f"Speed: {steps_per_sec:.1f} steps/s"
            )
            print(log_msg)

            with open(log_file, "a") as f:
                f.write(log_msg + "\n")

            running_loss = 0.0
            running_ce = 0.0
            running_kl = 0.0
            running_attn = 0.0
            running_ul = 0.0

        # Evaluation
        if global_step % EVAL_EVERY == 0:
            print("\n--- Evaluation ---")
            eval_results = evaluate(student, teacher, eval_loader, device, global_step)
            print(f"  Eval Loss: {eval_results['loss']:.4f}")
            print(f"  Eval CE: {eval_results['ce']:.4f}")
            print(f"  Eval KL: {eval_results['kl']:.4f}")
            print(f"  Eval UL: {eval_results['ul']:.4f}")

            # Generate samples using EMA weights and compute repetition score
            print("\n  Student samples (EMA):")
            ema.apply_shadow(student)
            gen_results, avg_rep = evaluate_generation_quality(
                student, tokenizer, test_prompts, device
            )
            ema.restore(student)

            for prompt, result in zip(test_prompts, gen_results):
                print(f"    {prompt!r}")
                print(f"      -> {result['text']}")
                print(f"      Rep: {result['repetition']:.2%}")

            print(f"\n  Avg Repetition: {avg_rep:.2%}", end="")

            # Track best checkpoint by repetition score
            if avg_rep < best_rep_score:
                best_rep_score = avg_rep
                best_checkpoint_step = global_step
                print(f" (NEW BEST!)")
            else:
                print(f" (best: {best_rep_score:.2%} @ step {best_checkpoint_step})")

            print()

            with open(log_file, "a") as f:
                f.write(f"\n--- Eval at step {global_step} ---\n")
                f.write(f"Eval Loss: {eval_results['loss']:.4f}\n")
                f.write(f"Eval CE: {eval_results['ce']:.4f}\n")
                f.write(f"Eval KL: {eval_results['kl']:.4f}\n")
                f.write(f"Avg Repetition: {avg_rep:.2%}\n")
                for prompt, result in zip(test_prompts, gen_results):
                    f.write(f"  {prompt!r} -> {result['text']} (rep={result['repetition']:.2%})\n")
                f.write("\n")

        # Save checkpoint
        if global_step % SAVE_EVERY == 0:
            checkpoint_dir = OUTPUT_DIR / f"checkpoint-{global_step}"
            checkpoint_dir.mkdir(exist_ok=True)
            student.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            print(f"  Saved checkpoint to {checkpoint_dir}")

    # Final evaluation
    print("\n" + "=" * 70)
    print("Final Evaluation")
    print("=" * 70)

    eval_results = evaluate(student, teacher, eval_loader, device, TOTAL_STEPS)
    print(f"Final Eval Loss: {eval_results['loss']:.4f}")
    print(f"Final Eval CE: {eval_results['ce']:.4f}")
    print(f"Final Eval KL: {eval_results['kl']:.4f}")

    print("\nFinal student samples (EMA):")
    ema.apply_shadow(student)
    student_samples = generate_samples(student, tokenizer, test_prompts, device)
    ema.restore(student)
    for prompt, output in zip(test_prompts, student_samples):
        print(f"  {prompt!r}")
        print(f"    -> {output}")

    print("\nTeacher samples (reference):")
    teacher_samples = generate_samples(teacher, tokenizer, test_prompts, device)
    for prompt, output in zip(test_prompts, teacher_samples):
        print(f"  {prompt!r}")
        print(f"    -> {output}")

    # Save final model
    final_dir = OUTPUT_DIR / "final"
    final_dir.mkdir(exist_ok=True)
    student.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nSaved final model to {final_dir}")

    # Save summary
    with open(log_file, "a") as f:
        f.write(f"\n{'=' * 70}\n")
        f.write("Training Complete\n")
        f.write(f"{'=' * 70}\n")
        f.write(f"Final Eval Loss: {eval_results['loss']:.4f}\n")
        f.write(f"Final Eval CE: {eval_results['ce']:.4f}\n")
        f.write(f"Final Eval KL: {eval_results['kl']:.4f}\n")
        f.write(f"Total time: {time.time() - start_time:.1f}s\n")

    print(f"\nTotal training time: {time.time() - start_time:.1f}s")
    print(f"Training log saved to {log_file}")
    print("\nDone!")


if __name__ == "__main__":
    main()
