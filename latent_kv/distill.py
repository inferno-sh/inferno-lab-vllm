# SPDX-License-Identifier: Apache-2.0
"""
Distillation training for Latent KV attention.

Trains a converted model to match the outputs of a full-rank teacher model.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass
class DistillationConfig:
    """Configuration for distillation training."""

    # Loss weights
    logit_weight: float = 0.7  # KL divergence on logits
    hidden_weight: float = 0.2  # MSE on hidden states
    attention_weight: float = 0.1  # MSE on attention patterns

    # Temperature for KL divergence
    temperature: float = 2.0

    # Which layers to match for hidden state loss
    hidden_layer_stride: int = 4  # Every 4th layer

    # Training hyperparameters
    learning_rate: float = 5e-5
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Phases
    phase1_steps: int = 1000  # Warmup: only train latent projections
    phase1_lr: float = 1e-5
    phase2_steps: int = 10000  # Full training
    phase3_steps: int = 0  # Fine-tuning (optional)
    phase3_lr: float = 1e-6

    # Batch size
    batch_size: int = 4
    gradient_accumulation_steps: int = 4

    # Logging
    log_every: int = 100
    eval_every: int = 500
    save_every: int = 1000


class DistillationLoss(nn.Module):
    """
    Combined distillation loss for latent KV training.

    Components:
    1. Logit KL divergence - match output distribution
    2. Hidden state MSE - match intermediate representations
    3. Attention pattern MSE - match attention patterns (optional)
    """

    def __init__(self, config: DistillationConfig):
        super().__init__()
        self.config = config
        self.temperature = config.temperature

    def forward(
        self,
        student_outputs,
        teacher_outputs,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Compute distillation loss.

        Args:
            student_outputs: Student model outputs (with hidden_states, attentions)
            teacher_outputs: Teacher model outputs (with hidden_states, attentions)
            attention_mask: Attention mask for masking padded positions

        Returns:
            Dict with total_loss and component losses
        """
        losses = {}

        # 1. Logit KL divergence
        student_logits = student_outputs.logits
        teacher_logits = teacher_outputs.logits

        # Apply temperature scaling
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)

        # KL divergence (scaled by T^2 as per Hinton et al.)
        kl_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="batchmean",
        ) * (self.temperature ** 2)

        losses["logit_kl"] = kl_loss

        # 2. Hidden state MSE (on selected layers)
        if (
            self.config.hidden_weight > 0
            and hasattr(student_outputs, "hidden_states")
            and student_outputs.hidden_states is not None
        ):
            student_hiddens = student_outputs.hidden_states
            teacher_hiddens = teacher_outputs.hidden_states

            # Match every Nth layer
            stride = self.config.hidden_layer_stride
            hidden_loss = 0.0
            num_layers = 0

            for i in range(0, len(student_hiddens), stride):
                if i < len(teacher_hiddens):
                    s_h = student_hiddens[i]
                    t_h = teacher_hiddens[i]

                    # Apply attention mask if provided
                    if attention_mask is not None:
                        mask = attention_mask.unsqueeze(-1).float()
                        s_h = s_h * mask
                        t_h = t_h * mask

                    hidden_loss += F.mse_loss(s_h, t_h)
                    num_layers += 1

            if num_layers > 0:
                hidden_loss = hidden_loss / num_layers

            losses["hidden_mse"] = hidden_loss
        else:
            losses["hidden_mse"] = torch.tensor(0.0, device=student_logits.device)

        # 3. Attention pattern MSE (optional, expensive)
        if (
            self.config.attention_weight > 0
            and hasattr(student_outputs, "attentions")
            and student_outputs.attentions is not None
        ):
            student_attns = student_outputs.attentions
            teacher_attns = teacher_outputs.attentions

            attn_loss = 0.0
            num_layers = 0

            # Match every Nth layer (attention patterns are large)
            stride = self.config.hidden_layer_stride * 2
            for i in range(0, len(student_attns), stride):
                if i < len(teacher_attns):
                    s_a = student_attns[i]
                    t_a = teacher_attns[i]
                    attn_loss += F.mse_loss(s_a, t_a)
                    num_layers += 1

            if num_layers > 0:
                attn_loss = attn_loss / num_layers

            losses["attention_mse"] = attn_loss
        else:
            losses["attention_mse"] = torch.tensor(0.0, device=student_logits.device)

        # Total weighted loss
        total_loss = (
            self.config.logit_weight * losses["logit_kl"]
            + self.config.hidden_weight * losses["hidden_mse"]
            + self.config.attention_weight * losses["attention_mse"]
        )

        losses["total"] = total_loss

        return losses


class LatentKVDistillationTrainer:
    """
    Trainer for distilling a full-rank teacher into a latent KV student.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        config: DistillationConfig,
        tokenizer=None,
    ):
        """
        Initialize trainer.

        Args:
            teacher: Full-rank teacher model (frozen)
            student: Latent KV student model (to be trained)
            config: Training configuration
            tokenizer: Tokenizer for data processing
        """
        self.teacher = teacher
        self.student = student
        self.config = config
        self.tokenizer = tokenizer

        # Freeze teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Loss function
        self.loss_fn = DistillationLoss(config)

        # Optimizer will be created in train()
        self.optimizer = None
        self.scheduler = None

        # Training state
        self.global_step = 0
        self.current_phase = 1

    def _create_optimizer(self, phase: int):
        """Create optimizer for the given training phase."""
        if phase == 1:
            # Phase 1: Only train latent projections
            lr = self.config.phase1_lr
            params = []
            for name, param in self.student.named_parameters():
                if any(x in name for x in ["k_down", "k_up", "v_down", "v_up", "anchor"]):
                    param.requires_grad = True
                    params.append(param)
                else:
                    param.requires_grad = False
            print(f"Phase 1: Training {len(params)} parameter groups (latent only)")
        else:
            # Phase 2/3: Train all parameters
            lr = self.config.learning_rate if phase == 2 else self.config.phase3_lr
            for param in self.student.parameters():
                param.requires_grad = True
            params = self.student.parameters()
            print(f"Phase {phase}: Training all parameters")

        self.optimizer = torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=self.config.weight_decay,
        )

    def _get_lr_scheduler(self, num_training_steps: int):
        """Create learning rate scheduler with warmup."""
        from torch.optim.lr_scheduler import LambdaLR

        warmup_steps = self.config.warmup_steps

        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0,
                float(num_training_steps - current_step)
                / float(max(1, num_training_steps - warmup_steps)),
            )

        return LambdaLR(self.optimizer, lr_lambda)

    def train_step(self, batch: dict) -> dict:
        """
        Single training step.

        Args:
            batch: Dict with input_ids, attention_mask

        Returns:
            Dict with loss values
        """
        self.student.train()

        # Move batch to device
        device = next(self.student.parameters()).device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=self.config.attention_weight > 0,
            )

        # Student forward
        student_outputs = self.student(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=self.config.attention_weight > 0,
        )

        # Compute loss
        losses = self.loss_fn(student_outputs, teacher_outputs, attention_mask)

        # Backward
        loss = losses["total"] / self.config.gradient_accumulation_steps
        loss.backward()

        return {k: v.item() for k, v in losses.items()}

    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        eval_fn: Optional[Callable] = None,
        save_fn: Optional[Callable] = None,
    ):
        """
        Full training loop with phases.

        Args:
            train_dataloader: Training data
            eval_dataloader: Evaluation data (optional)
            eval_fn: Custom evaluation function (optional)
            save_fn: Custom save function (optional)
        """
        total_steps = (
            self.config.phase1_steps
            + self.config.phase2_steps
            + self.config.phase3_steps
        )

        print(f"\nStarting distillation training for {total_steps} steps")
        print(f"  Phase 1: {self.config.phase1_steps} steps (latent only)")
        print(f"  Phase 2: {self.config.phase2_steps} steps (full)")
        print(f"  Phase 3: {self.config.phase3_steps} steps (fine-tune)")

        # Phase transitions
        phase_boundaries = [
            self.config.phase1_steps,
            self.config.phase1_steps + self.config.phase2_steps,
            total_steps,
        ]

        # Start with phase 1
        self._create_optimizer(1)
        self.scheduler = self._get_lr_scheduler(self.config.phase1_steps)

        # Training loop
        running_losses = {"total": 0.0, "logit_kl": 0.0, "hidden_mse": 0.0}
        data_iter = iter(train_dataloader)

        while self.global_step < total_steps:
            # Get batch (cycle through data)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_dataloader)
                batch = next(data_iter)

            # Training step
            losses = self.train_step(batch)

            # Accumulate losses for logging
            for k in running_losses:
                if k in losses:
                    running_losses[k] += losses[k]

            # Gradient accumulation step
            if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.student.parameters(),
                    self.config.max_grad_norm,
                )

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            self.global_step += 1

            # Logging
            if self.global_step % self.config.log_every == 0:
                avg_losses = {
                    k: v / self.config.log_every for k, v in running_losses.items()
                }
                lr = self.scheduler.get_last_lr()[0]
                print(
                    f"Step {self.global_step}/{total_steps} | "
                    f"Phase {self.current_phase} | "
                    f"Loss: {avg_losses['total']:.4f} | "
                    f"KL: {avg_losses['logit_kl']:.4f} | "
                    f"Hidden: {avg_losses['hidden_mse']:.4f} | "
                    f"LR: {lr:.2e}"
                )
                running_losses = {k: 0.0 for k in running_losses}

            # Evaluation
            if eval_dataloader is not None and self.global_step % self.config.eval_every == 0:
                eval_loss = self.evaluate(eval_dataloader)
                print(f"  Eval loss: {eval_loss:.4f}")

                if eval_fn is not None:
                    eval_fn(self.student, self.global_step)

            # Save checkpoint
            if save_fn is not None and self.global_step % self.config.save_every == 0:
                save_fn(self.student, self.global_step)

            # Phase transitions
            if self.global_step == phase_boundaries[0] and self.current_phase == 1:
                print("\n=== Transitioning to Phase 2 ===\n")
                self.current_phase = 2
                self._create_optimizer(2)
                self.scheduler = self._get_lr_scheduler(self.config.phase2_steps)

            elif self.global_step == phase_boundaries[1] and self.current_phase == 2:
                if self.config.phase3_steps > 0:
                    print("\n=== Transitioning to Phase 3 ===\n")
                    self.current_phase = 3
                    self._create_optimizer(3)
                    self.scheduler = self._get_lr_scheduler(self.config.phase3_steps)

        print("\nTraining complete!")

    @torch.no_grad()
    def evaluate(self, eval_dataloader: DataLoader) -> float:
        """
        Evaluate the student model.

        Args:
            eval_dataloader: Evaluation data

        Returns:
            Average loss
        """
        self.student.eval()
        total_loss = 0.0
        num_batches = 0

        device = next(self.student.parameters()).device

        for batch in eval_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            teacher_outputs = self.teacher(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            student_outputs = self.student(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            losses = self.loss_fn(student_outputs, teacher_outputs, attention_mask)
            total_loss += losses["total"].item()
            num_batches += 1

        self.student.train()
        return total_loss / max(1, num_batches)


def create_distillation_dataloader(
    tokenizer,
    texts: list[str],
    batch_size: int = 4,
    max_length: int = 512,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for distillation training.

    Args:
        tokenizer: Tokenizer
        texts: List of training texts
        batch_size: Batch size
        max_length: Maximum sequence length
        shuffle: Whether to shuffle

    Returns:
        DataLoader
    """
    from torch.utils.data import Dataset

    class TextDataset(Dataset):
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

    dataset = TextDataset(texts, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
