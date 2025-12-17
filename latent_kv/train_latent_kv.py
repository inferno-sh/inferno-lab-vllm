#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Training script for Latent KV attention.

Usage:
    python -m latent_kv.train_latent_kv --model Qwen/Qwen2.5-0.5B-Instruct --output ./latent_kv_model

This script:
1. Loads a pretrained model
2. Converts it to use Latent KV attention
3. Distills from the original model
4. Saves the converted model
"""

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from latent_kv.convert import convert_qwen2_to_latent_kv, verify_conversion, print_model_comparison
from latent_kv.config import get_default_config
from latent_kv.distill import (
    DistillationConfig,
    LatentKVDistillationTrainer,
    create_distillation_dataloader,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Latent KV model via distillation")

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./latent_kv_output",
        help="Output directory for checkpoints",
    )

    # Compression
    parser.add_argument(
        "--compression",
        type=str,
        default="moderate",
        choices=["aggressive", "moderate", "conservative"],
        help="Compression level",
    )

    # Training
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--phase1-steps", type=int, default=500, help="Phase 1 (warmup) steps")
    parser.add_argument("--phase2-steps", type=int, default=2000, help="Phase 2 (full) steps")
    parser.add_argument("--max-length", type=int, default=256, help="Max sequence length")

    # Data
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext",
        help="Dataset name (from HuggingFace)",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="wikitext-2-raw-v1",
        help="Dataset configuration",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5000,
        help="Number of training samples",
    )

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-verify", action="store_true", help="Skip verification")
    parser.add_argument("--dry-run", action="store_true", help="Only convert, don't train")

    return parser.parse_args()


def load_training_data(args, tokenizer):
    """Load training data from HuggingFace datasets."""
    print(f"\nLoading dataset: {args.dataset}/{args.dataset_config}")

    dataset = load_dataset(args.dataset, args.dataset_config, split="train")

    # Get text column
    text_column = "text" if "text" in dataset.column_names else dataset.column_names[0]

    # Filter and sample
    texts = []
    for item in dataset:
        text = item[text_column]
        if text and len(text.strip()) > 50:  # Filter short/empty texts
            texts.append(text.strip())
            if len(texts) >= args.num_samples:
                break

    print(f"  Loaded {len(texts)} training samples")

    # Create dataloader
    dataloader = create_distillation_dataloader(
        tokenizer,
        texts,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=True,
    )

    return dataloader


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load model and tokenizer
    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load teacher model
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    teacher.eval()

    print(f"  Model loaded: {sum(p.numel() for p in teacher.parameters()):,} parameters")

    # Get latent config and print summary
    config = get_default_config(teacher.config, args.compression)
    print("\n")
    config.print_summary()

    # Convert to latent KV
    print(f"\nConverting model to Latent KV attention...")
    student = convert_qwen2_to_latent_kv(
        teacher,
        config=config,
        compression_level=args.compression,
        init_method="svd",
        copy_model=True,
    )

    # Move student to device
    student = student.to(device)

    # Print comparison
    print_model_comparison(teacher, student)

    # Verify conversion
    if not args.no_verify:
        print("\nVerifying conversion...")
        results = verify_conversion(teacher, student, tokenizer)
        print(f"  Top-1 match: {results['all_top1_match']}")
        print(f"  Avg max logit diff: {results['avg_max_diff']:.4f}")
        print(f"  Avg mean logit diff: {results['avg_mean_diff']:.4f}")

    if args.dry_run:
        print("\nDry run complete. Exiting.")
        return

    # Load training data
    train_dataloader = load_training_data(args, tokenizer)

    # Create distillation config
    distill_config = DistillationConfig(
        learning_rate=args.lr,
        phase1_steps=args.phase1_steps,
        phase2_steps=args.phase2_steps,
        phase3_steps=0,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        log_every=50,
        eval_every=500,
        save_every=500,
    )

    # Create trainer
    trainer = LatentKVDistillationTrainer(
        teacher=teacher,
        student=student,
        config=distill_config,
        tokenizer=tokenizer,
    )

    # Save function
    def save_checkpoint(model, step):
        checkpoint_dir = output_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"  Saved checkpoint to {checkpoint_dir}")

    # Train
    print("\nStarting training...")
    trainer.train(
        train_dataloader=train_dataloader,
        save_fn=save_checkpoint,
    )

    # Save final model
    final_dir = output_dir / "final"
    final_dir.mkdir(exist_ok=True)
    student.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nSaved final model to {final_dir}")

    # Final verification
    print("\nFinal verification...")
    results = verify_conversion(teacher, student, tokenizer)
    print(f"  Top-1 match: {results['all_top1_match']}")
    print(f"  Avg max logit diff: {results['avg_max_diff']:.4f}")

    # Cache size info
    cache_info = student.latent_kv_config.cache_size_reduction()
    print(f"\nKV Cache size reduction: {cache_info:.1%}")


if __name__ == "__main__":
    main()
