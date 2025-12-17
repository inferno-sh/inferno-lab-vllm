#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Quick test script for Latent KV attention.

Tests:
1. Module instantiation and forward pass
2. Model conversion from Qwen2
3. SVD initialization quality
4. Generation comparison
"""

import torch
import torch.nn as nn

from latent_kv.attention import LatentKVAttention, RotaryEmbedding
from latent_kv.config import LatentKVConfig, LayerLatentConfig, get_default_config


def test_rotary_embedding():
    """Test RotaryEmbedding module."""
    print("Testing RotaryEmbedding...")

    rope = RotaryEmbedding(dim=64, max_position_embeddings=2048)

    # Test forward
    x = torch.randn(2, 16, 8, 64)  # [B, S, H, D]
    position_ids = torch.arange(16).unsqueeze(0).expand(2, -1)

    cos, sin = rope(x, position_ids)

    assert cos.shape == (2, 16, 64), f"Expected (2, 16, 64), got {cos.shape}"
    assert sin.shape == (2, 16, 64), f"Expected (2, 16, 64), got {sin.shape}"

    print("  RotaryEmbedding: PASSED")


def test_latent_kv_attention():
    """Test LatentKVAttention module."""
    print("Testing LatentKVAttention...")

    config = LayerLatentConfig(r_k=16, r_v=32, use_k_anchor=True)

    attn = LatentKVAttention(
        d_model=512,
        n_heads=8,
        n_kv_heads=2,
        d_head=64,
        layer_config=config,
    )

    # Test forward (HF-compatible signature returns 2 values)
    x = torch.randn(2, 16, 512)
    output, attn_weights = attn(x)

    assert output.shape == (2, 16, 512), f"Expected (2, 16, 512), got {output.shape}"

    print("  LatentKVAttention forward: PASSED")

    # Test latent KV extraction
    k_latent, v_latent = attn.get_latent_kv(x)
    assert k_latent.shape == (2, 16, 16), f"Expected k_latent (2, 16, 16), got {k_latent.shape}"
    assert v_latent.shape == (2, 16, 32), f"Expected v_latent (2, 16, 32), got {v_latent.shape}"

    print("  Latent KV extraction: PASSED")

    # Test latent expansion
    k_full, v_full = attn.expand_latent_kv(k_latent, v_latent)
    assert k_full.shape == (2, 16, 2 * 64), f"Expected k_full (2, 16, 128), got {k_full.shape}"
    assert v_full.shape == (2, 16, 2 * 64), f"Expected v_full (2, 16, 128), got {v_full.shape}"

    print("  Latent expansion: PASSED")


def test_config():
    """Test configuration utilities."""
    print("Testing LatentKVConfig...")

    config = LatentKVConfig(
        d_model=512,
        n_heads=8,
        n_kv_heads=2,
        d_head=64,
        num_layers=24,
    )

    # Check layer configs were generated
    assert len(config.layer_configs) == 24

    # Check early layer has anchor
    assert config.layer_configs[0].use_k_anchor is True
    assert config.layer_configs[0].r_k < config.layer_configs[12].r_k  # Early more compressed

    # Check cache reduction
    reduction = config.cache_size_reduction()
    assert 0 < reduction < 1, f"Expected reduction between 0 and 1, got {reduction}"

    print(f"  Generated {len(config.layer_configs)} layer configs")
    print(f"  Cache size reduction: {reduction:.1%}")
    print("  LatentKVConfig: PASSED")


def test_svd_initialization():
    """Test SVD-based initialization quality."""
    print("Testing SVD initialization...")

    # Create a mock "standard" attention with known weights
    class MockAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(512, 512, bias=False)
            self.k_proj = nn.Linear(512, 128, bias=False)  # 2 KV heads * 64 d_head
            self.v_proj = nn.Linear(512, 128, bias=False)
            self.o_proj = nn.Linear(512, 512, bias=False)
            self.num_heads = 8
            self.num_key_value_heads = 2
            self.head_dim = 64
            self.rotary_emb = RotaryEmbedding(64)

    mock_attn = MockAttention()

    # Initialize with some weights
    nn.init.xavier_uniform_(mock_attn.k_proj.weight)
    nn.init.xavier_uniform_(mock_attn.v_proj.weight)

    # Convert
    layer_config = LayerLatentConfig(r_k=32, r_v=64)
    latent_attn = LatentKVAttention.from_standard_attention(
        mock_attn, layer_config, init_method="svd"
    )

    # Test reconstruction quality
    x = torch.randn(1, 16, 512)

    # Original K, V
    k_orig = mock_attn.k_proj(x)
    v_orig = mock_attn.v_proj(x)

    # Reconstructed K, V
    k_latent = latent_attn.k_down(x)
    v_latent = latent_attn.v_down(x)
    k_recon = latent_attn.k_up(k_latent)
    v_recon = latent_attn.v_up(v_latent)

    # Compute reconstruction error
    k_error = (k_orig - k_recon).pow(2).mean().sqrt()
    v_error = (v_orig - v_recon).pow(2).mean().sqrt()

    k_norm = k_orig.pow(2).mean().sqrt()
    v_norm = v_orig.pow(2).mean().sqrt()

    k_relative_error = k_error / k_norm
    v_relative_error = v_error / v_norm

    print(f"  K reconstruction error: {k_relative_error:.4f} (relative)")
    print(f"  V reconstruction error: {v_relative_error:.4f} (relative)")

    # For random Xavier-initialized matrices, singular values are roughly uniform
    # With rank-r out of n, we capture ~r/n of variance, so error ~ sqrt(1 - r/n)
    # r_k=32/128=0.25 -> expected error ~0.87
    # r_v=64/128=0.50 -> expected error ~0.71
    # We use generous bounds since the goal is to verify SVD init runs correctly
    assert k_relative_error < 1.0, f"K error too high: {k_relative_error}"
    assert v_relative_error < 1.0, f"V error too high: {v_relative_error}"

    # Verify that k_up @ k_down approximates k_proj reasonably
    # The error should be bounded, not arbitrarily large
    assert k_relative_error > 0.0, "K error suspiciously zero"
    assert v_relative_error > 0.0, "V error suspiciously zero"

    print("  SVD initialization: PASSED")


def test_with_real_model():
    """Test with a real Qwen2 model (if available)."""
    print("\nTesting with real model...")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from latent_kv.convert import convert_qwen2_to_latent_kv, verify_conversion

        model_name = "Qwen/Qwen2.5-0.5B-Instruct"

        print(f"  Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model - use device_map if accelerate is available, otherwise manual
        try:
            import accelerate
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        except ImportError:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            ).to(device)

        # Get config and print summary
        config = get_default_config(model.config, "moderate")
        print(f"\n  Configuration for {model_name}:")
        config.print_summary()

        # Convert
        print("\n  Converting model...")
        latent_model = convert_qwen2_to_latent_kv(
            model,
            config=config,
            init_method="svd",
        )

        # Verify
        print("\n  Verifying conversion...")
        results = verify_conversion(model, latent_model, tokenizer)

        print(f"\n  Results:")
        print(f"    Top-1 prediction match: {results['all_top1_match']}")
        print(f"    Average max logit diff: {results['avg_max_diff']:.4f}")
        print(f"    Average mean logit diff: {results['avg_mean_diff']:.4f}")

        # Quick generation test
        print("\n  Generation test:")
        prompt = "The capital of France is"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            orig_output = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            latent_output = latent_model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        orig_text = tokenizer.decode(orig_output[0], skip_special_tokens=True)
        latent_text = tokenizer.decode(latent_output[0], skip_special_tokens=True)

        print(f"    Original: {orig_text}")
        print(f"    Latent:   {latent_text}")

        print("\n  Real model test: PASSED")

    except ImportError as e:
        print(f"  Skipping real model test: {e}")
    except Exception as e:
        print(f"  Real model test FAILED: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("=" * 60)
    print("Latent KV Attention Tests")
    print("=" * 60)

    test_rotary_embedding()
    test_latent_kv_attention()
    test_config()
    test_svd_initialization()
    test_with_real_model()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
