"""Unit tests for ScaleFlow models modules:
- scale_embed.py
- encoder.py
- decoder.py
- model.py
"""

import math
import shutil
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from scaleflow.models.scale_embed import (
    SCALE_RESOLUTIONS,
    RESOLUTION_TO_SCALE,
    SinusoidalScaleEmbedding,
    resolution_to_scale,
    scale_to_resolution,
)
from scaleflow.models.encoder import (
    DepthwiseGatedConvBlock,
    MultiHeadSelfAttentionBlock,
    HybridTextEncoder,
)
from scaleflow.models.decoder import (
    GRN,
    ConvNeXtV2AdaLNBlock,
    ScaleAwareCrossAttention,
    PixelShuffleUpsampler,
    ScaleFlowDecoder,
)
from scaleflow.models.model import (
    ScaleFlowConfig,
    ScaleFlowModel,
)
from scaleflow.tokenizer import BPETokenizer


class TestSinusoidalScaleEmbedding(unittest.TestCase):
    """Test SinusoidalScaleEmbedding and scale/resolution helpers."""

    def setUp(self):
        self.dim = 128
        self.embed = SinusoidalScaleEmbedding(dim=self.dim, max_scale=8, use_mlp=True)

    def test_resolution_mapping(self):
        """Verify discrete scale levels [0..7] mapping to resolutions 8x8 to 1024x1024."""
        expected = [8, 16, 32, 64, 128, 256, 512, 1024]
        for s in range(8):
            res = scale_to_resolution(s)
            self.assertEqual(res, expected[s], f"Scale {s} should map to {expected[s]}, got {res}")
            rev_s = resolution_to_scale(res)
            self.assertEqual(rev_s, s, f"Resolution {res} should map back to scale {s}, got {rev_s}")

    def test_forward_tensor(self):
        """Test embedding generation from 1D tensor input."""
        scales = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long)
        out = self.embed(scales)
        self.assertEqual(out.shape, (8, self.dim))
        self.assertFalse(torch.isnan(out).any())

    def test_forward_scalar_and_int(self):
        """Test embedding generation from scalar/integer input."""
        out_int = self.embed(0)
        self.assertEqual(out_int.shape, (1, self.dim))

        out_float = self.embed(3.5)
        self.assertEqual(out_float.shape, (1, self.dim))

    def test_distinct_embeddings(self):
        """Ensure distinct scale levels produce distinct embeddings."""
        scales = torch.arange(8)
        out = self.embed(scales)
        # Check cosine similarity between level 0 and level 7
        norm_out = out / out.norm(dim=-1, keepdim=True)
        sim = (norm_out[0] * norm_out[7]).sum().item()
        self.assertLess(sim, 0.99, "Distinct scale levels should produce distinct embeddings")

    def test_gradient_flow(self):
        """Verify gradient flows through scale embedding."""
        scales = torch.tensor([1, 2, 3], dtype=torch.float32)
        out = self.embed(scales)
        loss = out.sum()
        loss.backward()
        for p in self.embed.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad)


class TestHybridTextEncoder(unittest.TestCase):
    """Test Hybrid LFM2-style text encoder."""

    def setUp(self):
        self.vocab_size = 256
        self.d_model = 64
        self.depth = 6
        self.max_seq_len = 32
        self.encoder = HybridTextEncoder(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            depth=self.depth,
            num_heads=4,
            conv_kernel_size=5,
            conv_expansion=2,
            max_seq_len=self.max_seq_len,
        )

    def test_layer_ratio(self):
        """Verify 2:1 ratio of Conv to Attention layers."""
        conv_count = sum(1 for layer in self.encoder.layers if isinstance(layer, DepthwiseGatedConvBlock))
        attn_count = sum(1 for layer in self.encoder.layers if isinstance(layer, MultiHeadSelfAttentionBlock))
        self.assertEqual(conv_count, 4, f"Expected 4 Conv blocks for depth=6, got {conv_count}")
        self.assertEqual(attn_count, 2, f"Expected 2 Attention blocks for depth=6, got {attn_count}")
        self.assertEqual(conv_count / attn_count, 2.0, "Ratio must be exactly 2:1")

    def test_forward_output_shape(self):
        """Verify output shape e_text in R^D."""
        B, L = 4, 16
        input_ids = torch.randint(0, self.vocab_size, (B, L))
        e_text = self.encoder(input_ids)
        self.assertEqual(e_text.shape, (B, self.d_model))
        self.assertFalse(torch.isnan(e_text).any())

    def test_forward_with_mask(self):
        """Verify forward pass with attention mask."""
        B, L = 2, 8
        input_ids = torch.randint(1, self.vocab_size, (B, L))
        # Mask out second half of sequence
        attention_mask = torch.tensor([
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 0],
        ], dtype=torch.long)
        e_text = self.encoder(input_ids, attention_mask=attention_mask)
        self.assertEqual(e_text.shape, (B, self.d_model))

    def test_return_sequence(self):
        """Verify return_sequence returns both pooled and sequence embeddings."""
        B, L = 2, 10
        input_ids = torch.randint(0, self.vocab_size, (B, L))
        pooled, seq = self.encoder(input_ids, return_sequence=True)
        self.assertEqual(pooled.shape, (B, self.d_model))
        self.assertEqual(seq.shape, (B, L, self.d_model))

    def test_gradient_flow(self):
        """Verify gradient flow through encoder."""
        input_ids = torch.randint(0, self.vocab_size, (2, 8))
        e_text = self.encoder(input_ids)
        loss = e_text.sum()
        loss.backward()
        self.assertIsNotNone(self.encoder.token_embed.weight.grad)


class TestScaleFlowDecoder(unittest.TestCase):
    """Test Recursive ScaleFlowDecoder and ConvNeXt V2 components."""

    def setUp(self):
        self.cond_dim = 64
        self.hidden_dim = 32
        self.decoder = ScaleFlowDecoder(
            in_channels=3,
            out_channels=3,
            hidden_dim=self.hidden_dim,
            cond_dim=self.cond_dim,
            depth=2,
        )

    def test_grn(self):
        """Verify GRN layer preserves shape and normalizes."""
        grn = GRN(dim=self.hidden_dim)
        x = torch.randn(2, 8, 8, self.hidden_dim)
        out = grn(x)
        self.assertEqual(out.shape, x.shape)

    def test_convnext_v2_adaln_block(self):
        """Verify ConvNeXt V2 block with AdaLN preserves spatial shape."""
        block = ConvNeXtV2AdaLNBlock(dim=self.hidden_dim, cond_dim=self.cond_dim)
        x = torch.randn(2, self.hidden_dim, 16, 16)
        c = torch.randn(2, self.cond_dim)
        out = block(x, c)
        self.assertEqual(out.shape, x.shape)

    def test_pixelshuffle_upsampler(self):
        """Verify PixelShuffle(2x) doubles spatial resolution."""
        upsampler = PixelShuffleUpsampler(in_channels=self.hidden_dim, out_channels=3, upscale_factor=2)
        x = torch.randn(2, self.hidden_dim, 16, 16)
        out = upsampler(x)
        self.assertEqual(out.shape, (2, 3, 32, 32))

    def test_all_scales_shape_transitions(self):
        """Thoroughly verify tensor shapes across all scales (8->16->...->1024)."""
        scales = [8, 16, 32, 64, 128, 256, 512]
        expected_outputs = [16, 32, 64, 128, 256, 512, 1024]

        e_text = torch.randn(1, self.cond_dim)
        e_scale = torch.randn(1, self.cond_dim)

        for in_res, out_res in zip(scales, expected_outputs):
            x = torch.randn(1, 3, in_res, in_res)
            out = self.decoder(x, e_text=e_text, e_scale=e_scale, upsample=True)
            self.assertEqual(
                out.shape,
                (1, 3, out_res, out_res),
                f"Expected upsampling from {in_res}x{in_res} to {out_res}x{out_res}, got {out.shape}",
            )

    def test_same_resolution_head(self):
        """Verify decoder with upsample=False retains resolution."""
        x = torch.randn(2, 3, 32, 32)
        e_text = torch.randn(2, self.cond_dim)
        e_scale = torch.randn(2, self.cond_dim)
        out = self.decoder(x, e_text=e_text, e_scale=e_scale, upsample=False)
        self.assertEqual(out.shape, (2, 3, 32, 32))

    def test_scale_aware_cross_attention_block(self):
        """Verify ScaleAwareCrossAttention behavior across scale threshold."""
        cross_attn = ScaleAwareCrossAttention(
            dim=self.hidden_dim,
            context_dim=self.cond_dim,
            num_heads=4,
            cross_attention_max_scale=3,
        )
        x = torch.randn(2, self.hidden_dim, 16, 16)
        text_seq = torch.randn(2, 8, self.cond_dim)

        # 1. At scale 2 (<= 3): active, but gate=0 initially -> returns x
        out_init = cross_attn(x, text_sequence=text_seq, scale=2)
        torch.testing.assert_close(out_init, x)

        # Set gate to non-zero -> output differs
        cross_attn.gate.data.fill_(1.0)
        out_active = cross_attn(x, text_sequence=text_seq, scale=2)
        self.assertEqual(out_active.shape, x.shape)
        self.assertFalse(torch.allclose(out_active, x))

        # 2. At scale 4 (> 3): completely bypassed with 0 compute -> returns x unchanged
        out_bypassed = cross_attn(x, text_sequence=text_seq, scale=4)
        torch.testing.assert_close(out_bypassed, x)

        # 3. If text_sequence is None: returns x unchanged
        out_none = cross_attn(x, text_sequence=None, scale=1)
        torch.testing.assert_close(out_none, x)


class TestScaleFlowModel(unittest.TestCase):
    """Test integrated ScaleFlowModel."""

    def setUp(self):
        self.config = ScaleFlowConfig(
            vocab_size=128,
            text_dim=64,
            text_depth=3,
            text_heads=4,
            conv_kernel_size=3,
            conv_expansion=2,
            max_seq_len=16,
            decoder_dim=32,
            decoder_depth=2,
            in_channels=3,
            out_channels=3,
        )
        self.model = ScaleFlowModel(config=self.config)

    def test_parameter_counts(self):
        """Verify parameter counting utilities."""
        total = self.model.num_parameters()
        trainable = self.model.num_parameters(only_trainable=True)
        breakdown = self.model.parameter_breakdown()

        self.assertGreater(total, 0)
        self.assertEqual(total, trainable)
        self.assertIn("text_encoder", breakdown)
        self.assertIn("scale_embed", breakdown)
        self.assertIn("decoder", breakdown)
        self.assertEqual(
            breakdown["text_encoder"] + breakdown["scale_embed"] + breakdown["decoder"],
            total,
        )

    def test_forward_with_tokens(self):
        """Verify forward pass taking input_ids directly."""
        B = 2
        x = torch.randn(B, 3, 8, 8)
        input_ids = torch.randint(0, self.config.vocab_size, (B, 10))
        scale = 0

        out = self.model(x, scale=scale, input_ids=input_ids)
        self.assertEqual(out.shape, (B, 3, 16, 16))

    def test_forward_with_precomputed_embeddings(self):
        """Verify forward pass with precomputed text and scale embeddings."""
        B = 2
        x = torch.randn(B, 3, 16, 16)
        e_text = torch.randn(B, self.config.text_dim)
        e_scale = torch.randn(B, self.config.text_dim)

        out = self.model(x, e_text=e_text, e_scale=e_scale, upsample=True)
        self.assertEqual(out.shape, (B, 3, 32, 32))

    def test_recursive_pyramid_generation(self):
        """Verify recursive multi-resolution pyramid generation from 8x8 up to 1024x1024."""
        x_init = torch.randn(1, 3, 8, 8)
        input_ids = torch.tensor([[1, 5, 10, 2]])
        e_text = self.model.encode_text(input_ids)

        # 7 recursive steps: 8 -> 16 -> 32 -> 64 -> 128 -> 256 -> 512 -> 1024
        pyramid = self.model.generate_pyramid(x_init, e_text=e_text, start_scale=0, num_steps=7)

        expected_sizes = [8, 16, 32, 64, 128, 256, 512, 1024]
        self.assertEqual(len(pyramid), len(expected_sizes))
        for img, size in zip(pyramid, expected_sizes):
            self.assertEqual(img.shape, (1, 3, size, size))

    def test_encode_prompts_with_tokenizer(self):
        """Verify encode_prompts integrates seamlessly with BPETokenizer."""
        tokenizer = BPETokenizer()
        tokenizer.train(["a photo of a cat", "a scenic mountain view"], vocab_size=32)
        e_text = self.model.encode_prompts(["a photo of a cat", "mountain view"], tokenizer=tokenizer)
        self.assertEqual(e_text.shape, (2, self.config.text_dim))
        self.assertFalse(torch.isnan(e_text).any())

    def test_cross_attention_forward_and_pyramid(self):
        """Verify model forward pass and pyramid generation with cross-attention enabled."""
        B = 2
        x = torch.randn(B, 3, 8, 8)
        input_ids = torch.randint(0, self.config.vocab_size, (B, 8))
        
        # Forward with tokens automatically computes both e_text and text_sequence
        out = self.model(x, scale=0, input_ids=input_ids, upsample=True)
        self.assertEqual(out.shape, (B, 3, 16, 16))
        self.assertFalse(torch.isnan(out).any())

        # Forward with explicit text_sequence
        e_text, seq = self.model.encode_text(input_ids, return_sequence=True)
        out_explicit = self.model(x, scale=0, e_text=e_text, text_sequence=seq, upsample=True)
        self.assertEqual(out_explicit.shape, (B, 3, 16, 16))

        # Full pyramid with text_sequence
        pyramid = self.model.generate_pyramid(
            x[:1], e_text=e_text[:1], text_sequence=seq[:1], start_scale=0, num_steps=7
        )
        self.assertEqual(len(pyramid), 8)
        self.assertEqual(pyramid[-1].shape, (1, 3, 1024, 1024))

    def test_save_and_load_pretrained(self):
        """Verify .save_pretrained() and .from_pretrained() round-trip."""
        temp_dir = tempfile.mkdtemp()
        try:
            # 1. Save
            self.model.save_pretrained(temp_dir)
            self.assertTrue((Path(temp_dir) / "config.json").exists())
            self.assertTrue((Path(temp_dir) / "model.pt").exists())

            # 2. Load
            loaded_model = ScaleFlowModel.from_pretrained(temp_dir)
            loaded_model.eval()
            self.model.eval()

            # 3. Check configuration equality
            self.assertEqual(self.model.config.to_dict(), loaded_model.config.to_dict())

            # 4. Check forward output equality
            x = torch.randn(1, 3, 8, 8)
            e_text = torch.randn(1, self.config.text_dim)
            e_scale = torch.randn(1, self.config.text_dim)

            with torch.no_grad():
                out_orig = self.model(x, e_text=e_text, e_scale=e_scale)
                out_loaded = loaded_model(x, e_text=e_text, e_scale=e_scale)

            torch.testing.assert_close(out_orig, out_loaded, rtol=1e-5, atol=1e-5)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
