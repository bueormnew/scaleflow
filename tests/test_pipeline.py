"""
Unit tests for ScaleFlow pipeline, sampler, and image utilities.
"""

from pathlib import Path
import tempfile
import unittest
import torch
from PIL import Image

from scaleflow.models.model import ScaleFlowModel, ScaleFlowConfig
from scaleflow.tokenizer.bpe import BPETokenizer
from scaleflow.pipeline.sampler import MultiscaleFlowSampler
from scaleflow.pipeline.pipeline import ScaleFlowPipeline
from scaleflow.pipeline.utils import (
    tensor_to_pil,
    pil_to_tensor,
    make_image_grid,
    save_image,
)


class TestPipelineUtils(unittest.TestCase):
    """Test pipeline utility functions."""

    def test_tensor_to_pil_and_back(self):
        # Create dummy tensor in [-1, 1]
        t = torch.zeros(1, 3, 32, 32)  # mid-gray
        pil_img = tensor_to_pil(t, value_range="symmetric")
        self.assertIsInstance(pil_img, Image.Image)
        self.assertEqual(pil_img.size, (32, 32))

        # Convert back
        t_rec = pil_to_tensor(pil_img, value_range="symmetric")
        self.assertEqual(t_rec.shape, (1, 3, 32, 32))
        self.assertTrue(torch.allclose(t, t_rec, atol=0.05))

    def test_tensor_to_pil_batch(self):
        t_batch = torch.randn(3, 3, 16, 16).clamp(-1, 1)
        pil_list = tensor_to_pil(t_batch)
        self.assertIsInstance(pil_list, list)
        self.assertEqual(len(pil_list), 3)
        for img in pil_list:
            self.assertEqual(img.size, (16, 16))

    def test_make_image_grid(self):
        imgs = [Image.new("RGB", (20, 20), color=(i * 40, 100, 200)) for i in range(4)]
        grid = make_image_grid(imgs, rows=2, cols=2, padding=2)
        # grid size = 2 * 20 + 3 * 2 = 46
        self.assertEqual(grid.size, (46, 46))

    def test_save_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "output.png"
            img = Image.new("RGB", (30, 30), color="blue")
            save_image(img, save_path)
            self.assertTrue(save_path.exists())
            with Image.open(save_path) as loaded:
                self.assertEqual(loaded.size, (30, 30))


class TestSampler(unittest.TestCase):
    """Test MultiscaleFlowSampler."""

    def setUp(self):
        # Create a lightweight model config for fast tests
        self.config = ScaleFlowConfig(
            vocab_size=100,
            text_dim=64,
            text_depth=2,
            text_heads=2,
            decoder_dim=32,
            decoder_depth=2,
        )
        self.model = ScaleFlowModel(self.config)
        self.sampler = MultiscaleFlowSampler(self.model)

    def test_sample_single_step(self):
        # 1 scale step: 8x8 -> 16x16
        prompt_embeds = torch.randn(1, 64)
        out = self.sampler.sample(prompt_embeds, steps=1)
        self.assertEqual(out.shape, (1, 3, 16, 16))
        self.assertTrue((out >= -1.0).all() and (out <= 1.0).all())

    def test_sample_three_steps(self):
        # 3 scale steps: 8 -> 16 -> 32 -> 64
        prompt_embeds = torch.randn(2, 64)
        out = self.sampler.sample(prompt_embeds, steps=3)
        self.assertEqual(out.shape, (2, 3, 64, 64))

    def test_sample_with_cfg(self):
        prompt_embeds = torch.randn(1, 64)
        neg_embeds = torch.randn(1, 64)
        out = self.sampler.sample(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=neg_embeds,
            guidance_scale=2.5,
            steps=2,
        )
        self.assertEqual(out.shape, (1, 3, 32, 32))

    def test_sample_return_pyramid(self):
        prompt_embeds = torch.randn(1, 64)
        pyramid = self.sampler.sample(prompt_embeds, steps=3, return_pyramid=True)
        self.assertEqual(len(pyramid), 4)  # [8, 16, 32, 64]
        self.assertEqual(pyramid[0].shape, (1, 3, 8, 8))
        self.assertEqual(pyramid[1].shape, (1, 3, 16, 16))
        self.assertEqual(pyramid[2].shape, (1, 3, 32, 32))
        self.assertEqual(pyramid[3].shape, (1, 3, 64, 64))

    def test_seed_reproducibility(self):
        prompt_embeds = torch.randn(1, 64)
        out1 = self.sampler.sample(prompt_embeds, steps=2, seed=42)
        out2 = self.sampler.sample(prompt_embeds, steps=2, seed=42)
        self.assertTrue(torch.allclose(out1, out2))


class TestScaleFlowPipeline(unittest.TestCase):
    """Test ScaleFlowPipeline end-to-end inference and serialization."""

    def setUp(self):
        self.config = ScaleFlowConfig(
            vocab_size=100,
            text_dim=64,
            text_depth=2,
            text_heads=2,
            decoder_dim=32,
            decoder_depth=2,
        )
        self.model = ScaleFlowModel(self.config)
        self.tokenizer = BPETokenizer()
        self.tokenizer.train(["a photo of a cat", "a futuristic city at sunset"], vocab_size=50)
        self.pipe = ScaleFlowPipeline(model=self.model, tokenizer=self.tokenizer)

    def test_pipeline_generate_pil(self):
        img = self.pipe("a photo of a cat", steps=2, seed=123)
        self.assertIsInstance(img, Image.Image)
        # 8 -> 16 -> 32
        self.assertEqual(img.size, (32, 32))

    def test_pipeline_batch_generate(self):
        prompts = ["a cat", "a futuristic city"]
        imgs = self.pipe(prompts, steps=1, seed=99)
        self.assertIsInstance(imgs, list)
        self.assertEqual(len(imgs), 2)
        for im in imgs:
            self.assertIsInstance(im, Image.Image)
            self.assertEqual(im.size, (16, 16))

    def test_pipeline_return_all_scales(self):
        pyramid_imgs = self.pipe("a cat", steps=3, return_all_scales=True, seed=1)
        self.assertIsInstance(pyramid_imgs, list)
        self.assertEqual(len(pyramid_imgs), 4)  # [8, 16, 32, 64]
        expected_sizes = [(8, 8), (16, 16), (32, 32), (64, 64)]
        for im, expected in zip(pyramid_imgs, expected_sizes):
            self.assertEqual(im.size, expected)

    def test_pipeline_device_movement(self):
        pipe = self.pipe.to("cpu")
        self.assertEqual(pipe.device.type, "cpu")

    def test_pipeline_cross_attention_enabled(self):
        # Configure model with active scale-aware cross-attention
        config = ScaleFlowConfig(
            vocab_size=100,
            text_dim=64,
            text_depth=2,
            text_heads=2,
            decoder_dim=32,
            decoder_depth=2,
            use_cross_attention=True,
            cross_attention_max_scale=2,
            cross_attention_heads=2,
        )
        model = ScaleFlowModel(config)
        pipe = ScaleFlowPipeline(model=model, tokenizer=self.tokenizer)

        # Generate image through cross-attention pipeline
        img = pipe("a futuristic city at sunset", steps=3, seed=42)
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (64, 64))

    def test_pipeline_cfg_with_cross_attention(self):
        # Test CFG with different length positive and negative prompts under cross-attention
        config = ScaleFlowConfig(
            vocab_size=100,
            text_dim=64,
            text_depth=2,
            text_heads=2,
            decoder_dim=32,
            decoder_depth=2,
            use_cross_attention=True,
            cross_attention_max_scale=2,
        )
        model = ScaleFlowModel(config)
        pipe = ScaleFlowPipeline(model=model, tokenizer=self.tokenizer)

        img = pipe(
            prompt="a futuristic city at sunset",
            negative_prompt="blurry",
            guidance_scale=2.0,
            steps=2,
            seed=101,
        )
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (32, 32))

    def test_save_and_from_pretrained(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir) / "test_checkpoint"
            self.pipe.save_pretrained(save_dir)

            self.assertTrue((save_dir / "config.json").exists())
            self.assertTrue((save_dir / "model.pt").exists())
            self.assertTrue((save_dir / "tokenizer.json").exists())

            loaded_pipe = ScaleFlowPipeline.from_pretrained(save_dir)
            self.assertEqual(loaded_pipe.model.config.text_dim, 64)
            self.assertEqual(loaded_pipe.tokenizer.vocab, self.tokenizer.vocab)

            # Generate from loaded pipeline
            img = loaded_pipe("a photo of a cat", steps=1, seed=42)
            self.assertIsInstance(img, Image.Image)
            self.assertEqual(img.size, (16, 16))


if __name__ == "__main__":
    unittest.main()
