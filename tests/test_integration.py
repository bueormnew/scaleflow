"""End-to-end integration test for the ScaleFlow library.

Verifies the complete lifecycle:
1. Tokenizer training and serialization.
2. Lightweight ScaleFlowModel initialization and parameter verification.
3. Multiscale OT-CFM dataset generation and DataLoader setup.
4. ScaleFlowTrainer execution for 1 dummy training step.
5. Checkpoint saving (both trainer checkpoint and pretrained pipeline checkpoint).
6. Checkpoint loading into ScaleFlowPipeline.
7. End-to-end inference generation producing valid PIL images across scales.
8. Rigorous image validity checks (channels, resolution, non-trivial pixel variance, valid value ranges).
"""

from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from PIL import Image

from scaleflow import (
    ScaleFlowModel,
    ScaleFlowConfig,
    ScaleFlowPipeline,
    BPETokenizer,
    ScaleFlowTrainer,
    TrainerConfig,
    MultiscaleOTFlowMatching,
    SyntheticImageTextDataset,
    save_image,
)
from scaleflow.training.dataset import create_dataloader


class TestScaleFlowEndToEndIntegration(unittest.TestCase):
    """Full lifecycle integration test suite for ScaleFlow."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.work_dir = Path(self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_full_training_to_inference_pipeline(self):
        """Execute end-to-end: train 1 dummy step, save, load, generate image, verify."""
        seed = 42
        torch.manual_seed(seed)

        # ---------------------------------------------------------
        # Step 1: Tokenizer Training
        # ---------------------------------------------------------
        prompts = [
            "a vibrant blue sphere on reflective floor",
            "golden spiral geometric art pattern",
            "minimalist architecture under cloudy sky",
            "neon glowing cube in dark room",
        ]
        tokenizer = BPETokenizer()
        tokenizer.train(prompts, vocab_size=64)
        self.assertGreaterEqual(len(tokenizer.vocab), 30)

        tokenizer_path = self.work_dir / "tokenizer.json"
        tokenizer.save(tokenizer_path)
        self.assertTrue(tokenizer_path.exists())

        loaded_tok = BPETokenizer.load(tokenizer_path)
        self.assertEqual(tokenizer.vocab, loaded_tok.vocab)

        # ---------------------------------------------------------
        # Step 2: Model Configuration & Instantiation
        # ---------------------------------------------------------
        config = ScaleFlowConfig(
            vocab_size=max(64, len(tokenizer.vocab)),
            text_dim=64,
            text_depth=2,
            text_heads=2,
            conv_kernel_size=3,
            max_seq_len=32,
            max_scale=3,  # Scales 0..3 (8x8 -> 16x16 -> 32x32 -> 64x64)
            decoder_dim=32,
            decoder_depth=2,
            decoder_mlp_ratio=2.0,
        )
        model = ScaleFlowModel(config)
        self.assertGreater(model.num_parameters(), 1000)

        # ---------------------------------------------------------
        # Step 3: Dataset & DataLoader
        # ---------------------------------------------------------
        dataset = SyntheticImageTextDataset(
            num_samples=4,
            image_size=64,
            tokenizer=tokenizer,
            max_seq_len=32,
            seed=seed,
        )
        self.assertEqual(len(dataset), 4)

        train_loader = create_dataloader(
            dataset=dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
        )
        batch = next(iter(train_loader))
        self.assertIn("images", batch)
        self.assertEqual(batch["images"].shape, (2, 3, 64, 64))

        # ---------------------------------------------------------
        # Step 4: Trainer Setup & Execute 1 Step
        # ---------------------------------------------------------
        ckpt_dir = self.work_dir / "checkpoints"
        trainer_config = TrainerConfig(
            learning_rate=1e-3,
            weight_decay=0.01,
            warmup_steps=0,
            max_steps=1,
            min_scale=0,
            max_scale=2,  # sample scales up to 32x32
            log_every_n_steps=1,
            checkpoint_dir=str(ckpt_dir),
            device="cpu",
            seed=seed,
        )
        flow_matcher = MultiscaleOTFlowMatching()
        trainer = ScaleFlowTrainer(
            model=model,
            flow_matcher=flow_matcher,
            train_dataloader=train_loader,
            config=trainer_config,
            tokenizer=tokenizer,
        )

        # Perform 1 dummy train step
        initial_params = [p.clone().detach() for p in model.parameters() if p.requires_grad]
        step_metrics = trainer.training_step(batch)

        self.assertIn("loss", step_metrics)
        self.assertTrue(np.isfinite(step_metrics["loss"]))
        self.assertGreater(step_metrics["loss"], 0.0)

        # Confirm weights were updated by optimizer step
        updated_params = [p.clone().detach() for p in model.parameters() if p.requires_grad]
        param_diffs = [
            torch.norm(p_init - p_up).item()
            for p_init, p_up in zip(initial_params, updated_params)
        ]
        self.assertTrue(any(diff > 0.0 for diff in param_diffs), "Model weights must update during train step.")

        # ---------------------------------------------------------
        # Step 5: Save Checkpoint
        # ---------------------------------------------------------
        # A. Save raw trainer state
        saved_ckpt = trainer.save_checkpoint(tag="step_1")
        self.assertTrue(saved_ckpt.exists())
        self.assertGreater(saved_ckpt.stat().st_size, 0)

        # B. Save complete pretrained pipeline format
        pretrained_dir = self.work_dir / "pretrained_pipeline"
        pipeline = ScaleFlowPipeline(model=model, tokenizer=tokenizer)
        pipeline.save_pretrained(pretrained_dir)

        self.assertTrue((pretrained_dir / "config.json").exists())
        self.assertTrue((pretrained_dir / "model.pt").exists())
        self.assertTrue((pretrained_dir / "tokenizer.json").exists())

        # ---------------------------------------------------------
        # Step 6: Load into ScaleFlowPipeline
        # ---------------------------------------------------------
        # Method A: from pretrained directory
        loaded_pipe = ScaleFlowPipeline.from_pretrained(pretrained_dir)
        self.assertEqual(loaded_pipe.model.config.text_dim, 64)
        self.assertEqual(loaded_pipe.model.config.vocab_size, config.vocab_size)
        self.assertEqual(len(loaded_pipe.tokenizer.vocab), len(tokenizer.vocab))

        # Method B: loading trainer state dict into fresh model
        fresh_model = ScaleFlowModel(config)
        raw_state = torch.load(saved_ckpt, map_location="cpu", weights_only=False)
        fresh_model.load_state_dict(raw_state["model_state_dict"])
        pipe_from_ckpt = ScaleFlowPipeline(model=fresh_model, tokenizer=tokenizer)

        # ---------------------------------------------------------
        # Step 7 & 8: Inference & Image Validity Checks
        # ---------------------------------------------------------
        test_prompt = "a vibrant blue sphere on reflective floor"

        for pipe_instance in (loaded_pipe, pipe_from_ckpt):
            # A. Generate single image (2 steps: 8x8 -> 16x16 -> 32x32)
            gen_img = pipe_instance(test_prompt, steps=2, seed=123)

            # Validity assertions
            self.assertIsInstance(gen_img, Image.Image)
            self.assertEqual(gen_img.size, (32, 32))
            self.assertEqual(gen_img.mode, "RGB")

            # Check pixel array properties
            arr = np.array(gen_img)
            self.assertEqual(arr.shape, (32, 32, 3))
            self.assertEqual(arr.dtype, np.uint8)
            self.assertFalse(np.isnan(arr).any(), "Generated image must not contain NaN.")
            self.assertGreater(arr.std(), 0.0, "Generated image should not be completely flat/uniform.")

            # Save and verify reloading
            img_file = self.work_dir / "test_out.png"
            save_image(gen_img, img_file)
            self.assertTrue(img_file.exists())
            with Image.open(img_file) as reloaded:
                self.assertEqual(reloaded.size, (32, 32))
                self.assertEqual(reloaded.mode, "RGB")

            # B. Generate full pyramid output
            pyramid = pipe_instance(test_prompt, steps=3, return_all_scales=True, seed=123)
            self.assertEqual(len(pyramid), 4)  # 8x8, 16x16, 32x32, 64x64
            expected_dims = [(8, 8), (16, 16), (32, 32), (64, 64)]
            for img, expected_dim in zip(pyramid, expected_dims):
                self.assertIsInstance(img, Image.Image)
                self.assertEqual(img.size, expected_dim)
                self.assertEqual(img.mode, "RGB")

            # C. Batch inference
            batch_prompts = [
                "golden spiral geometric art pattern",
                "neon glowing cube in dark room",
            ]
            batch_imgs = pipe_instance(batch_prompts, steps=1, seed=456)
            self.assertEqual(len(batch_imgs), 2)
            for im in batch_imgs:
                self.assertIsInstance(im, Image.Image)
                self.assertEqual(im.size, (16, 16))

            # D. Tensor output mode
            tensors = pipe_instance(test_prompt, steps=2, output_type="pt", seed=123)
            self.assertIsInstance(tensors, torch.Tensor)
            self.assertEqual(tensors.shape, (1, 3, 32, 32))
            self.assertFalse(torch.isnan(tensors).any())
            self.assertTrue((tensors >= -1.5).all() and (tensors <= 1.5).all())

    def test_trainer_train_loop_and_checkpoint_resume(self):
        """Verify trainer loop execution, checkpoint persistence, and seamless training resumption."""
        tokenizer = BPETokenizer()
        tokenizer.train(["synthetic circle", "synthetic triangle"], vocab_size=32)

        config = ScaleFlowConfig(
            vocab_size=32,
            text_dim=32,
            text_depth=1,
            text_heads=1,
            conv_kernel_size=3,
            max_seq_len=16,
            max_scale=2,
            decoder_dim=16,
            decoder_depth=1,
            decoder_mlp_ratio=1.5,
        )
        model = ScaleFlowModel(config)

        dataset = SyntheticImageTextDataset(
            num_samples=4,
            image_size=32,
            tokenizer=tokenizer,
            max_seq_len=16,
            seed=99,
        )
        train_loader = create_dataloader(dataset, batch_size=2, shuffle=False)

        ckpt_dir = self.work_dir / "loop_checkpoints"
        trainer_config = TrainerConfig(
            learning_rate=1e-3,
            warmup_steps=0,
            max_steps=2,
            min_scale=0,
            max_scale=1,
            log_every_n_steps=1,
            save_every_n_steps=2,
            checkpoint_dir=str(ckpt_dir),
            device="cpu",
            seed=99,
        )
        flow_matcher = MultiscaleOTFlowMatching()
        trainer = ScaleFlowTrainer(
            model=model,
            flow_matcher=flow_matcher,
            train_dataloader=train_loader,
            config=trainer_config,
            tokenizer=tokenizer,
        )

        # Run 2 training steps via train() loop
        history = trainer.train()
        self.assertEqual(trainer.global_step, 2)
        final_ckpt = ckpt_dir / "checkpoint_final.pt"
        self.assertTrue(final_ckpt.exists())

        # Resume in a second trainer
        resumed_model = ScaleFlowModel(config)
        trainer_resumed = ScaleFlowTrainer(
            model=resumed_model,
            flow_matcher=flow_matcher,
            train_dataloader=train_loader,
            config=trainer_config,
            tokenizer=tokenizer,
        )
        trainer_resumed.load_checkpoint(final_ckpt)
        self.assertEqual(trainer_resumed.global_step, 2)

        # Train 1 more step
        batch = next(iter(train_loader))
        step_metrics = trainer_resumed.training_step(batch)
        self.assertIn("loss", step_metrics)
        self.assertFalse(np.isnan(step_metrics["loss"]))

    def test_pipeline_guidance_and_generation_modes(self):
        """Verify CFG guidance, negative prompt modulation, and seed reproducibility."""
        tokenizer = BPETokenizer()
        tokenizer.train(["a serene landscape", "blurry low quality"], vocab_size=32)

        config = ScaleFlowConfig(
            vocab_size=32,
            text_dim=32,
            text_depth=1,
            text_heads=1,
            max_scale=2,
            decoder_dim=16,
            decoder_depth=1,
        )
        model = ScaleFlowModel(config)
        pipeline = ScaleFlowPipeline(model=model, tokenizer=tokenizer)

        # CFG test
        img_cfg = pipeline(
            prompt="a serene landscape",
            negative_prompt="blurry low quality",
            guidance_scale=2.0,
            steps=1,
            seed=42,
        )
        self.assertIsInstance(img_cfg, Image.Image)
        self.assertEqual(img_cfg.size, (16, 16))

        # Seed reproducibility
        img_a = pipeline("a serene landscape", steps=1, seed=1234)
        img_b = pipeline("a serene landscape", steps=1, seed=1234)
        self.assertTrue(np.array_equal(np.array(img_a), np.array(img_b)))


if __name__ == "__main__":
    unittest.main()
