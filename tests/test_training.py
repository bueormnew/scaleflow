"""Comprehensive unit tests for ScaleFlow training modules:
- flow_matching.py: Multiscale OT-CFM, pyramid generation, velocity target, OT trajectory.
- dataset.py: ScaleFlowDataset, SyntheticImageTextDataset, collator, DataLoader.
- losses.py: FlowMatchingLoss, MultiscaleFlowLoss, VelocityErrorMetric.
- trainer.py: ScaleFlowTrainer, optimizer setup, LR scheduling, checkpoint saving/loading.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image

from scaleflow.models.model import ScaleFlowConfig, ScaleFlowModel
from scaleflow.tokenizer.bpe import BPETokenizer
from scaleflow.training.flow_matching import (
    DEFAULT_SCALE_RESOLUTIONS,
    FlowMatchingBatch,
    MultiscaleOTFlowMatching,
    build_image_pyramid,
)
from scaleflow.training.dataset import (
    ScaleFlowDataset,
    SyntheticImageTextDataset,
    collate_multiscale,
    create_dataloader,
)
from scaleflow.training.losses import (
    FlowMatchingLoss,
    MultiscaleFlowLoss,
    VelocityErrorMetric,
)
from scaleflow.training.trainer import (
    ScaleFlowTrainer,
    TrainerConfig,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)


class TestLossesAndMetrics(unittest.TestCase):
    """Test loss functions and velocity error tracking metrics."""

    def setUp(self):
        self.B, self.C, self.H, self.W = 2, 3, 16, 16
        self.v_pred = torch.randn(self.B, self.C, self.H, self.W)
        self.v_target = torch.randn(self.B, self.C, self.H, self.W)

    def test_flow_matching_loss_types(self):
        """Test MSE, L1, and Huber loss computation."""
        mse_loss_fn = FlowMatchingLoss(loss_type="mse")
        l1_loss_fn = FlowMatchingLoss(loss_type="l1")
        huber_loss_fn = FlowMatchingLoss(loss_type="huber", huber_beta=0.1)

        l_mse = mse_loss_fn(self.v_pred, self.v_target)
        l_l1 = l1_loss_fn(self.v_pred, self.v_target)
        l_huber = huber_loss_fn(self.v_pred, self.v_target)

        self.assertEqual(l_mse.ndim, 0)
        self.assertEqual(l_l1.ndim, 0)
        self.assertEqual(l_huber.ndim, 0)

        self.assertGreater(l_mse.item(), 0.0)
        self.assertGreater(l_l1.item(), 0.0)
        self.assertGreater(l_huber.item(), 0.0)

        # Zero error should yield zero loss
        zero_loss = mse_loss_fn(self.v_target, self.v_target)
        self.assertAlmostEqual(zero_loss.item(), 0.0, places=6)

    def test_multiscale_flow_loss_weighting(self):
        """Test per-scale loss weighting."""
        scale_weights = {0: 1.0, 1: 2.0, 2: 0.5}
        ms_loss_fn = MultiscaleFlowLoss(scale_weights=scale_weights, base_loss_type="mse")

        loss_s0 = ms_loss_fn(self.v_pred, self.v_target, scale_level=0)
        loss_s1 = ms_loss_fn(self.v_pred, self.v_target, scale_level=1)

        # loss_s1 should be 2x loss_s0 because weight for 1 is 2.0 vs 1.0
        self.assertAlmostEqual(loss_s1.item() / loss_s0.item(), 2.0, places=4)

    def test_velocity_error_metric(self):
        """Test metric accumulation and computation (MSE, MAE, RMSE, cosine_sim)."""
        metric = VelocityErrorMetric()

        # Update batch
        res = metric.update(self.v_pred, self.v_target, scale_level=0)
        self.assertIn("mse", res)
        self.assertIn("mae", res)
        self.assertIn("rmse", res)
        self.assertIn("cosine_sim", res)

        # Aggregate computation
        agg = metric.compute()
        self.assertAlmostEqual(agg["mse"], res["mse"], places=4)
        self.assertIn("scale_0_mse", agg)

        # Verify perfect match gives cosine similarity 1.0 and 0.0 MSE
        metric.reset()
        res_perf = metric.update(self.v_target, self.v_target, scale_level=1)
        self.assertAlmostEqual(res_perf["mse"], 0.0, places=5)
        self.assertAlmostEqual(res_perf["cosine_sim"], 1.0, places=5)


class TestFlowMatchingPyramidAndTrajectory(unittest.TestCase):
    """Test multi-resolution pyramid generation and Optimal Transport flow matching."""

    def setUp(self):
        self.ot_fm = MultiscaleOTFlowMatching()

    def test_image_pyramid_generation(self):
        """Verify image pyramid generation from 8x8 up to 1024x1024."""
        img_1024 = torch.randn(1, 3, 1024, 1024)
        pyramid = build_image_pyramid(img_1024)

        expected_resolutions = [8, 16, 32, 64, 128, 256, 512, 1024]
        for s in range(8):
            self.assertIn(s, pyramid)
            res = expected_resolutions[s]
            self.assertEqual(pyramid[s].shape, (1, 3, res, res))

    def test_custom_pyramid_resolutions(self):
        """Test pyramid generation with custom scale dict."""
        custom_scales = {0: 16, 1: 32, 2: 64}
        img_64 = torch.randn(2, 3, 64, 64)
        pyramid = build_image_pyramid(img_64, resolutions=custom_scales)

        self.assertEqual(len(pyramid), 3)
        self.assertEqual(pyramid[0].shape, (2, 3, 16, 16))
        self.assertEqual(pyramid[1].shape, (2, 3, 32, 32))
        self.assertEqual(pyramid[2].shape, (2, 3, 64, 64))

    def test_sample_scale_and_time(self):
        """Verify scale and time sampling bounds."""
        s = self.ot_fm.sample_scale(batch_size=4, min_scale=0, max_scale=7, per_batch=True)
        self.assertIsInstance(s, int)
        self.assertTrue(0 <= s <= 7)

        s_vec = self.ot_fm.sample_scale(batch_size=4, min_scale=1, max_scale=5, per_batch=False)
        self.assertEqual(s_vec.shape, (4,))
        self.assertTrue(torch.all((s_vec >= 1) & (s_vec <= 5)))

        t = self.ot_fm.sample_time(batch_size=8)
        self.assertEqual(t.shape, (8,))
        self.assertTrue(torch.all((t >= 0.0) & (t <= 1.0)))

    def test_ot_trajectory_math(self):
        """Verify OT flow trajectory formulas x_t = (1 - t) x_0 + t x_1 and v_t = x_1 - x_0."""
        fm = MultiscaleOTFlowMatching(sigma_min=0.0)
        B, C, H, W = 2, 3, 16, 16
        x_0 = torch.zeros(B, C, H, W)
        x_1 = torch.ones(B, C, H, W) * 2.0

        # At t = 0
        t_0 = torch.zeros(B)
        x_t0, _, v_t0 = fm.compute_ot_trajectory(x_0, x_1, t=t_0)
        self.assertTrue(torch.allclose(x_t0, x_0))
        self.assertTrue(torch.allclose(v_t0, torch.ones_like(x_0) * 2.0))

        # At t = 1
        t_1 = torch.ones(B)
        x_t1, _, v_t1 = fm.compute_ot_trajectory(x_0, x_1, t=t_1)
        self.assertTrue(torch.allclose(x_t1, x_1))
        self.assertTrue(torch.allclose(v_t1, torch.ones_like(x_0) * 2.0))

        # At t = 0.5
        t_half = torch.ones(B) * 0.5
        x_thalf, _, _ = fm.compute_ot_trajectory(x_0, x_1, t=t_half)
        self.assertTrue(torch.allclose(x_thalf, torch.ones(B, C, H, W)))

    def test_prepare_flow_batch(self):
        """Verify prepare_flow_batch produces all necessary tensors with exact shapes."""
        img_32 = torch.randn(2, 3, 32, 32)
        pyramid = build_image_pyramid(img_32, resolutions={0: 8, 1: 16, 2: 32})

        # Test scale 0 (base scale)
        batch_s0 = self.ot_fm.prepare_flow_batch(pyramid, scale_level=0)
        self.assertEqual(batch_s0.scale, 0)
        self.assertEqual(batch_s0.x_1.shape, (2, 3, 8, 8))
        self.assertEqual(batch_s0.x_0.shape, (2, 3, 8, 8))
        self.assertEqual(batch_s0.x_t.shape, (2, 3, 8, 8))
        self.assertEqual(batch_s0.velocity_target.shape, (2, 3, 8, 8))
        self.assertEqual(batch_s0.t.shape, (2,))

        # Test scale 1 (intermediate scale with previous scale)
        batch_s1 = self.ot_fm.prepare_flow_batch(pyramid, scale_level=1)
        self.assertEqual(batch_s1.scale, 1)
        self.assertEqual(batch_s1.x_prev.shape, (2, 3, 8, 8))
        self.assertEqual(batch_s1.x_1.shape, (2, 3, 16, 16))
        self.assertEqual(batch_s1.x_0.shape, (2, 3, 16, 16))
        self.assertEqual(batch_s1.velocity_target.shape, (2, 3, 16, 16))


class TestDatasetAndDataLoader(unittest.TestCase):
    """Test ScaleFlowDataset, synthetic dataset, disk loading, and DataLoader helper."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.image_dir = Path(self.temp_dir) / "images"
        self.image_dir.mkdir(parents=True, exist_ok=True)

        # Create 3 dummy images and text files
        for i in range(3):
            img = Image.new("RGB", (64, 64), color=(i * 50, 100, 150))
            img_path = self.image_dir / f"sample_{i:02d}.png"
            img.save(img_path)
            txt_path = self.image_dir / f"sample_{i:02d}.txt"
            txt_path.write_text(f"caption for sample {i}", encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_synthetic_dataset(self):
        """Test SyntheticImageTextDataset procedural generation."""
        dataset = SyntheticImageTextDataset(num_samples=10, image_size=32, return_pyramid=True)
        self.assertEqual(len(dataset), 10)

        item = dataset[0]
        self.assertIn("image", item)
        self.assertIn("caption", item)
        self.assertIn("pyramid", item)
        self.assertEqual(item["image"].shape, (3, 32, 32))
        self.assertIsInstance(item["caption"], str)
        self.assertGreater(len(item["caption"]), 0)

        # Check values normalized to [-1, 1]
        self.assertTrue(item["image"].min() >= -1.0)
        self.assertTrue(item["image"].max() <= 1.0)

    def test_disk_dataset_paired_txt(self):
        """Test loading dataset from directory with paired image + txt files."""
        dataset = ScaleFlowDataset(image_dir=self.image_dir, image_size=32)
        self.assertEqual(len(dataset), 3)

        item = dataset[0]
        self.assertEqual(item["image"].shape, (3, 32, 32))
        self.assertTrue(item["caption"].startswith("caption for sample"))

    def test_disk_dataset_json_metadata(self):
        """Test loading dataset from metadata JSON file."""
        json_file = Path(self.temp_dir) / "metadata.json"
        meta = [
            {"image": str(self.image_dir / "sample_00.png"), "caption": "json caption 0"},
            {"image": str(self.image_dir / "sample_01.png"), "caption": "json caption 1"},
        ]
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(meta, f)

        dataset = ScaleFlowDataset(captions_file=json_file, image_size=32)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0]["caption"], "json caption 0")

    def test_tokenizer_integration(self):
        """Test dataset integration with BPETokenizer."""
        tokenizer = BPETokenizer()
        sample_texts = ["a red circle on a black background", "a blue square on white"]
        tokenizer.train(sample_texts, vocab_size=32)

        dataset = SyntheticImageTextDataset(
            num_samples=4,
            image_size=32,
            tokenizer=tokenizer,
            max_seq_len=16,
        )
        item = dataset[0]
        self.assertIn("input_ids", item)
        self.assertIn("attention_mask", item)
        self.assertEqual(item["input_ids"].shape, (16,))
        self.assertEqual(item["attention_mask"].shape, (16,))

    def test_dataloader_creation_and_batching(self):
        """Test create_dataloader helper and collator."""
        dataset = SyntheticImageTextDataset(num_samples=6, image_size=32)
        dataloader = create_dataloader(dataset, batch_size=2, shuffle=False)

        batch = next(iter(dataloader))
        self.assertIn("images", batch)
        self.assertIn("captions", batch)
        self.assertEqual(batch["images"].shape, (2, 3, 32, 32))
        self.assertEqual(len(batch["captions"]), 2)


class TestScaleFlowTrainer(unittest.TestCase):
    """Test ScaleFlowTrainer: optimization, scheduling, gradient clipping, checkpointing, and overfitting."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = ScaleFlowConfig(
            vocab_size=64,
            text_dim=32,
            text_depth=2,
            text_heads=2,
            conv_kernel_size=3,
            conv_expansion=2,
            max_seq_len=16,
            decoder_dim=16,
            decoder_depth=2,
            in_channels=3,
            out_channels=3,
        )
        self.model = ScaleFlowModel(config=self.config)

        # Scale resolutions for fast testing: 8x8, 16x16, 32x32
        self.test_scales = {0: 8, 1: 16, 2: 32}
        self.flow_matcher = MultiscaleOTFlowMatching(scale_resolutions=self.test_scales)

        # Toy dataset
        self.dataset = SyntheticImageTextDataset(
            num_samples=8,
            image_size=32,
            max_seq_len=16,
        )
        # Dummy token IDs for tests
        for s in self.dataset.samples:
            s["input_ids"] = torch.randint(1, 60, (16,))
            s["attention_mask"] = torch.ones(16, dtype=torch.long)

        self.train_loader = create_dataloader(self.dataset, batch_size=2)
        self.val_loader = create_dataloader(self.dataset, batch_size=2)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_trainer_single_step(self):
        """Verify trainer executes a training step with proper metric returns."""
        t_config = TrainerConfig(
            max_steps=5,
            learning_rate=1e-3,
            checkpoint_dir=self.temp_dir,
            min_scale=0,
            max_scale=2,
            prediction_mode="direct",
        )
        trainer = ScaleFlowTrainer(
            model=self.model,
            flow_matcher=self.flow_matcher,
            train_dataloader=self.train_loader,
            val_dataloader=self.val_loader,
            config=t_config,
        )

        batch = next(iter(self.train_loader))
        step_metrics = trainer.training_step(batch)

        self.assertIn("loss", step_metrics)
        self.assertIn("lr", step_metrics)
        self.assertIn("mse", step_metrics)
        self.assertIn("cosine_sim", step_metrics)
        self.assertFalse(torch.isnan(torch.tensor(step_metrics["loss"])))

    def test_trainer_upsample_prediction_mode(self):
        """Verify trainer functions in prediction_mode='upsample'."""
        t_config = TrainerConfig(
            max_steps=2,
            learning_rate=1e-3,
            checkpoint_dir=self.temp_dir,
            min_scale=0,
            max_scale=1,
            prediction_mode="upsample",
        )
        trainer = ScaleFlowTrainer(
            model=self.model,
            flow_matcher=self.flow_matcher,
            train_dataloader=self.train_loader,
            config=t_config,
        )

        batch = next(iter(self.train_loader))
        step_metrics = trainer.training_step(batch)
        self.assertIn("loss", step_metrics)
        self.assertGreater(step_metrics["loss"], 0.0)

    def test_checkpoint_save_and_load(self):
        """Verify full state restoration from checkpoint."""
        t_config = TrainerConfig(
            max_steps=10,
            learning_rate=1e-3,
            checkpoint_dir=self.temp_dir,
            min_scale=0,
            max_scale=2,
        )
        trainer = ScaleFlowTrainer(
            model=self.model,
            flow_matcher=self.flow_matcher,
            train_dataloader=self.train_loader,
            config=t_config,
        )

        batch = next(iter(self.train_loader))
        trainer.training_step(batch)
        trainer.global_step = 42
        trainer.current_epoch = 3
        trainer.best_val_loss = 0.123

        # Save checkpoint
        ckpt_path = trainer.save_checkpoint(tag="unit_test")
        self.assertTrue(ckpt_path.exists())

        # Create fresh trainer and restore
        new_model = ScaleFlowModel(config=self.config)
        new_trainer = ScaleFlowTrainer(
            model=new_model,
            flow_matcher=self.flow_matcher,
            train_dataloader=self.train_loader,
            config=t_config,
        )
        new_trainer.load_checkpoint(ckpt_path)

        self.assertEqual(new_trainer.global_step, 42)
        self.assertEqual(new_trainer.current_epoch, 3)
        self.assertAlmostEqual(new_trainer.best_val_loss, 0.123, places=4)

        # Check parameter equality
        for (n1, p1), (n2, p2) in zip(self.model.named_parameters(), new_model.named_parameters()):
            self.assertEqual(n1, n2)
            self.assertTrue(torch.allclose(p1, p2), f"Parameter {n1} not equal after load")

    def test_validation_step(self):
        """Verify validation step computes metrics across val_dataloader."""
        t_config = TrainerConfig(
            checkpoint_dir=self.temp_dir,
            min_scale=0,
            max_scale=2,
        )
        trainer = ScaleFlowTrainer(
            model=self.model,
            flow_matcher=self.flow_matcher,
            train_dataloader=self.train_loader,
            val_dataloader=self.val_loader,
            config=t_config,
        )

        val_metrics = trainer.validation_step()
        self.assertIn("val_loss", val_metrics)
        self.assertIn("mse", val_metrics)
        self.assertIn("cosine_sim", val_metrics)
        self.assertGreater(val_metrics["val_loss"], 0.0)

    def test_toy_training_convergence(self):
        """Overfit single batch for 25 steps to verify gradient updates and loss reduction."""
        torch.manual_seed(123)
        t_config = TrainerConfig(
            max_steps=25,
            learning_rate=5e-3,
            checkpoint_dir=self.temp_dir,
            min_scale=1,
            max_scale=1,
            warmup_steps=0,
            prediction_mode="direct",
        )
        trainer = ScaleFlowTrainer(
            model=self.model,
            flow_matcher=self.flow_matcher,
            train_dataloader=self.train_loader,
            config=t_config,
        )

        # Fixed single batch
        batch = next(iter(self.train_loader))

        losses = []
        for _ in range(25):
            res = trainer.training_step(batch)
            losses.append(res["loss"])

        # Final loss should be lower than initial loss
        initial_loss = losses[0]
        final_loss = losses[-1]
        self.assertLess(
            final_loss,
            initial_loss,
            f"Expected loss to decrease: initial {initial_loss:.4f} vs final {final_loss:.4f}",
        )

    def test_device_selection_and_gpu_config(self):
        """Verify device selection logic, config options, and graceful fallbacks."""
        from scaleflow.training.trainer import resolve_device, get_device_info

        # 1. Test explicit device resolution
        cpu_dev = resolve_device("cpu")
        self.assertEqual(cpu_dev.type, "cpu")

        auto_dev = resolve_device(None)
        self.assertIn(auto_dev.type, ("cuda", "mps", "cpu"))

        # 2. Test device info reporting
        info = get_device_info(cpu_dev)
        self.assertIn("device_type", info)
        self.assertEqual(info["device_type"], "cpu")

        # 3. Test TrainerConfig device, AMP, and DataParallel defaults
        cfg = TrainerConfig(device="cpu", mixed_precision=True, dataparallel=True)
        self.assertEqual(cfg.device, "cpu")
        self.assertTrue(cfg.mixed_precision)
        self.assertTrue(cfg.dataparallel)

        # 4. Instantiate trainer on CPU: mixed_precision and dataparallel should degrade gracefully
        trainer = ScaleFlowTrainer(
            model=self.model,
            flow_matcher=self.flow_matcher,
            train_dataloader=self.train_loader,
            config=cfg,
        )
        self.assertEqual(trainer.device.type, "cpu")
        self.assertFalse(trainer._amp_enabled)  # disabled on CPU safely
        self.assertIsNone(trainer._scaler)


if __name__ == "__main__":
    unittest.main()
