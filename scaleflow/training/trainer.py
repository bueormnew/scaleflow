"""ScaleFlowTrainer for training Multiscale Optimal Transport Flow Matching models.

Features:
- AdamW optimizer configuration with weight decay separation.
- Learning rate schedulers: Cosine with warmup, linear warmup, constant with warmup.
- Gradient clipping (clip_grad_norm_).
- Training and validation metrics tracking (MSE, MAE, RMSE, cosine similarity, per-scale loss).
- Comprehensive checkpoint saving and loading (weights, optimizer, scheduler, RNG states).
- Multi-scale training supporting direct velocity prediction or recursive upsample prediction.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import json
import math
import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from scaleflow.training.flow_matching import MultiscaleOTFlowMatching, FlowMatchingBatch
from scaleflow.training.losses import FlowMatchingLoss, MultiscaleFlowLoss, VelocityErrorMetric


@dataclass
class TrainerConfig:
    """Configuration dataclass for ScaleFlowTrainer."""
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    adam_eps: float = 1e-8
    gradient_clip_val: float = 1.0

    # LR schedule
    lr_scheduler_type: str = "cosine"  # "cosine", "linear", "constant"
    warmup_steps: int = 50
    min_lr_ratio: float = 0.01

    # Training duration
    max_steps: int = 1000
    max_epochs: Optional[int] = None

    # Flow matching settings
    prediction_mode: str = "direct"  # "direct" (same resolution) or "upsample" (2x resolution)
    min_scale: int = 0
    max_scale: int = 7

    # Logging and evaluation
    log_every_n_steps: int = 10
    eval_every_n_steps: int = 100
    save_every_n_steps: int = 250
    checkpoint_dir: str = "checkpoints"

    # Hardware & Precision
    device: Optional[str] = None  # None for auto-detect (cuda, cpu)
    mixed_precision: bool = False
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainerConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """Create a schedule with linear warmup and cosine decay."""
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        progress = min(1.0, max(0.0, progress))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def get_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """Create a schedule with linear warmup and linear decay."""
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        progress = min(1.0, max(0.0, progress))
        decay = 1.0 - progress
        return min_lr_ratio + (1.0 - min_lr_ratio) * decay

    return LambdaLR(optimizer, lr_lambda)


class ScaleFlowTrainer:
    """Comprehensive trainer for ScaleFlow models."""

    def __init__(
        self,
        model: nn.Module,
        flow_matcher: MultiscaleOTFlowMatching,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        config: Optional[TrainerConfig] = None,
        tokenizer: Optional[Any] = None,
    ) -> None:
        """Initialize ScaleFlowTrainer.
        
        Args:
            model: PyTorch ScaleFlowModel (or compatible module).
            flow_matcher: MultiscaleOTFlowMatching instance.
            train_dataloader: Training DataLoader.
            val_dataloader: Optional validation DataLoader.
            optimizer: Optional pre-configured optimizer.
            lr_scheduler: Optional pre-configured learning rate scheduler.
            config: TrainerConfig instance.
            tokenizer: Optional tokenizer for string prompts.
        """
        self.config = config or TrainerConfig()
        self.model = model
        self.flow_matcher = flow_matcher
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.tokenizer = tokenizer

        # Determine compute device
        if self.config.device is not None:
            self.device = torch.device(self.config.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        # Setup Optimizer
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = self._build_optimizer()

        # Setup LR Scheduler
        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler
        else:
            self.lr_scheduler = self._build_lr_scheduler()

        # Metrics tracking
        self.train_metric = VelocityErrorMetric()
        self.val_metric = VelocityErrorMetric()
        self.metrics_history: List[Dict[str, Any]] = []

        # Training state
        self.global_step: int = 0
        self.current_epoch: int = 0
        self.best_val_loss: float = float("inf")

        # Checkpoint directory
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build AdamW optimizer separating 1D/norm/bias weights from decay."""
        decay_params = []
        no_decay_params = []
        no_decay_names = ("bias", "norm", "gamma", "beta", "ada_proj")

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name.lower() for nd in no_decay_names) or param.ndim <= 1:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        return AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_eps,
        )

    def _build_lr_scheduler(self) -> Optional[Any]:
        """Build learning rate scheduler."""
        sched_type = self.config.lr_scheduler_type.lower()
        if sched_type == "cosine":
            return get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=self.config.max_steps,
                min_lr_ratio=self.config.min_lr_ratio,
            )
        elif sched_type == "linear":
            return get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=self.config.max_steps,
                min_lr_ratio=self.config.min_lr_ratio,
            )
        elif sched_type == "constant":
            return None
        return None

    def training_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Execute a single forward-backward training step.
        
        Args:
            batch: Batch dictionary from DataLoader.
            
        Returns:
            Dictionary with step metrics ('loss', 'lr', 'mse', 'mae', 'cosine_sim', 'scale').
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Extract images and prepare multi-scale pyramid
        images = batch["images"].to(self.device)
        B = images.shape[0]

        if "pyramids" in batch:
            pyramid = {s: t.to(self.device) for s, t in batch["pyramids"].items()}
        else:
            pyramid = self.flow_matcher.build_pyramid(images)

        # Sample scale level
        scale = self.flow_matcher.sample_scale(
            batch_size=B,
            min_scale=self.config.min_scale,
            max_scale=self.config.max_scale,
            per_batch=True,
        )

        flow_batch = self.flow_matcher.prepare_flow_batch(pyramid=pyramid, scale_level=scale)

        # Text conditioning
        input_ids = batch.get("input_ids")
        if input_ids is not None:
            input_ids = input_ids.to(self.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        e_text = batch.get("e_text")
        if e_text is not None:
            e_text = e_text.to(self.device)

        if input_ids is None and e_text is None:
            if "captions" in batch and self.tokenizer is not None:
                encoded = self.tokenizer(batch["captions"], padding=True, return_tensors="pt")
                input_ids = encoded["input_ids"].to(self.device)
                if "attention_mask" in encoded:
                    attention_mask = encoded["attention_mask"].to(self.device)
            elif hasattr(self.model, "config") and hasattr(self.model.config, "text_dim"):
                e_text = torch.zeros(B, self.model.config.text_dim, device=self.device)
            else:
                input_ids = torch.zeros((B, 1), dtype=torch.long, device=self.device)

        # Forward pass through model based on prediction_mode
        if self.config.prediction_mode == "upsample":
            # Model takes previous scale (resolution r_prev) and outputs at 2x resolution (r_curr)
            v_pred = self.model(
                flow_batch.x_prev,
                scale=scale,
                input_ids=input_ids,
                attention_mask=attention_mask,
                e_text=e_text,
                upsample=True,
            )
        else:
            # Direct mode: Model takes noisy state x_t at resolution r_curr and predicts v_t
            v_pred = self.model(
                flow_batch.x_t,
                scale=scale,
                input_ids=input_ids,
                attention_mask=attention_mask,
                e_text=e_text,
                upsample=False,
            )

        # Compute regression loss
        loss = self.flow_matcher.compute_loss(v_pred, flow_batch.velocity_target)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if self.config.gradient_clip_val > 0.0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.gradient_clip_val,
            )

        # Optimizer & scheduler steps
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # Update and collect metrics
        metric_res = self.train_metric.update(
            v_pred=v_pred,
            v_target=flow_batch.velocity_target,
            scale_level=scale,
        )

        current_lr = self.optimizer.param_groups[0]["lr"]

        step_output = {
            "loss": loss.item(),
            "lr": current_lr,
            "scale": scale,
            "mse": metric_res["mse"],
            "mae": metric_res["mae"],
            "cosine_sim": metric_res["cosine_sim"],
        }

        return step_output

    @torch.no_grad()
    def validation_step(self) -> Dict[str, float]:
        """Execute validation loop over val_dataloader.
        
        Returns:
            Dictionary with aggregated validation metrics.
        """
        if self.val_dataloader is None:
            return {}

        self.model.eval()
        self.val_metric.reset()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_dataloader:
            images = batch["images"].to(self.device)
            B = images.shape[0]

            if "pyramids" in batch:
                pyramid = {s: t.to(self.device) for s, t in batch["pyramids"].items()}
            else:
                pyramid = self.flow_matcher.build_pyramid(images)

            scale = self.flow_matcher.sample_scale(
                batch_size=B,
                min_scale=self.config.min_scale,
                max_scale=self.config.max_scale,
                per_batch=True,
            )

            flow_batch = self.flow_matcher.prepare_flow_batch(pyramid=pyramid, scale_level=scale)

            input_ids = batch.get("input_ids")
            if input_ids is not None:
                input_ids = input_ids.to(self.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            e_text = batch.get("e_text")
            if e_text is not None:
                e_text = e_text.to(self.device)

            if input_ids is None and e_text is None:
                if "captions" in batch and self.tokenizer is not None:
                    encoded = self.tokenizer(batch["captions"], padding=True, return_tensors="pt")
                    input_ids = encoded["input_ids"].to(self.device)
                    if "attention_mask" in encoded:
                        attention_mask = encoded["attention_mask"].to(self.device)
                elif hasattr(self.model, "config") and hasattr(self.model.config, "text_dim"):
                    e_text = torch.zeros(B, self.model.config.text_dim, device=self.device)
                else:
                    input_ids = torch.zeros((B, 1), dtype=torch.long, device=self.device)

            if self.config.prediction_mode == "upsample":
                v_pred = self.model(
                    flow_batch.x_prev,
                    scale=scale,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    e_text=e_text,
                    upsample=True,
                )
            else:
                v_pred = self.model(
                    flow_batch.x_t,
                    scale=scale,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    e_text=e_text,
                    upsample=False,
                )

            loss = self.flow_matcher.compute_loss(v_pred, flow_batch.velocity_target)
            total_loss += loss.item()
            num_batches += 1

            self.val_metric.update(
                v_pred=v_pred,
                v_target=flow_batch.velocity_target,
                scale_level=scale,
            )

        val_summary = self.val_metric.compute()
        val_summary["val_loss"] = total_loss / max(num_batches, 1)

        return val_summary

    def save_checkpoint(
        self,
        path: Optional[Union[str, Path]] = None,
        tag: Optional[str] = None,
    ) -> Path:
        """Save a complete training checkpoint.
        
        Args:
            path: Optional full filepath to save checkpoint to.
            tag: Optional name tag (e.g. 'best', 'step_500').
            
        Returns:
            Path of saved checkpoint file.
        """
        if path is not None:
            save_path = Path(path)
        else:
            filename = f"checkpoint_{tag or f'step_{self.global_step}'}.pt"
            save_path = self.checkpoint_dir / filename

        save_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "best_val_loss": self.best_val_loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "torch_rng_state": torch.get_rng_state(),
        }

        if self.lr_scheduler is not None:
            state["scheduler_state_dict"] = self.lr_scheduler.state_dict()

        torch.save(state, save_path)
        return save_path

    def load_checkpoint(
        self,
        path: Union[str, Path],
        load_optimizer: bool = True,
        load_scheduler: bool = True,
    ) -> Dict[str, Any]:
        """Load training state from a checkpoint.
        
        Args:
            path: Path to checkpoint file.
            load_optimizer: Whether to restore optimizer state.
            load_scheduler: Whether to restore scheduler state.
            
        Returns:
            Loaded checkpoint state dictionary.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if load_scheduler and self.lr_scheduler and "scheduler_state_dict" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.global_step = checkpoint.get("global_step", 0)
        self.current_epoch = checkpoint.get("current_epoch", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        return checkpoint

    def train(
        self,
        max_steps: Optional[int] = None,
        max_epochs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Main training loop.
        
        Args:
            max_steps: Total steps to train for (defaults to config.max_steps).
            max_epochs: Total epochs to train for (defaults to config.max_epochs).
            
        Returns:
            Dictionary summary with total steps, epochs, and metrics history.
        """
        total_steps = max_steps or self.config.max_steps
        max_ep = max_epochs or self.config.max_epochs or 1000000

        self.train_metric.reset()

        while self.global_step < total_steps and self.current_epoch < max_ep:
            for batch in self.train_dataloader:
                step_metrics = self.training_step(batch)
                self.global_step += 1

                step_metrics["step"] = self.global_step
                step_metrics["epoch"] = self.current_epoch

                # Logging
                if self.global_step % self.config.log_every_n_steps == 0 or self.global_step == 1:
                    self.metrics_history.append(step_metrics)

                # Validation
                if (
                    self.val_dataloader is not None
                    and self.global_step % self.config.eval_every_n_steps == 0
                ):
                    val_metrics = self.validation_step()
                    step_metrics.update(val_metrics)
                    if val_metrics.get("val_loss", float("inf")) < self.best_val_loss:
                        self.best_val_loss = val_metrics["val_loss"]
                        self.save_checkpoint(tag="best")

                # Regular checkpoint saving
                if self.global_step % self.config.save_every_n_steps == 0:
                    self.save_checkpoint()

                if self.global_step >= total_steps:
                    break

            self.current_epoch += 1

        # Save final checkpoint
        final_path = self.save_checkpoint(tag="final")

        return {
            "global_step": self.global_step,
            "epochs": self.current_epoch,
            "final_checkpoint": str(final_path),
            "history": self.metrics_history,
        }
