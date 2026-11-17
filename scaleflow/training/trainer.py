"""ScaleFlowTrainer for training Multiscale Optimal Transport Flow Matching models.

Features:
- AdamW optimizer configuration with weight decay separation.
- Learning rate schedulers: Cosine with warmup, linear warmup, constant with warmup.
- Gradient clipping (clip_grad_norm_).
- Training and validation metrics tracking (MSE, MAE, RMSE, cosine similarity, per-scale loss).
- Comprehensive checkpoint saving and loading (weights, optimizer, scheduler, RNG states).
- Multi-scale training supporting direct velocity prediction or recursive upsample prediction.
- Native GPU support with Automatic Mixed Precision (AMP) via torch.cuda.amp.
- Multi-GPU parallelism via torch.nn.DataParallel (single-node multi-GPU).
- Automatic device selection: CUDA > MPS (Apple Silicon) > CPU.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import json
import math
import os
import time
import warnings
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from scaleflow.training.flow_matching import MultiscaleOTFlowMatching, FlowMatchingBatch
from scaleflow.training.losses import FlowMatchingLoss, MultiscaleFlowLoss, VelocityErrorMetric


# ---------------------------------------------------------------------------
# Device resolution helper (no architecture changes)
# ---------------------------------------------------------------------------

def resolve_device(device: Optional[str] = None) -> torch.device:
    """Resolve the best available compute device.

    Priority: explicit request > CUDA > MPS (Apple Silicon) > CPU.

    Args:
        device: Optional explicit device string ('cuda', 'cuda:0', 'mps', 'cpu', etc.).
                Pass None for automatic selection.

    Returns:
        torch.device instance ready for use.
    """
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS support (Apple M-series)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_info(device: torch.device) -> Dict[str, Any]:
    """Return a human-readable summary of the selected device.

    Args:
        device: The resolved torch.device.

    Returns:
        Dict with 'device_type', 'device_name', 'cuda_available', 'gpu_count',
        'mps_available', and 'cuda_version'.
    """
    info: Dict[str, Any] = {
        "device_type": device.type,
        "device_name": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "mps_available": (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    }
    if torch.cuda.is_available() and device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        info["device_name"] = torch.cuda.get_device_name(idx)
        info["total_memory_gb"] = round(
            torch.cuda.get_device_properties(idx).total_memory / 1e9, 2
        )
    return info


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

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
    prediction_mode: str = "direct"  # "direct" or "upsample"
    min_scale: int = 0
    max_scale: int = 7

    # Logging and evaluation
    log_every_n_steps: int = 10
    eval_every_n_steps: int = 100
    save_every_n_steps: int = 250
    checkpoint_dir: str = "checkpoints"

    # Hardware & Precision
    device: Optional[str] = None       # None -> auto-detect (cuda > mps > cpu)
    mixed_precision: bool = False       # AMP float16 on CUDA; silently ignored on CPU/MPS
    dataparallel: bool = False          # Wrap model in nn.DataParallel for multi-GPU
    dataparallel_device_ids: Optional[List[int]] = None  # e.g. [0, 1, 2, 3]
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainerConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# LR Schedulers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class ScaleFlowTrainer:
    """Comprehensive trainer for ScaleFlow models with full GPU/CPU support.

    Key GPU features (all opt-in via TrainerConfig, zero architecture changes):
    - Automatic device selection: CUDA > MPS > CPU.
    - Automatic Mixed Precision (AMP) via torch.cuda.amp.GradScaler for 2-3x
      throughput on CUDA with float16 (mixed_precision=True).
    - Multi-GPU DataParallel via torch.nn.DataParallel for single-node
      multi-GPU training (dataparallel=True).
    - non_blocking tensor transfers for async host-to-GPU copies.
    """

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
            model: PyTorch ScaleFlowModel (or any nn.Module compatible with the
                   ScaleFlow forward signature). Architecture is NEVER modified.
            flow_matcher: MultiscaleOTFlowMatching instance.
            train_dataloader: Training DataLoader.
            val_dataloader: Optional validation DataLoader.
            optimizer: Optional pre-configured optimizer (skips _build_optimizer).
            lr_scheduler: Optional pre-configured LR scheduler.
            config: TrainerConfig instance. Defaults to TrainerConfig() if None.
            tokenizer: Optional tokenizer for string captions.
        """
        self.config = config or TrainerConfig()
        self.tokenizer = tokenizer

        # ------------------------------------------------------------------
        # 1. Resolve device (CUDA > MPS > CPU, or explicit)
        # ------------------------------------------------------------------
        self.device = resolve_device(self.config.device)
        self._device_info = get_device_info(self.device)

        # ------------------------------------------------------------------
        # 2. Model -> device, optional DataParallel (multi-GPU)
        #    _raw_model always points to the unwrapped model for param access.
        # ------------------------------------------------------------------
        self._raw_model = model  # unwrapped reference (never modified)
        model = model.to(self.device)

        if self.config.dataparallel and self.device.type == "cuda":
            n_gpus = torch.cuda.device_count()
            if n_gpus > 1:
                device_ids = self.config.dataparallel_device_ids
                if device_ids is not None:
                    device_ids = [d for d in device_ids if d < n_gpus]
                self.model: nn.Module = nn.DataParallel(model, device_ids=device_ids)
                print(
                    f"[ScaleFlowTrainer] DataParallel enabled on "
                    f"{len(self.model.device_ids)} GPU(s): {self.model.device_ids}"
                )
            else:
                warnings.warn(
                    "dataparallel=True but only 1 GPU detected. Running single-GPU.",
                    UserWarning,
                )
                self.model = model
        else:
            self.model = model

        self.flow_matcher = flow_matcher
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # ------------------------------------------------------------------
        # 3. Optimizer & scheduler
        # ------------------------------------------------------------------
        self.optimizer = optimizer if optimizer is not None else self._build_optimizer()
        self.lr_scheduler = lr_scheduler if lr_scheduler is not None else self._build_lr_scheduler()

        # ------------------------------------------------------------------
        # 4. Automatic Mixed Precision (AMP) — CUDA only
        # ------------------------------------------------------------------
        self._amp_enabled = (
            self.config.mixed_precision and self.device.type == "cuda"
        )
        if self.config.mixed_precision and self.device.type != "cuda":
            warnings.warn(
                f"mixed_precision=True is only supported on CUDA. "
                f"Running on '{self.device.type}' — AMP disabled automatically.",
                UserWarning,
            )
        self._scaler: Optional[torch.cuda.amp.GradScaler] = (
            torch.cuda.amp.GradScaler() if self._amp_enabled else None
        )

        # ------------------------------------------------------------------
        # 5. Metrics & state
        # ------------------------------------------------------------------
        self.train_metric = VelocityErrorMetric()
        self.val_metric = VelocityErrorMetric()
        self.metrics_history: List[Dict[str, Any]] = []

        self.global_step: int = 0
        self.current_epoch: int = 0
        self.best_val_loss: float = float("inf")

        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._print_device_summary()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _print_device_summary(self) -> None:
        info = self._device_info
        amp_str = "AMP=ON (float16)" if self._amp_enabled else "AMP=OFF"
        dp_str = (
            f"DataParallel={len(self.model.device_ids)}xGPU"
            if isinstance(self.model, nn.DataParallel)
            else "SingleDevice"
        )
        name = info.get("device_name", str(self.device))
        mem = f" ({info['total_memory_gb']} GB)" if "total_memory_gb" in info else ""
        print(f"[ScaleFlowTrainer] Device: {name}{mem} | {amp_str} | {dp_str}")

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build AdamW with weight decay only on non-bias/norm params."""
        decay_params = []
        no_decay_params = []
        no_decay_names = ("bias", "norm", "gamma", "beta", "ada_proj")

        for name, param in self._raw_model.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name.lower() for nd in no_decay_names) or param.ndim <= 1:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        return AdamW(
            [
                {"params": decay_params, "weight_decay": self.config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_eps,
        )

    def _build_lr_scheduler(self) -> Optional[Any]:
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
        return None

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move all tensors in a batch dict to self.device (non_blocking for GPU)."""
        non_blocking = self.device.type == "cuda"
        moved: Dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                moved[k] = v.to(self.device, non_blocking=non_blocking)
            elif isinstance(v, dict):
                moved[k] = {
                    sk: sv.to(self.device, non_blocking=non_blocking)
                    if isinstance(sv, torch.Tensor) else sv
                    for sk, sv in v.items()
                }
            else:
                moved[k] = v
        return moved

    def _prepare_text_conditioning(
        self,
        batch: Dict[str, Any],
        B: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Extract or synthesize text conditioning tensors."""
        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")
        e_text = batch.get("e_text")

        if input_ids is None and e_text is None:
            if "captions" in batch and self.tokenizer is not None:
                encoded = self.tokenizer(
                    batch["captions"], padding=True, return_tensors="pt"
                )
                input_ids = encoded["input_ids"].to(self.device)
                if "attention_mask" in encoded:
                    attention_mask = encoded["attention_mask"].to(self.device)
            elif hasattr(self.model, "config") and hasattr(self.model.config, "text_dim"):
                e_text = torch.zeros(B, self.model.config.text_dim, device=self.device)
            else:
                input_ids = torch.zeros((B, 1), dtype=torch.long, device=self.device)

        return input_ids, attention_mask, e_text

    def _run_model(
        self,
        x: torch.Tensor,
        scale: int,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        e_text: Optional[torch.Tensor],
        upsample: bool,
    ) -> torch.Tensor:
        """Run model forward with optional AMP autocast."""
        kwargs = dict(
            scale=scale,
            input_ids=input_ids,
            attention_mask=attention_mask,
            e_text=e_text,
            upsample=upsample,
        )
        if self._amp_enabled:
            with torch.cuda.amp.autocast():
                return self.model(x, **kwargs)
        return self.model(x, **kwargs)

    # ------------------------------------------------------------------
    # Training & validation steps
    # ------------------------------------------------------------------

    def training_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Single forward-backward training step (AMP-aware).

        Args:
            batch: Batch dict from DataLoader.

        Returns:
            Dict with 'loss', 'lr', 'mse', 'mae', 'cosine_sim', 'scale'.
        """
        self.model.train()
        self.optimizer.zero_grad()

        batch = self._move_batch_to_device(batch)
        images = batch["images"]
        B = images.shape[0]

        pyramid = batch["pyramids"] if "pyramids" in batch else self.flow_matcher.build_pyramid(images)

        scale = self.flow_matcher.sample_scale(
            batch_size=B,
            min_scale=self.config.min_scale,
            max_scale=self.config.max_scale,
            per_batch=True,
        )

        flow_batch = self.flow_matcher.prepare_flow_batch(pyramid=pyramid, scale_level=scale)
        input_ids, attention_mask, e_text = self._prepare_text_conditioning(batch, B)

        upsample = self.config.prediction_mode == "upsample"
        x_input = flow_batch.x_prev if upsample else flow_batch.x_t

        v_pred = self._run_model(x_input, scale, input_ids, attention_mask, e_text, upsample)
        loss = self.flow_matcher.compute_loss(v_pred, flow_batch.velocity_target)

        # AMP backward + optimizer step
        if self._amp_enabled and self._scaler is not None:
            self._scaler.scale(loss).backward()
            if self.config.gradient_clip_val > 0.0:
                self._scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self._raw_model.parameters(), max_norm=self.config.gradient_clip_val
                )
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            loss.backward()
            if self.config.gradient_clip_val > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    self._raw_model.parameters(), max_norm=self.config.gradient_clip_val
                )
            self.optimizer.step()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        metric_res = self.train_metric.update(
            v_pred=v_pred, v_target=flow_batch.velocity_target, scale_level=scale
        )

        return {
            "loss": loss.item(),
            "lr": self.optimizer.param_groups[0]["lr"],
            "scale": scale,
            "mse": metric_res["mse"],
            "mae": metric_res["mae"],
            "cosine_sim": metric_res["cosine_sim"],
        }

    @torch.no_grad()
    def validation_step(self) -> Dict[str, float]:
        """Validation loop over val_dataloader."""
        if self.val_dataloader is None:
            return {}

        self.model.eval()
        self.val_metric.reset()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_dataloader:
            batch = self._move_batch_to_device(batch)
            images = batch["images"]
            B = images.shape[0]

            pyramid = batch["pyramids"] if "pyramids" in batch else self.flow_matcher.build_pyramid(images)
            scale = self.flow_matcher.sample_scale(
                batch_size=B,
                min_scale=self.config.min_scale,
                max_scale=self.config.max_scale,
                per_batch=True,
            )

            flow_batch = self.flow_matcher.prepare_flow_batch(pyramid=pyramid, scale_level=scale)
            input_ids, attention_mask, e_text = self._prepare_text_conditioning(batch, B)

            upsample = self.config.prediction_mode == "upsample"
            x_input = flow_batch.x_prev if upsample else flow_batch.x_t

            v_pred = self._run_model(x_input, scale, input_ids, attention_mask, e_text, upsample)
            loss = self.flow_matcher.compute_loss(v_pred, flow_batch.velocity_target)

            total_loss += loss.item()
            num_batches += 1
            self.val_metric.update(
                v_pred=v_pred, v_target=flow_batch.velocity_target, scale_level=scale
            )

        val_summary = self.val_metric.compute()
        val_summary["val_loss"] = total_loss / max(num_batches, 1)
        return val_summary

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        path: Optional[Union[str, Path]] = None,
        tag: Optional[str] = None,
    ) -> Path:
        """Save a complete training checkpoint.

        Args:
            path: Optional explicit save path.
            tag: Optional tag string ('best', 'final', 'step_500', …).

        Returns:
            Path of the saved checkpoint file.
        """
        if path is not None:
            save_path = Path(path)
        else:
            filename = f"checkpoint_{tag or f'step_{self.global_step}'}.pt"
            save_path = self.checkpoint_dir / filename

        save_path.parent.mkdir(parents=True, exist_ok=True)

        state: Dict[str, Any] = {
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "best_val_loss": self.best_val_loss,
            "model_state_dict": self._raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "torch_rng_state": torch.get_rng_state(),
        }

        if self.lr_scheduler is not None:
            state["scheduler_state_dict"] = self.lr_scheduler.state_dict()

        if self._scaler is not None:
            state["amp_scaler_state_dict"] = self._scaler.state_dict()

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
            path: Path to checkpoint .pt file.
            load_optimizer: Restore optimizer state.
            load_scheduler: Restore LR scheduler state.

        Returns:
            Loaded checkpoint dict.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        self._raw_model.load_state_dict(checkpoint["model_state_dict"])

        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if load_scheduler and self.lr_scheduler and "scheduler_state_dict" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self._scaler is not None and "amp_scaler_state_dict" in checkpoint:
            self._scaler.load_state_dict(checkpoint["amp_scaler_state_dict"])

        self.global_step = checkpoint.get("global_step", 0)
        self.current_epoch = checkpoint.get("current_epoch", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        return checkpoint

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(
        self,
        max_steps: Optional[int] = None,
        max_epochs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Main training loop.

        Args:
            max_steps: Override config.max_steps.
            max_epochs: Override config.max_epochs.

        Returns:
            Summary dict with 'global_step', 'epochs', 'final_checkpoint', 'history'.
        """
        total_steps = max_steps or self.config.max_steps
        max_ep = max_epochs or self.config.max_epochs or 1_000_000

        self.train_metric.reset()

        while self.global_step < total_steps and self.current_epoch < max_ep:
            for batch in self.train_dataloader:
                step_metrics = self.training_step(batch)
                self.global_step += 1

                step_metrics["step"] = self.global_step
                step_metrics["epoch"] = self.current_epoch

                if self.global_step % self.config.log_every_n_steps == 0 or self.global_step == 1:
                    self.metrics_history.append(step_metrics)

                if (
                    self.val_dataloader is not None
                    and self.global_step % self.config.eval_every_n_steps == 0
                ):
                    val_metrics = self.validation_step()
                    step_metrics.update(val_metrics)
                    if val_metrics.get("val_loss", float("inf")) < self.best_val_loss:
                        self.best_val_loss = val_metrics["val_loss"]
                        self.save_checkpoint(tag="best")

                if self.global_step % self.config.save_every_n_steps == 0:
                    self.save_checkpoint()

                if self.global_step >= total_steps:
                    break

            self.current_epoch += 1

        final_path = self.save_checkpoint(tag="final")

        return {
            "global_step": self.global_step,
            "epochs": self.current_epoch,
            "final_checkpoint": str(final_path),
            "history": self.metrics_history,
        }
