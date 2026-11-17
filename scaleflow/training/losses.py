"""Loss functions and metrics for Multiscale Flow Matching."""

from typing import Dict, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowMatchingLoss(nn.Module):
    """Flow Matching regression loss.
    
    Computes regression error between predicted velocity v_pred and
    target velocity v_target.
    Supports MSE, L1, and Huber (Smooth L1) loss, with optional time or scale weights.
    """

    def __init__(
        self,
        loss_type: str = "mse",
        huber_beta: float = 0.05,
        reduction: str = "mean",
    ) -> None:
        """Initialize FlowMatchingLoss.
        
        Args:
            loss_type: 'mse', 'l1', or 'huber'.
            huber_beta: Beta parameter for Huber / Smooth L1 loss.
            reduction: 'mean', 'sum', or 'none'.
        """
        super().__init__()
        valid_losses = ("mse", "l1", "huber", "smooth_l1")
        if loss_type.lower() not in valid_losses:
            raise ValueError(f"Unknown loss_type: {loss_type}. Choose from {valid_losses}")
        self.loss_type = loss_type.lower()
        self.huber_beta = huber_beta
        self.reduction = reduction

    def forward(
        self,
        v_pred: torch.Tensor,
        v_target: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute flow matching loss.
        
        Args:
            v_pred: Predicted velocity field tensor of shape (B, C, H, W).
            v_target: Target velocity field tensor of same shape.
            weights: Optional per-sample weights of shape (B,) or broadcastable.
            
        Returns:
            Computed scalar loss (if reduction is 'mean' or 'sum') or tensor.
        """
        if v_pred.shape != v_target.shape:
            raise ValueError(
                f"Shape mismatch: v_pred {v_pred.shape} vs v_target {v_target.shape}"
            )

        if self.loss_type == "mse":
            loss_raw = (v_pred - v_target) ** 2
        elif self.loss_type == "l1":
            loss_raw = torch.abs(v_pred - v_target)
        elif self.loss_type in ("huber", "smooth_l1"):
            loss_raw = F.smooth_l1_loss(v_pred, v_target, beta=self.huber_beta, reduction="none")
        else:
            loss_raw = (v_pred - v_target) ** 2

        # Mean over spatial and channel dimensions: shape (B,)
        per_sample_loss = loss_raw.mean(dim=tuple(range(1, v_pred.ndim)))

        if weights is not None:
            if weights.ndim == 1 and weights.shape[0] == per_sample_loss.shape[0]:
                per_sample_loss = per_sample_loss * weights
            else:
                per_sample_loss = per_sample_loss * weights.view(-1)

        if self.reduction == "mean":
            return per_sample_loss.mean()
        elif self.reduction == "sum":
            return per_sample_loss.sum()
        elif self.reduction == "none":
            return per_sample_loss
        else:
            return per_sample_loss.mean()


class MultiscaleFlowLoss(nn.Module):
    """Multiscale flow matching loss with per-scale weighting."""

    def __init__(
        self,
        scale_weights: Optional[Dict[int, float]] = None,
        base_loss_type: str = "mse",
        reduction: str = "mean",
    ) -> None:
        """Initialize MultiscaleFlowLoss.
        
        Args:
            scale_weights: Optional dict mapping scale index [0..7] to float weight.
            base_loss_type: Underlying loss type ('mse', 'l1', 'huber').
            reduction: 'mean' or 'sum'.
        """
        super().__init__()
        self.scale_weights = scale_weights or {}
        self.base_loss = FlowMatchingLoss(loss_type=base_loss_type, reduction=reduction)

    def forward(
        self,
        v_pred: torch.Tensor,
        v_target: torch.Tensor,
        scale_level: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        """Compute scale-weighted flow matching loss.
        
        Args:
            v_pred: Predicted velocity tensor.
            v_target: Target velocity tensor.
            scale_level: Integer or tensor specifying current scale level.
            
        Returns:
            Weighted loss tensor.
        """
        base_loss = self.base_loss(v_pred, v_target)
        if isinstance(scale_level, torch.Tensor):
            if scale_level.numel() == 1:
                scale_idx = int(scale_level.item())
                weight = self.scale_weights.get(scale_idx, 1.0)
                return base_loss * weight
            else:
                # Per-sample scale levels
                weights = torch.tensor(
                    [self.scale_weights.get(int(s.item()), 1.0) for s in scale_level],
                    device=v_pred.device,
                    dtype=v_pred.dtype,
                )
                return self.base_loss(v_pred, v_target, weights=weights)
        else:
            weight = self.scale_weights.get(int(scale_level), 1.0)
            return base_loss * weight


class VelocityErrorMetric:
    """Computes and tracks velocity error metrics during training and evaluation."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset metric accumulators."""
        self.total_samples: int = 0
        self.sum_mse: float = 0.0
        self.sum_mae: float = 0.0
        self.sum_cosine_sim: float = 0.0
        self.scale_counts: Dict[int, int] = {}
        self.scale_mse: Dict[int, float] = {}

    @torch.no_grad()
    def update(
        self,
        v_pred: torch.Tensor,
        v_target: torch.Tensor,
        scale_level: Optional[Union[int, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """Update metrics with a new batch.
        
        Args:
            v_pred: Predicted velocity tensor (B, C, H, W).
            v_target: Target velocity tensor (B, C, H, W).
            scale_level: Optional scale level(s) for the batch.
            
        Returns:
            Batch-level metrics dict with keys: 'mse', 'mae', 'rmse', 'cosine_sim'.
        """
        batch_size = v_pred.shape[0]
        v_p = v_pred.detach()
        v_t = v_target.detach()

        # Flatten spatial and channel dimensions for vector ops
        flat_p = v_p.reshape(batch_size, -1)
        flat_t = v_t.reshape(batch_size, -1)

        # MSE and MAE
        mse_per_sample = torch.mean((flat_p - flat_t) ** 2, dim=-1)
        mae_per_sample = torch.mean(torch.abs(flat_p - flat_t), dim=-1)

        # Cosine similarity
        eps = 1e-8
        norm_p = torch.norm(flat_p, p=2, dim=-1, keepdim=True)
        norm_t = torch.norm(flat_t, p=2, dim=-1, keepdim=True)
        cos_sim = torch.sum(flat_p * flat_t, dim=-1, keepdim=True) / (norm_p * norm_t + eps)
        cos_sim_per_sample = cos_sim.squeeze(-1)

        batch_mse = mse_per_sample.mean().item()
        batch_mae = mae_per_sample.mean().item()
        batch_cos = cos_sim_per_sample.mean().item()

        # Accumulate
        self.total_samples += batch_size
        self.sum_mse += mse_per_sample.sum().item()
        self.sum_mae += mae_per_sample.sum().item()
        self.sum_cosine_sim += cos_sim_per_sample.sum().item()

        # Track per-scale if provided
        if scale_level is not None:
            if isinstance(scale_level, torch.Tensor):
                if scale_level.numel() == 1:
                    s_idx = int(scale_level.item())
                    self.scale_counts[s_idx] = self.scale_counts.get(s_idx, 0) + batch_size
                    self.scale_mse[s_idx] = self.scale_mse.get(s_idx, 0.0) + mse_per_sample.sum().item()
                else:
                    for s, m in zip(scale_level, mse_per_sample):
                        s_idx = int(s.item())
                        self.scale_counts[s_idx] = self.scale_counts.get(s_idx, 0) + 1
                        self.scale_mse[s_idx] = self.scale_mse.get(s_idx, 0.0) + m.item()
            else:
                s_idx = int(scale_level)
                self.scale_counts[s_idx] = self.scale_counts.get(s_idx, 0) + batch_size
                self.scale_mse[s_idx] = self.scale_mse.get(s_idx, 0.0) + mse_per_sample.sum().item()

        return {
            "mse": batch_mse,
            "mae": batch_mae,
            "rmse": float(torch.sqrt(torch.tensor(batch_mse)).item()),
            "cosine_sim": batch_cos,
        }

    def compute(self) -> Dict[str, float]:
        """Compute accumulated aggregate metrics.
        
        Returns:
            Dict containing mean MSE, MAE, RMSE, cosine similarity, and per-scale MSE.
        """
        if self.total_samples == 0:
            return {"mse": 0.0, "mae": 0.0, "rmse": 0.0, "cosine_sim": 0.0}

        avg_mse = self.sum_mse / self.total_samples
        avg_mae = self.sum_mae / self.total_samples
        avg_cos = self.sum_cosine_sim / self.total_samples

        result = {
            "mse": avg_mse,
            "mae": avg_mae,
            "rmse": float(avg_mse ** 0.5),
            "cosine_sim": avg_cos,
        }

        # Add per-scale MSE if any recorded
        for s, count in self.scale_counts.items():
            if count > 0:
                result[f"scale_{s}_mse"] = self.scale_mse[s] / count

        return result
