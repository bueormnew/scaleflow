"""Multiscale Optimal Transport Flow Matching (OT-CFM) implementation.

Features:
- Multi-resolution image pyramid generation (8x8 ... 1024x1024).
- Scale level sampling s in [0..7].
- Noise interpolation between noisy previous scale and clean next scale.
- Continuous Optimal Transport velocity field targets v_t = d(x_t)/dt.
- Flow matching regression loss computation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from scaleflow.training.losses import FlowMatchingLoss, MultiscaleFlowLoss


# Standard ScaleFlow resolutions corresponding to discrete scale levels 0 to 7
DEFAULT_SCALE_RESOLUTIONS: Dict[int, int] = {
    0: 8,
    1: 16,
    2: 32,
    3: 64,
    4: 128,
    5: 256,
    6: 512,
    7: 1024,
}


def build_image_pyramid(
    image: torch.Tensor,
    resolutions: Optional[Union[List[int], Dict[int, int]]] = None,
    mode: str = "bilinear",
) -> Dict[int, torch.Tensor]:
    """Generate a multi-resolution image pyramid from a high-resolution image tensor.
    
    Args:
        image: High-resolution tensor of shape (B, C, H, W) normalized to [-1, 1].
        resolutions: List of spatial dimensions (e.g. [8, 16, 32, ...]) or
                     dict mapping scale level to resolution. Defaults to DEFAULT_SCALE_RESOLUTIONS.
        mode: Interpolation mode ('bilinear', 'bicubic', or 'area').
        
    Returns:
        Dict mapping scale index s to tensor of shape (B, C, res_s, res_s).
    """
    if resolutions is None:
        resolutions = DEFAULT_SCALE_RESOLUTIONS

    if isinstance(resolutions, (list, tuple)):
        scale_map = {idx: res for idx, res in enumerate(resolutions)}
    else:
        scale_map = dict(resolutions)

    B, C, H, W = image.shape
    pyramid: Dict[int, torch.Tensor] = {}

    for s, res in scale_map.items():
        if H == res and W == res:
            pyramid[s] = image
        else:
            # Use antialias=True when downsampling if supported by mode
            use_antialias = mode in ("bilinear", "bicubic") and (H > res or W > res)
            try:
                pyramid[s] = F.interpolate(
                    image,
                    size=(res, res),
                    mode=mode,
                    align_corners=False if mode in ("bilinear", "bicubic") else None,
                    antialias=use_antialias,
                )
            except TypeError:
                # Fallback for older torch without antialias argument
                pyramid[s] = F.interpolate(
                    image,
                    size=(res, res),
                    mode=mode,
                    align_corners=False if mode in ("bilinear", "bicubic") else None,
                )

    return pyramid


@dataclass
class FlowMatchingBatch:
    """Container holding intermediate and target tensors for a training step."""
    scale: int
    x_prev: torch.Tensor       # Shape (B, C, H_prev, W_prev) or base representation
    x_0: torch.Tensor          # Source state at current scale (B, C, H_curr, W_curr)
    x_1: torch.Tensor          # Clean target image at current scale (B, C, H_curr, W_curr)
    x_t: torch.Tensor          # Interpolated noisy state at time t (B, C, H_curr, W_curr)
    t: torch.Tensor            # Time values of shape (B,)
    velocity_target: torch.Tensor  # Target velocity field v_t (B, C, H_curr, W_curr)


class MultiscaleOTFlowMatching(nn.Module):
    """Multiscale Optimal Transport Conditional Flow Matching (OT-CFM).
    
    Implements continuous flow matching trajectories across multi-resolution scales:
    - At scale 0 (8x8): source x_0 is standard Gaussian noise, target x_1 is clean 8x8.
    - At scale s > 0: source x_0 is noisy upsampled previous scale x^{(s-1)},
      interpolated with clean target x^{(s)}.
    - Trajectory: x_t = (1 - (1 - sigma_min) * t) * x_0 + t * x_1
    - Target velocity: v_t = x_1 - (1 - sigma_min) * x_0
    """

    def __init__(
        self,
        scale_resolutions: Optional[Dict[int, int]] = None,
        sigma_min: float = 1e-4,
        noise_prev_scale_sigma: float = 0.5,
        prev_scale_noise_mode: str = "blend",
        loss_type: str = "mse",
        huber_beta: float = 0.05,
    ) -> None:
        """Initialize MultiscaleOTFlowMatching.
        
        Args:
            scale_resolutions: Mapping from scale index to resolution.
                               Defaults to {0: 8, 1: 16, ..., 7: 1024}.
            sigma_min: Small standard deviation added to OT path for stability.
            noise_prev_scale_sigma: Fraction of noise mixed into previous scale (0.0 to 1.0).
            prev_scale_noise_mode: 'blend' ((1-s)*x + s*eps), 'additive' (x + s*eps), or 'noise_only' (eps).
            loss_type: 'mse', 'l1', or 'huber'.
            huber_beta: Beta for Huber loss if loss_type='huber'.
        """
        super().__init__()
        self.scale_resolutions = scale_resolutions or DEFAULT_SCALE_RESOLUTIONS
        self.num_scales = len(self.scale_resolutions)
        self.sigma_min = sigma_min
        self.noise_prev_scale_sigma = noise_prev_scale_sigma
        self.prev_scale_noise_mode = prev_scale_noise_mode

        self.loss_fn = FlowMatchingLoss(loss_type=loss_type, huber_beta=huber_beta)
        self.multiscale_loss_fn = MultiscaleFlowLoss(base_loss_type=loss_type)

    def sample_scale(
        self,
        batch_size: int = 1,
        min_scale: int = 0,
        max_scale: Optional[int] = None,
        device: Optional[torch.device] = None,
        per_batch: bool = True,
    ) -> Union[int, torch.Tensor]:
        """Sample discrete scale level(s) in [min_scale, max_scale].
        
        Args:
            batch_size: Batch size.
            min_scale: Minimum scale index (default: 0).
            max_scale: Maximum scale index (default: max scale in scale_resolutions).
            device: Torch device.
            per_batch: If True, returns a single integer scale for the whole batch
                       (standard for spatial convolution batches).
                       If False, returns tensor of shape (B,).
                       
        Returns:
            Integer scale index if per_batch=True, else 1D tensor of shape (B,).
        """
        if max_scale is None:
            max_scale = max(self.scale_resolutions.keys())

        if per_batch:
            s = int(torch.randint(min_scale, max_scale + 1, (1,)).item())
            return s
        else:
            return torch.randint(min_scale, max_scale + 1, (batch_size,), device=device)

    def sample_time(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Sample continuous time steps t uniformly from [0, 1].
        
        Args:
            batch_size: Number of samples.
            device: Device for output tensor.
            dtype: Tensor dtype.
            
        Returns:
            Tensor of shape (B,) with values in [0, 1].
        """
        return torch.rand((batch_size,), device=device, dtype=dtype)

    def build_pyramid(
        self,
        image: torch.Tensor,
        mode: str = "bilinear",
    ) -> Dict[int, torch.Tensor]:
        """Generate multi-resolution image pyramid for this flow matcher's scales."""
        return build_image_pyramid(image, resolutions=self.scale_resolutions, mode=mode)

    def noise_interpolate_previous_scale(
        self,
        x_prev: Optional[torch.Tensor],
        target_res: int,
        noise: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        batch_size: int = 1,
        channels: int = 3,
    ) -> torch.Tensor:
        """Interpolate previous scale image with Gaussian noise at target resolution.
        
        Args:
            x_prev: Tensor of previous scale at resolution (r_prev, r_prev),
                    or None if at base scale 0.
            target_res: Target resolution for current scale.
            noise: Optional pre-generated noise tensor.
            device: Torch device if x_prev is None.
            dtype: Torch dtype if x_prev is None.
            batch_size: Batch size if x_prev is None.
            channels: Number of image channels (default: 3).
            
        Returns:
            Source tensor x_0 of shape (B, C, target_res, target_res).
        """
        if x_prev is None:
            # Scale 0: Pure Gaussian noise
            if noise is not None:
                return noise
            dev = device or torch.device("cpu")
            dt = dtype or torch.float32
            return torch.randn(batch_size, channels, target_res, target_res, device=dev, dtype=dt)

        B, C, H, W = x_prev.shape
        if noise is None:
            noise = torch.randn(B, C, target_res, target_res, device=x_prev.device, dtype=x_prev.dtype)

        # Upsample previous scale to target resolution
        if H == target_res and W == target_res:
            x_up = x_prev
        else:
            x_up = F.interpolate(x_prev, size=(target_res, target_res), mode="bilinear", align_corners=False)

        sigma = self.noise_prev_scale_sigma
        if self.prev_scale_noise_mode == "blend":
            # Linear combination: (1 - sigma) * x_up + sigma * noise
            return (1.0 - sigma) * x_up + sigma * noise
        elif self.prev_scale_noise_mode == "additive":
            # Additive noise: x_up + sigma * noise
            return x_up + sigma * noise
        elif self.prev_scale_noise_mode == "noise_only":
            return noise
        else:
            return (1.0 - sigma) * x_up + sigma * noise

    def compute_ot_trajectory(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Optimal Transport flow interpolation and target velocity field.
        
        Args:
            x_0: Source state tensor (B, C, H, W).
            x_1: Target clean state tensor (B, C, H, W).
            t: Optional time tensor of shape (B,). If None, sampled uniformly in [0, 1].
            
        Returns:
            Tuple of (x_t, t, v_t):
            - x_t: Interpolated noisy state at time t of shape (B, C, H, W).
            - t: Time tensor of shape (B,).
            - v_t: Target velocity field of shape (B, C, H, W).
        """
        B = x_0.shape[0]
        if t is None:
            t = self.sample_time(B, device=x_0.device, dtype=x_0.dtype)

        # Reshape t to (B, 1, 1, 1) for broadcasting
        t_expand = t.view(B, *([1] * (x_0.ndim - 1)))

        # OT conditional flow:
        # x_t = (1 - (1 - sigma_min) * t) * x_0 + t * x_1
        # v_t = d(x_t)/dt = x_1 - (1 - sigma_min) * x_0
        alpha = 1.0 - self.sigma_min
        x_t = (1.0 - alpha * t_expand) * x_0 + t_expand * x_1
        v_t = x_1 - alpha * x_0

        return x_t, t, v_t

    def prepare_flow_batch(
        self,
        pyramid: Dict[int, torch.Tensor],
        scale_level: Optional[int] = None,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> FlowMatchingBatch:
        """Prepare all intermediate tensors and velocity targets for a flow matching step.
        
        Args:
            pyramid: Multi-resolution image pyramid dict {scale_idx: tensor}.
            scale_level: Scale index to sample. If None, sampled uniformly.
            t: Optional time tensor of shape (B,).
            noise: Optional noise tensor.
            
        Returns:
            FlowMatchingBatch dataclass holding x_prev, x_0, x_1, x_t, t, v_t.
        """
        if scale_level is None:
            scale_level = self.sample_scale(per_batch=True)

        res_curr = self.scale_resolutions[scale_level]
        x_1 = pyramid[scale_level]
        B, C, _, _ = x_1.shape

        if scale_level == 0:
            # Base scale: previous scale does not exist
            # Create a 4x4 representation or base noise
            x_prev = torch.zeros(B, C, res_curr // 2, res_curr // 2, device=x_1.device, dtype=x_1.dtype)
            x_0 = self.noise_interpolate_previous_scale(
                x_prev=None,
                target_res=res_curr,
                noise=noise,
                device=x_1.device,
                dtype=x_1.dtype,
                batch_size=B,
                channels=C,
            )
        else:
            prev_scale = scale_level - 1
            x_prev = pyramid[prev_scale]
            x_0 = self.noise_interpolate_previous_scale(
                x_prev=x_prev,
                target_res=res_curr,
                noise=noise,
            )

        x_t, t_sampled, v_t = self.compute_ot_trajectory(x_0=x_0, x_1=x_1, t=t)

        return FlowMatchingBatch(
            scale=scale_level,
            x_prev=x_prev,
            x_0=x_0,
            x_1=x_1,
            x_t=x_t,
            t=t_sampled,
            velocity_target=v_t,
        )

    def compute_loss(
        self,
        v_pred: torch.Tensor,
        v_target: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute Flow Matching regression loss.
        
        Args:
            v_pred: Predicted velocity field from model (B, C, H, W).
            v_target: Ground truth velocity field v_t (B, C, H, W).
            weights: Optional per-sample weights.
            
        Returns:
            Scalar regression loss tensor.
        """
        return self.loss_fn(v_pred, v_target, weights=weights)


# Aliases for backward and interface compatibility
MultiscaleFlowMatching = MultiscaleOTFlowMatching
FlowMatching = MultiscaleOTFlowMatching
