"""Sinusoidal scale embedding module for ScaleFlow.

Maps discrete scale levels s in [0..7] (corresponding to resolutions 8x8 up to 1024x1024)
to dense conditioning vectors in R^D.
"""

from typing import Union
import math
import torch
import torch.nn as nn


# Resolution table mapping scale level to spatial dimension (H = W)
SCALE_RESOLUTIONS = {
    0: 8,
    1: 16,
    2: 32,
    3: 64,
    4: 128,
    5: 256,
    6: 512,
    7: 1024,
}

RESOLUTION_TO_SCALE = {res: scale for scale, res in SCALE_RESOLUTIONS.items()}


def resolution_to_scale(resolution: int) -> int:
    """Convert resolution (e.g. 8, 16, ..., 1024) to scale index [0..7]."""
    if resolution in RESOLUTION_TO_SCALE:
        return RESOLUTION_TO_SCALE[resolution]
    # Fallback to log2 formula: 8 * 2^s = res => s = round(log2(res / 8))
    scale = int(round(math.log2(resolution / 8.0)))
    return max(0, min(7, scale))


def scale_to_resolution(scale: int) -> int:
    """Convert scale index [0..7] to resolution (8, 16, ..., 1024)."""
    if scale in SCALE_RESOLUTIONS:
        return SCALE_RESOLUTIONS[scale]
    return int(8 * (2 ** scale))


class SinusoidalScaleEmbedding(nn.Module):
    """Sinusoidal scale embedding for multiscale flow matching.
    
    Maps discrete scale levels s in [0..7] (or continuous scale values) to
    embedding vectors in R^dim.
    """

    def __init__(
        self,
        dim: int,
        max_scale: int = 8,
        use_mlp: bool = True,
        mlp_expansion: int = 2,
    ) -> None:
        """Initialize SinusoidalScaleEmbedding.
        
        Args:
            dim: Dimension of output embedding.
            max_scale: Number of scale levels (default: 8, for 8x8 to 1024x1024).
            use_mlp: Whether to project sinusoidal features through an MLP.
            mlp_expansion: Hidden dimension factor for the MLP projector.
        """
        super().__init__()
        self.dim = dim
        self.max_scale = max_scale
        self.use_mlp = use_mlp

        # Half dimension for sin/cos frequencies
        self.half_dim = dim // 2
        
        # Precompute frequency denominator: exp(-log(10000) * i / (half_dim - 1))
        # Registered as buffer so it moves with device and dtype
        freq_bands = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, self.half_dim, dtype=torch.float32)
            / max(self.half_dim - 1, 1)
        )
        self.register_buffer("freq_bands", freq_bands, persistent=True)

        if self.use_mlp:
            hidden_dim = dim * mlp_expansion
            self.mlp = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, dim),
            )
        else:
            self.mlp = nn.Identity()

    def get_sinusoidal_embedding(self, s: torch.Tensor) -> torch.Tensor:
        """Compute raw sinusoidal embeddings from scale values.
        
        Args:
            s: Tensor of shape (B,) or (B, 1) containing scale values.
            
        Returns:
            Tensor of shape (B, dim) with sinusoidal encoding.
        """
        if s.ndim == 1:
            s = s.unsqueeze(-1)  # (B, 1)
        s = s.to(dtype=torch.float32)

        # Outer product: (B, 1) * (1, half_dim) -> (B, half_dim)
        angles = s * self.freq_bands.unsqueeze(0)
        sin_emb = torch.sin(angles)
        cos_emb = torch.cos(angles)
        emb = torch.cat([sin_emb, cos_emb], dim=-1)  # (B, 2 * half_dim)

        # If dim is odd, pad with zeros
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)

        return emb

    def forward(
        self,
        scale: Union[int, float, torch.Tensor],
        device: torch.device = None,
    ) -> torch.Tensor:
        """Forward pass to compute scale embedding.
        
        Args:
            scale: Scale index or batch of indices [0..7].
                   Can be an int, float, or 1D/2D torch.Tensor.
            device: Optional torch.device to use if scale is int/float.
            
        Returns:
            Embedding tensor of shape (B, dim).
        """
        if not isinstance(scale, torch.Tensor):
            if device is None:
                device = self.freq_bands.device
            scale = torch.tensor([scale], dtype=torch.float32, device=device)
        elif scale.device != self.freq_bands.device:
            scale = scale.to(device=self.freq_bands.device)

        if scale.ndim == 0:
            scale = scale.unsqueeze(0)

        emb = self.get_sinusoidal_embedding(scale)
        return self.mlp(emb)
