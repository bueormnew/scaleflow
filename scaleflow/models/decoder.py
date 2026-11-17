"""Recursive ScaleFlowDecoder using ConvNeXt V2 blocks with AdaLN and PixelShuffle.

Features:
- ConvNeXt V2 blocks with 7x7 Depthwise convolutions.
- AdaLN (Adaptive Layer Normalization) conditioned on c = MLP(e_text + e_scale).
- Pointwise 1x1 convs, GELU, and GRN (Global Response Normalization).
- PixelShuffle(2x) upsampler head producing RGB image at 2x resolution.
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRN(nn.Module):
    """Global Response Normalization (GRN) layer from ConvNeXt V2.
    
    Normalizes feature maps across channels based on their L2 norm,
    promoting feature competition and preventing feature collapse.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """Initialize GRN.
        
        Args:
            dim: Number of channels.
            eps: Epsilon for numerical stability.
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Tensor of shape (B, H, W, C).
            
        Returns:
            Normalized tensor of shape (B, H, W, C).
        """
        # Spatial L2 norm: gx has shape (B, 1, 1, C)
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        # Normalize across channel dimension
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
        # Scale, shift and residual
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtV2AdaLNBlock(nn.Module):
    """ConvNeXt V2 block with AdaLN (Adaptive Layer Normalization).
    
    1. 7x7 Depthwise Conv2d
    2. Channels-last permutation (B, C, H, W) -> (B, H, W, C)
    3. LayerNorm modulated by conditioning c (AdaLN: scale, shift, gate)
    4. Pointwise Linear (dim -> 4*dim)
    5. GELU
    6. GRN (Global Response Normalization)
    7. Pointwise Linear (4*dim -> dim)
    8. Gated residual connection
    9. Permutation back to (B, C, H, W)
    """

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        mlp_ratio: float = 4.0,
        grn_eps: float = 1e-6,
    ) -> None:
        """Initialize ConvNeXtV2AdaLNBlock.
        
        Args:
            dim: Number of feature channels.
            cond_dim: Dimension of conditioning vector c.
            mlp_ratio: Multiplier for hidden dimension in FFN.
            grn_eps: Epsilon for GRN.
        """
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim

        # 7x7 Depthwise convolution
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=7,
            padding=3,
            groups=dim,
            bias=True,
        )

        # Normalization (elementwise_affine=False because AdaLN supplies scale and shift)
        self.norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)

        # AdaLN projection: maps condition vector c to (scale, shift, gate)
        self.ada_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 3 * dim),
        )
        # Initialize ada_proj to zero for identity starting point (AdaLN-Zero)
        nn.init.zeros_(self.ada_proj[-1].weight)
        nn.init.zeros_(self.ada_proj[-1].bias)

        # Pointwise expansions
        hidden_dim = int(dim * mlp_ratio)
        self.pwconv1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.grn = GRN(hidden_dim, eps=grn_eps)
        self.pwconv2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Feature map tensor of shape (B, C, H, W).
            c: Conditioning tensor of shape (B, cond_dim).
            
        Returns:
            Output feature map of shape (B, C, H, W).
        """
        residual = x

        # 1. Depthwise 7x7 conv
        x = self.dwconv(x)

        # 2. Permute to channels-last: (B, H, W, C)
        x = x.permute(0, 2, 3, 1)

        # 3. AdaLN modulation
        # ada_proj produces (B, 3 * dim)
        ada_params = self.ada_proj(c)  # (B, 3 * dim)
        scale, shift, gate = ada_params.chunk(3, dim=-1)
        scale = scale.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, dim)
        shift = shift.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, dim)
        gate = gate.unsqueeze(1).unsqueeze(1)    # (B, 1, 1, dim)

        x = self.norm(x)
        x = x * (1.0 + scale) + shift

        # 4. Pointwise conv1
        x = self.pwconv1(x)

        # 5. GELU
        x = self.act(x)

        # 6. GRN
        x = self.grn(x)

        # 7. Pointwise conv2
        x = self.pwconv2(x)

        # 8. Permute back to (B, C, H, W)
        x = x.permute(0, 3, 1, 2)

        # Gated residual: (B, C, 1, 1) gate
        gate = gate.permute(0, 3, 1, 2)
        return residual + gate * x


class PixelShuffleUpsampler(nn.Module):
    """PixelShuffle(2x) upsampler head producing RGB image at 2x resolution.
    
    Maps feature channels (C) to (out_channels * 4) via 3x3 conv,
    then applies PixelShuffle(2) to double spatial dimensions (2H, 2W).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 3,
        upscale_factor: int = 2,
    ) -> None:
        super().__init__()
        self.upscale_factor = upscale_factor
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * (upscale_factor ** 2),
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input feature map of shape (B, in_channels, H, W).
            
        Returns:
            RGB image tensor of shape (B, out_channels, H * 2, W * 2).
        """
        x = self.conv(x)
        return self.pixel_shuffle(x)


class ScaleFlowDecoder(nn.Module):
    """Recursive ScaleFlowDecoder.
    
    Processes image/latent representations at resolution HxW and condition
    c = MLP(e_text + e_scale), then uses a PixelShuffle(2x) upsampler head
    to produce RGB image at 2x resolution (2H x 2W).
    
    Can also be run recursively to generate multi-resolution pyramids
    from 8x8 up to 1024x1024.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        hidden_dim: int = 256,
        cond_dim: int = 512,
        depth: int = 4,
        mlp_ratio: float = 4.0,
    ) -> None:
        """Initialize ScaleFlowDecoder.
        
        Args:
            in_channels: Number of input image channels (e.g. 3 for RGB).
            out_channels: Number of output image channels (e.g. 3 for RGB).
            hidden_dim: Number of hidden channels in ConvNeXt V2 blocks.
            cond_dim: Dimension of text and scale embeddings.
            depth: Number of ConvNeXt V2 blocks.
            mlp_ratio: FFN expansion ratio in ConvNeXt V2 blocks.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.depth = depth

        # Conditioning MLP: c = MLP(e_text + e_scale)
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # Input projection from image space to feature space
        self.in_proj = nn.Conv2d(
            in_channels,
            hidden_dim,
            kernel_size=3,
            padding=1,
            bias=True,
        )

        # ConvNeXt V2 blocks with AdaLN
        self.blocks = nn.ModuleList([
            ConvNeXtV2AdaLNBlock(
                dim=hidden_dim,
                cond_dim=cond_dim,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(depth)
        ])

        # Final normalization before output head
        self.final_norm = nn.GroupNorm(num_groups=min(32, hidden_dim), num_channels=hidden_dim)

        # Upsampler head: PixelShuffle(2x) producing RGB image at 2x resolution
        self.upsampler = PixelShuffleUpsampler(
            in_channels=hidden_dim,
            out_channels=out_channels,
            upscale_factor=2,
        )

        # Optional direct head (same resolution) for velocity / direct prediction
        self.direct_head = nn.Conv2d(
            hidden_dim,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )

    def compute_conditioning(
        self,
        e_text: torch.Tensor,
        e_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Compute conditioning vector c = MLP(e_text + e_scale).
        
        Args:
            e_text: Text embedding of shape (B, cond_dim).
            e_scale: Scale embedding of shape (B, cond_dim).
            
        Returns:
            Conditioning vector c of shape (B, cond_dim).
        """
        # Broadcast e_scale if batch sizes differ (e.g. e_scale is single vector)
        if e_scale.ndim == 1:
            e_scale = e_scale.unsqueeze(0)
        if e_scale.shape[0] != e_text.shape[0]:
            e_scale = e_scale.expand(e_text.shape[0], -1)

        summed = e_text + e_scale
        return self.cond_mlp(summed)

    def forward(
        self,
        x: torch.Tensor,
        e_text: torch.Tensor,
        e_scale: torch.Tensor,
        upsample: bool = True,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input image/latent tensor of shape (B, in_channels, H, W).
            e_text: Text embedding of shape (B, cond_dim).
            e_scale: Scale embedding of shape (B, cond_dim).
            upsample: If True, uses PixelShuffle(2x) upsampler to output (B, out_channels, 2H, 2W).
                     If False, outputs at same resolution (B, out_channels, H, W).
                     
        Returns:
            Output image tensor at 2x resolution (if upsample=True) or 1x resolution.
        """
        # Condition vector c = MLP(e_text + e_scale)
        c = self.compute_conditioning(e_text, e_scale)

        # Project input image to feature space
        h = self.in_proj(x)

        # Pass through ConvNeXt V2 blocks with AdaLN
        for block in self.blocks:
            h = block(h, c)

        h = self.final_norm(h)

        if upsample:
            out = self.upsampler(h)
        else:
            out = self.direct_head(h)

        return out
