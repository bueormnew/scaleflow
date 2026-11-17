"""Recursive ScaleFlowDecoder using ConvNeXt V2 blocks with AdaLN, PixelShuffle,
and ScaleAwareCrossAttention.

Features:
- ConvNeXt V2 blocks with 7x7 Depthwise convolutions.
- AdaLN (Adaptive Layer Normalization) conditioned on c = MLP(e_text + e_scale).
- Pointwise 1x1 convs, GELU, and GRN (Global Response Normalization).
- ScaleAwareCrossAttention: Multi-head spatial cross-attention between visual queries
  and text tokens for early scales (<= 64x64), bypassed with 0 compute at high scales.
- PixelShuffle(2x) upsampler head producing RGB image at 2x resolution.
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .scale_embed import resolution_to_scale


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


class ScaleAwareCrossAttention(nn.Module):
    """Scale-aware multi-head cross-attention between visual queries and text keys/values.
    
    Conditioned on scale index s and cross_attention_max_scale (default 3, covering <= 64x64).
    - If s <= cross_attention_max_scale: executes spatial cross-attention with zero-initialized residual gate.
    - If s > cross_attention_max_scale (128 to 1024): completely bypassed with 0 extra compute.
    - If text_sequence is None: bypasses cleanly for backward compatibility.
    """

    def __init__(
        self,
        dim: int,
        context_dim: int,
        num_heads: int = 8,
        cross_attention_max_scale: int = 3,
        dropout: float = 0.0,
    ) -> None:
        """Initialize ScaleAwareCrossAttention.
        
        Args:
            dim: Visual feature channels.
            context_dim: Text feature channels (d_model).
            num_heads: Number of attention heads.
            cross_attention_max_scale: Maximum scale index for which cross-attention is executed (default: 3, <= 64x64).
            dropout: Attention dropout probability.
        """
        super().__init__()
        self.dim = dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.cross_attention_max_scale = cross_attention_max_scale

        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            kdim=context_dim,
            vdim=context_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        # Zero-initialized residual gate (gate * attn_out)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        text_sequence: Optional[torch.Tensor] = None,
        scale: Optional[Union[int, float, torch.Tensor]] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Visual feature map of shape (B, C, H, W).
            text_sequence: Text token features of shape (B, L, D), or None.
            scale: Scale level index [0..7] (or continuous scale). If None, inferred from resolution.
            text_mask: Optional attention mask (B, L) where 1 indicates valid token and 0 indicates pad.
            
        Returns:
            Updated visual feature map of shape (B, C, H, W).
        """
        # Backward compatibility: if text_sequence is None, bypass cleanly
        if text_sequence is None:
            return x

        # Check scale condition
        if scale is not None:
            if isinstance(scale, torch.Tensor):
                current_scale = int(scale.reshape(-1)[0].item())
            else:
                current_scale = int(scale)
        else:
            # Infer from spatial resolution
            current_scale = resolution_to_scale(max(x.shape[-2], x.shape[-1]))

        # If s > cross_attention_max_scale (e.g. 128 to 1024), completely bypassed with 0 extra compute
        if current_scale > self.cross_attention_max_scale:
            return x

        B, C, H, W = x.shape
        residual = x

        # Reshape visual queries to (B, H*W, C)
        q = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        q_norm = self.norm(q)

        # Prepare key_padding_mask: True for padding positions
        key_padding_mask = None
        if text_mask is not None:
            if text_mask.dtype == torch.bool:
                key_padding_mask = ~text_mask
            else:
                key_padding_mask = (text_mask == 0)

        # Multi-head cross-attention
        attn_out, _ = self.attn(
            query=q_norm,
            key=text_sequence,
            value=text_sequence,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        # Reshape back to (B, C, H, W)
        attn_out = attn_out.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return residual + self.gate * attn_out


class ConvNeXtV2AdaLNBlock(nn.Module):
    """ConvNeXt V2 block with AdaLN (Adaptive Layer Normalization) and ScaleAwareCrossAttention.
    
    1. 7x7 Depthwise Conv2d
    2. Channels-last permutation (B, C, H, W) -> (B, H, W, C)
    3. LayerNorm modulated by conditioning c (AdaLN: scale, shift, gate)
    4. Pointwise Linear (dim -> 4*dim)
    5. GELU
    6. GRN (Global Response Normalization)
    7. Pointwise Linear (4*dim -> dim)
    8. Gated residual connection
    9. Permutation back to (B, C, H, W)
    10. ScaleAwareCrossAttention (if enabled and scale <= max_scale)
    """

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        mlp_ratio: float = 4.0,
        grn_eps: float = 1e-6,
        use_cross_attention: bool = False,
        cross_attention_max_scale: int = 3,
        num_heads: int = 8,
    ) -> None:
        """Initialize ConvNeXtV2AdaLNBlock.
        
        Args:
            dim: Number of feature channels.
            cond_dim: Dimension of conditioning vector c.
            mlp_ratio: Multiplier for hidden dimension in FFN.
            grn_eps: Epsilon for GRN.
            use_cross_attention: Whether to include ScaleAwareCrossAttention.
            cross_attention_max_scale: Maximum scale for cross-attention.
            num_heads: Number of attention heads for cross-attention.
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

        # Scale-aware cross-attention
        if use_cross_attention:
            self.cross_attn = ScaleAwareCrossAttention(
                dim=dim,
                context_dim=cond_dim,
                num_heads=num_heads,
                cross_attention_max_scale=cross_attention_max_scale,
            )
        else:
            self.cross_attn = None

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        text_sequence: Optional[torch.Tensor] = None,
        scale: Optional[Union[int, float, torch.Tensor]] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Feature map tensor of shape (B, C, H, W).
            c: Conditioning tensor of shape (B, cond_dim).
            text_sequence: Optional text sequence tensor (B, L, D).
            scale: Optional scale level index [0..7].
            text_mask: Optional text padding mask.
            
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
        scale_mod, shift_mod, gate_mod = ada_params.chunk(3, dim=-1)
        scale_mod = scale_mod.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, dim)
        shift_mod = shift_mod.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, dim)
        gate_mod = gate_mod.unsqueeze(1).unsqueeze(1)    # (B, 1, 1, dim)

        x = self.norm(x)
        x = x * (1.0 + scale_mod) + shift_mod

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
        gate_mod = gate_mod.permute(0, 3, 1, 2)
        x = residual + gate_mod * x

        # 9. Spatial cross-attention (if configured)
        if self.cross_attn is not None:
            x = self.cross_attn(
                x,
                text_sequence=text_sequence,
                scale=scale,
                text_mask=text_mask,
            )

        return x


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
        use_cross_attention: bool = True,
        cross_attention_max_scale: int = 3,
        cross_attention_heads: int = 8,
        gradient_checkpointing: bool = False,
    ) -> None:
        """Initialize ScaleFlowDecoder.
        
        Args:
            in_channels: Number of input image channels (e.g. 3 for RGB).
            out_channels: Number of output image channels (e.g. 3 for RGB).
            hidden_dim: Number of hidden channels in ConvNeXt V2 blocks.
            cond_dim: Dimension of text and scale embeddings.
            depth: Number of ConvNeXt V2 blocks.
            mlp_ratio: FFN expansion ratio in ConvNeXt V2 blocks.
            use_cross_attention: Whether to use ScaleAwareCrossAttention in blocks.
            cross_attention_max_scale: Maximum scale for cross-attention execution (default: 3).
            cross_attention_heads: Attention heads for cross-attention.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.depth = depth
        self.use_cross_attention = use_cross_attention
        self.cross_attention_max_scale = cross_attention_max_scale
        self.cross_attention_heads = cross_attention_heads

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

        # ConvNeXt V2 blocks with AdaLN and optional ScaleAwareCrossAttention
        self.blocks = nn.ModuleList([
            ConvNeXtV2AdaLNBlock(
                dim=hidden_dim,
                cond_dim=cond_dim,
                mlp_ratio=mlp_ratio,
                use_cross_attention=use_cross_attention,
                cross_attention_max_scale=cross_attention_max_scale,
                num_heads=cross_attention_heads,
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

        self.gradient_checkpointing = gradient_checkpointing

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for lower VRAM usage."""
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False

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
        text_sequence: Optional[torch.Tensor] = None,
        scale: Optional[Union[int, float, torch.Tensor]] = None,
        text_mask: Optional[torch.Tensor] = None,
        upsample: bool = True,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input image/latent tensor of shape (B, in_channels, H, W).
            e_text: Text embedding of shape (B, cond_dim).
            e_scale: Scale embedding of shape (B, cond_dim).
            text_sequence: Optional text sequence features of shape (B, L, D).
            scale: Optional scale level index [0..7].
            text_mask: Optional text padding mask.
            upsample: If True, uses PixelShuffle(2x) upsampler to output (B, out_channels, 2H, 2W).
                     If False, outputs at same resolution (B, out_channels, H, W).
                     
        Returns:
            Output image tensor at 2x resolution (if upsample=True) or 1x resolution.
        """
        # Condition vector c = MLP(e_text + e_scale)
        c = self.compute_conditioning(e_text, e_scale)

        # Project input image to feature space
        h = self.in_proj(x)

        # Pass through ConvNeXt V2 blocks with AdaLN & ScaleAwareCrossAttention
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                h = torch.utils.checkpoint.checkpoint(
                    block,
                    h,
                    c,
                    text_sequence,
                    scale,
                    text_mask,
                    use_reentrant=False,
                )
            else:
                h = block(
                    h,
                    c,
                    text_sequence=text_sequence,
                    scale=scale,
                    text_mask=text_mask,
                )

        h = self.final_norm(h)

        if upsample:
            out = self.upsampler(h)
        else:
            out = self.direct_head(h)

        return out

        return out
