"""Hybrid LFM2-style text encoder for ScaleFlow.

Combines 1D Depthwise Gated Convolutions and Multi-Head Self-Attention
in a 2:1 ratio with global pooling to produce dense text conditioning e_text in R^D.
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseGatedConvBlock(nn.Module):
    """1D Depthwise Gated Convolution Block.
    
    Applies LayerNorm, in-projection to gated channels, depthwise 1D conv
    along sequence length, gated activation (GLU with GELU), and out-projection.
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 5,
        expansion: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.hidden_dim = d_model * expansion

        self.norm = nn.LayerNorm(d_model)
        # Project to 2 * hidden_dim for gating (e.g. value and gate)
        self.in_proj = nn.Linear(d_model, self.hidden_dim * 2)
        self.conv1d = nn.Conv1d(
            in_channels=self.hidden_dim * 2,
            out_channels=self.hidden_dim * 2,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=self.hidden_dim * 2,
            bias=True,
        )
        self.act = nn.GELU()
        self.out_proj = nn.Linear(self.hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, L, D).
            key_padding_mask: Optional boolean mask (B, L) where True indicates padding.
            
        Returns:
            Output tensor of shape (B, L, D).
        """
        residual = x
        x_norm = self.norm(x)

        # In-projection -> (B, L, 2 * hidden_dim)
        projected = self.in_proj(x_norm)

        # Zero out padding positions before conv to avoid leakage
        if key_padding_mask is not None:
            # key_padding_mask is True for padding tokens
            projected = projected.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        # Conv1d expects (B, Channels, Length)
        conv_in = projected.transpose(1, 2)
        conv_out = self.conv1d(conv_in)
        conv_out = conv_out.transpose(1, 2)  # (B, L, 2 * hidden_dim)

        # Gated activation: x1 * GELU(x2)
        val, gate = conv_out.chunk(2, dim=-1)
        gated = val * self.act(gate)

        out = self.out_proj(gated)
        out = self.dropout(out)

        return residual + out


class MultiHeadSelfAttentionBlock(nn.Module):
    """Multi-Head Self-Attention Block with Pre-LayerNorm and FFN."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(d_model)
        mlp_hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, L, D).
            key_padding_mask: Optional boolean mask (B, L) where True indicates padding.
            
        Returns:
            Output tensor of shape (B, L, D).
        """
        # Pre-LN Self Attention
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(
            norm_x,
            norm_x,
            norm_x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_out

        # Pre-LN MLP
        x = x + self.mlp(self.norm2(x))
        return x


class HybridTextEncoder(nn.Module):
    """Hybrid LFM2-style text encoder.
    
    Stacks 1D Depthwise Gated Convolutions and Multi-Head Self-Attention in
    a 2:1 ratio (2 Conv blocks for every 1 Attention block).
    Outputs a globally pooled embedding e_text in R^D.
    """

    def __init__(
        self,
        vocab_size: int = 4096,
        d_model: int = 512,
        depth: int = 6,
        num_heads: int = 8,
        conv_kernel_size: int = 5,
        conv_expansion: int = 2,
        mlp_ratio: float = 4.0,
        max_seq_len: int = 128,
        padding_idx: int = 0,
        dropout: float = 0.0,
    ) -> None:
        """Initialize HybridTextEncoder.
        
        Args:
            vocab_size: Vocabulary size.
            d_model: Embedding dimension (D).
            depth: Total number of transformer / conv layers (default: 6 -> 4 conv, 2 attn).
            num_heads: Number of attention heads.
            conv_kernel_size: Kernel size for 1D depthwise conv.
            conv_expansion: Channel expansion factor in gated conv.
            mlp_ratio: FFN hidden dimension multiplier.
            max_seq_len: Maximum input sequence length.
            padding_idx: Index of pad token.
            dropout: Dropout probability.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.depth = depth
        self.max_seq_len = max_seq_len
        self.padding_idx = padding_idx

        # Embeddings
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # Build layers with 2:1 Conv to Attention ratio:
        # Layer 0: Conv, Layer 1: Conv, Layer 2: Attention, Layer 3: Conv, ...
        self.layers = nn.ModuleList()
        for i in range(depth):
            if (i % 3) < 2:
                # 2 Conv layers
                layer = DepthwiseGatedConvBlock(
                    d_model=d_model,
                    kernel_size=conv_kernel_size,
                    expansion=conv_expansion,
                    dropout=dropout,
                )
            else:
                # 1 Attention layer
                layer = MultiHeadSelfAttentionBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
            self.layers.append(layer)

        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_sequence: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.
        
        Args:
            input_ids: Token ID tensor of shape (B, L).
            attention_mask: Mask of shape (B, L) where 1 indicates valid token
                           and 0 indicates padding token (or bool tensor).
            return_sequence: Whether to return sequence features as well.
            
        Returns:
            e_text: Global pooled text embedding of shape (B, D).
            (optional) seq_features: Sequence features of shape (B, L, D).
        """
        B, L = input_ids.shape
        if L > self.max_seq_len:
            raise ValueError(f"Sequence length {L} exceeds max_seq_len {self.max_seq_len}")

        # Positions: (1, L)
        device = input_ids.device
        positions = torch.arange(0, L, dtype=torch.long, device=device).unsqueeze(0)

        x = self.token_embed(input_ids) + self.pos_embed(positions)
        x = self.embed_dropout(x)

        # Prepare key_padding_mask: (B, L) where True indicates padding
        key_padding_mask = None
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                # If attention_mask has True for valid, invert for key_padding_mask
                # (PyTorch MultiheadAttention expects True for padding tokens)
                key_padding_mask = ~attention_mask
            else:
                key_padding_mask = (attention_mask == 0)

        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)

        x = self.final_norm(x)

        # Global pooling to produce e_text in R^D
        if attention_mask is not None:
            # Weighted mean pooling over valid tokens
            if attention_mask.dtype == torch.bool:
                valid_mask = attention_mask.float().unsqueeze(-1)
            else:
                valid_mask = attention_mask.to(dtype=torch.float32).unsqueeze(-1)

            pooled = (x * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)
        else:
            pooled = x.mean(dim=1)

        if return_sequence:
            return pooled, x
        return pooled
