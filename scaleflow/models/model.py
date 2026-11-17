"""ScaleFlow integrated model: combines TextEncoder, ScaleEmbedding, and RecursiveDecoder.

Provides:
- ScaleFlowConfig: Dataclass for configuration serialization.
- ScaleFlowModel: Core PyTorch module with .save_pretrained(), .from_pretrained(),
  and parameter counting utilities.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import os
import torch
import torch.nn as nn

from .encoder import HybridTextEncoder
from .decoder import ScaleFlowDecoder
from .scale_embed import (
    SinusoidalScaleEmbedding,
    scale_to_resolution,
    resolution_to_scale,
)


@dataclass
class ScaleFlowConfig:
    """Configuration dataclass for ScaleFlowModel."""
    # Text encoder config
    vocab_size: int = 4096
    text_dim: int = 512
    text_depth: int = 6
    text_heads: int = 8
    conv_kernel_size: int = 5
    conv_expansion: int = 2
    max_seq_len: int = 128
    text_dropout: float = 0.0

    # Scale embedding config
    max_scale: int = 8
    use_scale_mlp: bool = True

    # Decoder config
    in_channels: int = 3
    out_channels: int = 3
    decoder_dim: int = 256
    decoder_depth: int = 4
    decoder_mlp_ratio: float = 4.0
    use_cross_attention: bool = True
    cross_attention_max_scale: int = 3
    cross_attention_heads: int = 8
    gradient_checkpointing: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScaleFlowConfig":
        """Instantiate configuration from dictionary, ignoring unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def save_json(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "ScaleFlowConfig":
        """Load configuration from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


class ScaleFlowModel(nn.Module):
    """Integrated ScaleFlow model.
    
    Combines:
    - HybridTextEncoder (LFM2-style 1D gated conv + self-attention in 2:1 ratio)
    - SinusoidalScaleEmbedding (discrete scale levels 0..7 -> R^D)
    - ScaleFlowDecoder (ConvNeXt V2 blocks with AdaLN conditioned on c = MLP(e_text + e_scale),
      and PixelShuffle(2x) upsampler head)
    """

    def __init__(self, config: Optional[ScaleFlowConfig] = None, **kwargs: Any) -> None:
        """Initialize ScaleFlowModel.
        
        Args:
            config: ScaleFlowConfig instance. If None, default config is used
                    overridden by any keyword arguments.
            **kwargs: Overrides for configuration parameters.
        """
        super().__init__()
        if config is None:
            config = ScaleFlowConfig(**kwargs)
        elif kwargs:
            cfg_dict = config.to_dict()
            cfg_dict.update(kwargs)
            config = ScaleFlowConfig.from_dict(cfg_dict)

        self.config = config

        # 1. Text Encoder: produces e_text in R^{text_dim}
        self.text_encoder = HybridTextEncoder(
            vocab_size=config.vocab_size,
            d_model=config.text_dim,
            depth=config.text_depth,
            num_heads=config.text_heads,
            conv_kernel_size=config.conv_kernel_size,
            conv_expansion=config.conv_expansion,
            max_seq_len=config.max_seq_len,
            dropout=config.text_dropout,
        )

        # 2. Scale Embedding: maps s in [0..7] to R^{text_dim}
        self.scale_embed = SinusoidalScaleEmbedding(
            dim=config.text_dim,
            max_scale=config.max_scale,
            use_mlp=config.use_scale_mlp,
        )

        # 3. Recursive Decoder: ConvNeXt V2 with AdaLN conditioned on c = MLP(e_text + e_scale)
        # and optional ScaleAwareCrossAttention
        self.decoder = ScaleFlowDecoder(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            hidden_dim=config.decoder_dim,
            cond_dim=config.text_dim,
            depth=config.decoder_depth,
            mlp_ratio=config.decoder_mlp_ratio,
            use_cross_attention=config.use_cross_attention,
            cross_attention_max_scale=config.cross_attention_max_scale,
            cross_attention_heads=getattr(config, "cross_attention_heads", 8),
            gradient_checkpointing=getattr(config, "gradient_checkpointing", False),
        )

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_sequence: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Encode tokenized text into global pooled embedding e_text (and optional sequence).
        
        Args:
            input_ids: Token ID tensor of shape (B, L).
            attention_mask: Optional mask (B, L) with 1 for valid tokens and 0 for padding.
            return_sequence: Whether to return sequence features (B, L, D) along with e_text.
            
        Returns:
            e_text tensor of shape (B, text_dim), or tuple of (e_text, seq_features).
        """
        return self.text_encoder(
            input_ids,
            attention_mask=attention_mask,
            return_sequence=return_sequence,
        )

    def encode_prompts(
        self,
        prompts: Union[str, List[str]],
        tokenizer: Any,
        device: Optional[torch.device] = None,
        return_sequence: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        """Convenience method to encode raw text string prompt(s) using a tokenizer.
        
        Args:
            prompts: String or list of string prompts.
            tokenizer: BPETokenizer instance (supports batch call returning BatchEncoding).
            device: Optional device to place tensors on.
            return_sequence: Whether to return sequence features and attention mask.
            
        Returns:
            e_text tensor of shape (B, text_dim) or (e_text, text_sequence, attention_mask).
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        batch = tokenizer(prompts, padding=True, return_tensors="pt")
        target_device = device if device is not None else next(self.parameters()).device
        input_ids = batch.input_ids.to(target_device)
        attention_mask = batch.attention_mask.to(target_device) if batch.attention_mask is not None else None

        if return_sequence:
            e_text, text_sequence = self.encode_text(
                input_ids, attention_mask=attention_mask, return_sequence=True
            )
            return e_text, text_sequence, attention_mask

        return self.encode_text(input_ids, attention_mask=attention_mask)

    def embed_scale(
        self,
        scale: Union[int, float, torch.Tensor],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Compute scale embedding e_scale for discrete scale level(s) [0..7].
        
        Args:
            scale: Scale level s in [0..7], as an integer, float, or Tensor.
            device: Optional torch.device.
            
        Returns:
            e_scale tensor of shape (B, text_dim).
        """
        return self.scale_embed(scale, device=device)

    def forward(
        self,
        x: torch.Tensor,
        scale: Optional[Union[int, float, torch.Tensor]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        e_text: Optional[torch.Tensor] = None,
        e_scale: Optional[torch.Tensor] = None,
        text_sequence: Optional[torch.Tensor] = None,
        upsample: bool = True,
    ) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input image tensor of shape (B, C, H, W).
            scale: Scale level index [0..7] (used if e_scale is None).
            input_ids: Token IDs of shape (B, L) (used if e_text is None).
            attention_mask: Optional attention mask (B, L).
            e_text: Optional precomputed text embedding of shape (B, text_dim).
            e_scale: Optional precomputed scale embedding of shape (B, text_dim).
            text_sequence: Optional precomputed text sequence features of shape (B, L, text_dim).
            upsample: If True, outputs 2x resolution image (B, out_channels, 2H, 2W).
                     If False, outputs 1x resolution image (B, out_channels, H, W).
                     
        Returns:
            Output image tensor at 2x (or 1x) resolution.
        """
        # Resolve e_text and text_sequence
        if input_ids is not None:
            if e_text is None or (self.config.use_cross_attention and text_sequence is None):
                e_text_enc, text_seq_enc = self.text_encoder(
                    input_ids, attention_mask=attention_mask, return_sequence=True
                )
                if e_text is None:
                    e_text = e_text_enc
                if text_sequence is None:
                    text_sequence = text_seq_enc
        elif e_text is None:
            raise ValueError("Either e_text or input_ids must be provided to forward().")

        # Resolve e_scale
        if e_scale is None:
            if scale is None:
                raise ValueError("Either e_scale or scale must be provided to forward().")
            e_scale = self.embed_scale(scale, device=x.device)

        # Pass through recursive decoder with scale-aware cross-attention
        return self.decoder(
            x,
            e_text=e_text,
            e_scale=e_scale,
            text_sequence=text_sequence,
            scale=scale,
            text_mask=attention_mask,
            upsample=upsample,
        )

    def recursive_step(
        self,
        x: torch.Tensor,
        current_scale: int,
        e_text: torch.Tensor,
        text_sequence: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Perform one recursive upsampling step from current_scale to current_scale + 1.
        
        Args:
            x: Image tensor at current scale (B, C, H, W).
            current_scale: Integer scale index in [0..6].
            e_text: Text conditioning tensor (B, text_dim).
            text_sequence: Optional sequence features for cross-attention.
            text_mask: Optional text padding mask.
            
        Returns:
            Image tensor at 2x resolution (B, C, 2H, 2W).
        """
        e_scale = self.embed_scale(current_scale, device=x.device)
        return self.decoder(
            x,
            e_text=e_text,
            e_scale=e_scale,
            text_sequence=text_sequence,
            scale=current_scale,
            text_mask=text_mask,
            upsample=True,
        )

    def generate_pyramid(
        self,
        x_init: torch.Tensor,
        e_text: torch.Tensor,
        text_sequence: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        start_scale: int = 0,
        num_steps: int = 7,
    ) -> List[torch.Tensor]:
        """Generate a full multi-resolution image pyramid recursively.
        
        Starting from x_init at start_scale (e.g. 8x8), recursively calls
        the decoder up to (start_scale + num_steps) resolution.
        
        Args:
            x_init: Initial image tensor (B, C, H, W), e.g. at 8x8.
            e_text: Text conditioning tensor of shape (B, text_dim).
            text_sequence: Optional text sequence for cross-attention.
            text_mask: Optional text padding mask.
            start_scale: Starting scale index (0 corresponds to 8x8).
            num_steps: Number of recursive upsampling steps (default: 7 -> up to 1024x1024).
            
        Returns:
            List of image tensors [x_init, x_16, x_32, ..., x_1024].
        """
        pyramid = [x_init]
        curr = x_init
        for s in range(start_scale, start_scale + num_steps):
            curr = self.recursive_step(
                curr,
                current_scale=s,
                e_text=e_text,
                text_sequence=text_sequence,
                text_mask=text_mask,
            )
            pyramid.append(curr)
        return pyramid
        for s in range(start_scale, start_scale + num_steps):
            curr = self.recursive_step(curr, current_scale=s, e_text=e_text)
            pyramid.append(curr)
        return pyramid

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing in the decoder for ultra low VRAM footprint."""
        self.decoder.enable_gradient_checkpointing()
        self.config.gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        """Disable gradient checkpointing."""
        self.decoder.disable_gradient_checkpointing()
        self.config.gradient_checkpointing = False

    def num_parameters(self, only_trainable: bool = False) -> int:
        """Return total or trainable parameter count.
        
        Args:
            only_trainable: If True, count only parameters with requires_grad=True.
            
        Returns:
            Number of parameters.
        """
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def parameter_breakdown(self) -> Dict[str, int]:
        """Return parameter count breakdown by component."""
        return {
            "text_encoder": sum(p.numel() for p in self.text_encoder.parameters()),
            "scale_embed": sum(p.numel() for p in self.scale_embed.parameters()),
            "decoder": sum(p.numel() for p in self.decoder.parameters()),
            "total": self.num_parameters(only_trainable=False),
        }

    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        """Save model weights and configuration to a directory.
        
        Args:
            save_directory: Path to output directory.
        """
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save config
        config_path = save_dir / "config.json"
        self.config.save_json(config_path)

        # 2. Save weights
        weights_path = save_dir / "model.pt"
        torch.save(self.state_dict(), weights_path)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        map_location: Union[str, torch.device] = "cpu",
    ) -> "ScaleFlowModel":
        """Load a ScaleFlowModel from a directory containing config.json and model.pt.
        
        Args:
            pretrained_model_name_or_path: Directory path containing saved model.
            map_location: Target device for loaded weights.
            
        Returns:
            Loaded ScaleFlowModel in eval mode.
        """
        model_dir = Path(pretrained_model_name_or_path)
        config_path = model_dir / "config.json"
        weights_path = model_dir / "model.pt"

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        if not weights_path.exists():
            # Check for alternative filename
            alt_path = model_dir / "pytorch_model.bin"
            if alt_path.exists():
                weights_path = alt_path
            else:
                raise FileNotFoundError(f"Weights file not found at {weights_path}")

        config = ScaleFlowConfig.from_json(config_path)
        model = cls(config=config)
        state_dict = torch.load(weights_path, map_location=map_location)
        model.load_state_dict(state_dict)
        model.eval()
        return model
