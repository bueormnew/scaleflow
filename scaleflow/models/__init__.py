"""ScaleFlow Models Module.

Contains:
- SinusoidalScaleEmbedding: Discrete scale levels [0..7] mapping to embedding vectors.
- HybridTextEncoder: LFM2-style 1D gated conv + self-attention in 2:1 ratio.
- ScaleFlowDecoder: Recursive ConvNeXt V2 with AdaLN and PixelShuffle(2x) upsampler.
- ScaleFlowModel: Full integrated model with .save_pretrained() and .from_pretrained().
- ScaleFlowConfig: Model configuration dataclass.
"""

from .scale_embed import (
    SinusoidalScaleEmbedding,
    SCALE_RESOLUTIONS,
    RESOLUTION_TO_SCALE,
    scale_to_resolution,
    resolution_to_scale,
)
from .encoder import (
    DepthwiseGatedConvBlock,
    MultiHeadSelfAttentionBlock,
    HybridTextEncoder,
)
from .decoder import (
    GRN,
    ConvNeXtV2AdaLNBlock,
    ScaleAwareCrossAttention,
    PixelShuffleUpsampler,
    ScaleFlowDecoder,
)
from .model import (
    ScaleFlowConfig,
    ScaleFlowModel,
)

__all__ = [
    # Scale Embedding
    "SinusoidalScaleEmbedding",
    "SCALE_RESOLUTIONS",
    "RESOLUTION_TO_SCALE",
    "scale_to_resolution",
    "resolution_to_scale",
    # Text Encoder
    "DepthwiseGatedConvBlock",
    "MultiHeadSelfAttentionBlock",
    "HybridTextEncoder",
    # Decoder
    "GRN",
    "ConvNeXtV2AdaLNBlock",
    "ScaleAwareCrossAttention",
    "PixelShuffleUpsampler",
    "ScaleFlowDecoder",
    # Integrated Model
    "ScaleFlowConfig",
    "ScaleFlowModel",
]
