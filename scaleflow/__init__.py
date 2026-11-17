"""
ScaleFlow: Recursive Multiscale Flow Matching for Image Generation.
"""

__version__ = "0.1.0"

from scaleflow.pipeline import (
    ScaleFlowPipeline,
    MultiscaleFlowSampler,
    EulerFlowSampler,
    ScaleFlowSampler,
    tensor_to_pil,
    pil_to_tensor,
    make_image_grid,
    save_image,
)
from scaleflow.models import (
    ScaleFlowModel,
    ScaleFlowConfig,
    ScaleFlowDecoder,
    HybridTextEncoder,
    SinusoidalScaleEmbedding,
)
from scaleflow.tokenizer import BPETokenizer
from scaleflow.training import (
    ScaleFlowTrainer,
    TrainerConfig,
    MultiscaleOTFlowMatching,
    SyntheticImageTextDataset,
    resolve_device,
    get_device_info,
)

__all__ = [
    "__version__",
    # Pipeline / Inference
    "ScaleFlowPipeline",
    "MultiscaleFlowSampler",
    "EulerFlowSampler",
    "ScaleFlowSampler",
    "tensor_to_pil",
    "pil_to_tensor",
    "make_image_grid",
    "save_image",
    # Models
    "ScaleFlowModel",
    "ScaleFlowConfig",
    "ScaleFlowDecoder",
    "HybridTextEncoder",
    "SinusoidalScaleEmbedding",
    # Tokenizer
    "BPETokenizer",
    # Training
    "ScaleFlowTrainer",
    "TrainerConfig",
    "MultiscaleOTFlowMatching",
    "SyntheticImageTextDataset",
    # Device utilities
    "resolve_device",
    "get_device_info",
]
