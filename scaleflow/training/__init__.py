"""ScaleFlow Training Pipeline: Multiscale Optimal Transport Flow Matching (OT-CFM)."""

from .flow_matching import (
    DEFAULT_SCALE_RESOLUTIONS,
    FlowMatching,
    FlowMatchingBatch,
    MultiscaleFlowMatching,
    MultiscaleOTFlowMatching,
    build_image_pyramid,
)
from .dataset import (
    ScaleFlowDataset,
    SyntheticImageTextDataset,
    collate_multiscale,
    create_dataloader,
)
from .losses import (
    FlowMatchingLoss,
    MultiscaleFlowLoss,
    VelocityErrorMetric,
)
from .trainer import (
    ScaleFlowTrainer,
    TrainerConfig,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

__all__ = [
    "DEFAULT_SCALE_RESOLUTIONS",
    "FlowMatching",
    "FlowMatchingBatch",
    "MultiscaleFlowMatching",
    "MultiscaleOTFlowMatching",
    "build_image_pyramid",
    "ScaleFlowDataset",
    "SyntheticImageTextDataset",
    "collate_multiscale",
    "create_dataloader",
    "FlowMatchingLoss",
    "MultiscaleFlowLoss",
    "VelocityErrorMetric",
    "ScaleFlowTrainer",
    "TrainerConfig",
    "get_cosine_schedule_with_warmup",
    "get_linear_schedule_with_warmup",
]
