"""
ScaleFlowPipeline: End-to-end user-facing generation pipeline for ScaleFlow.

Provides high-level text-to-image generation:
- `.from_pretrained()` to load checkpoints
- `.to(device)` for GPU/CPU placement
- `__call__(prompt, steps=7, seed=None)` returning PIL Images
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn
from PIL import Image

from scaleflow.models.model import ScaleFlowModel, ScaleFlowConfig
from scaleflow.tokenizer.bpe import BPETokenizer
from scaleflow.pipeline.sampler import MultiscaleFlowSampler
from scaleflow.pipeline.utils import tensor_to_pil


class ScaleFlowPipeline:
    """User-facing inference pipeline for ScaleFlow text-to-image generation.
    
    Combines BPETokenizer, ScaleFlowModel, and MultiscaleFlowSampler into a seamless
    API returning ready-to-use PIL Images.
    """

    def __init__(
        self,
        model: ScaleFlowModel,
        tokenizer: Optional[BPETokenizer] = None,
        sampler: Optional[MultiscaleFlowSampler] = None,
    ) -> None:
        """Initialize ScaleFlowPipeline.
        
        Args:
            model: Trained ScaleFlowModel.
            tokenizer: BPETokenizer instance. If None, an empty tokenizer is initialized.
            sampler: MultiscaleFlowSampler instance. If None, initialized from model.
        """
        self.model = model
        self.tokenizer = tokenizer if tokenizer is not None else BPETokenizer()
        self.sampler = sampler if sampler is not None else MultiscaleFlowSampler(model=model)

    @property
    def device(self) -> torch.device:
        """Return the device on which the model parameters reside."""
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def to(
        self,
        device: Union[str, torch.device],
        dtype: Optional[torch.dtype] = None,
    ) -> "ScaleFlowPipeline":
        """Move pipeline models to the specified device and optional dtype.
        
        Args:
            device: Target torch device ('cpu', 'cuda', etc.).
            dtype: Optional torch dtype (e.g. torch.float16, torch.float32).
            
        Returns:
            self
        """
        self.model = self.model.to(device=device, dtype=dtype)
        self.sampler.model = self.model
        return self

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        map_location: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ) -> "ScaleFlowPipeline":
        """Load pipeline from a local directory or checkpoint path.
        
        Looks for:
        - `config.json` and `model.pt` (loaded into ScaleFlowModel)
        - `tokenizer.json` (loaded into BPETokenizer if present)
        
        Args:
            pretrained_model_name_or_path: Path to model directory.
            map_location: Device to map model weights to.
            **kwargs: Extra arguments passed to ScaleFlowModel.
            
        Returns:
            Instantiated and configured ScaleFlowPipeline.
        """
        model_dir = Path(pretrained_model_name_or_path)

        # 1. Load model
        model = ScaleFlowModel.from_pretrained(model_dir, map_location=map_location)

        # 2. Load tokenizer if present
        tokenizer_path = model_dir / "tokenizer.json"
        if tokenizer_path.exists():
            tokenizer = BPETokenizer.load(tokenizer_path)
        else:
            tokenizer = BPETokenizer()

        # 3. Create sampler
        sampler = MultiscaleFlowSampler(model=model)

        return cls(model=model, tokenizer=tokenizer, sampler=sampler)

    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        """Save model and tokenizer to a directory.
        
        Args:
            save_directory: Destination folder.
        """
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save_pretrained(save_dir)

        # Save tokenizer
        if self.tokenizer is not None and len(self.tokenizer.vocab) > 0:
            self.tokenizer.save(save_dir / "tokenizer.json")

    def encode_prompts(
        self,
        prompts: Union[str, List[str]],
    ) -> torch.Tensor:
        """Encode text prompt(s) into pooled conditioning vector(s) e_text.
        
        Args:
            prompts: Single string prompt or list of string prompts.
            
        Returns:
            Conditioning tensor of shape (B, D) on pipeline device.
        """
        if isinstance(prompts, str):
            prompt_list = [prompts]
        else:
            prompt_list = list(prompts)

        # Tokenize batch
        encoded = self.tokenizer.batch_encode(
            prompt_list,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded.input_ids.to(self.device)
        attention_mask = encoded.attention_mask.to(self.device)

        # Encode text to e_text (B, D)
        return self.model.encode_text(input_ids, attention_mask=attention_mask)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        steps: int = 7,
        seed: Optional[int] = None,
        guidance_scale: float = 1.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_all_scales: bool = False,
        **kwargs: Any,
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, List[torch.Tensor]]:
        """Generate images conditioned on text prompts.
        
        Args:
            prompt: Text prompt or list of prompts describing desired image.
            steps: Number of recursive scale steps (default: 7 -> 8x8 up to 1024x1024).
            seed: Optional random seed for reproducible generation.
            guidance_scale: Classifier-free guidance scale (default: 1.0 = disabled).
            negative_prompt: Optional negative prompt(s) when guidance_scale > 1.0.
            generator: Optional PyTorch random generator.
            latents: Optional initial noise tensor (B, C, 8, 8).
            output_type: 'pil' to return PIL Image(s), 'pt' / 'tensor' for raw tensors.
            return_all_scales: If True, returns images at all generated scale resolutions.
            **kwargs: Extra parameters passed to the sampler.
            
        Returns:
            A PIL Image (if single prompt and output_type='pil'),
            a list of PIL Images (if list of prompts),
            or torch.Tensor(s) (if output_type='pt').
        """
        is_single_prompt = isinstance(prompt, str)
        if is_single_prompt:
            prompt_list = [prompt]
        else:
            prompt_list = list(prompt)

        # Setup generator if seed provided
        if seed is not None and generator is None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # 1. Encode conditioning
        prompt_embeds = self.encode_prompts(prompt_list)

        # 2. Encode negative conditioning for CFG if enabled
        negative_prompt_embeds = None
        if guidance_scale > 1.0:
            if negative_prompt is None:
                neg_list = ["" for _ in prompt_list]
            elif isinstance(negative_prompt, str):
                neg_list = [negative_prompt] * len(prompt_list)
            else:
                neg_list = list(negative_prompt)
            negative_prompt_embeds = self.encode_prompts(neg_list)

        # 3. Run recursive multiscale sampling
        output = self.sampler.sample(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            seed=seed,
            latents=latents,
            return_pyramid=return_all_scales,
            **kwargs,
        )

        # 4. Format outputs
        if output_type.lower() in ("pt", "tensor"):
            return output

        if return_all_scales:
            # output is a list of tensors [x_0, x_1, ..., x_final]
            pyramid_images = []
            for t in output:
                pil_imgs = tensor_to_pil(t, value_range="symmetric")
                if is_single_prompt and isinstance(pil_imgs, list):
                    pyramid_images.append(pil_imgs[0])
                else:
                    pyramid_images.append(pil_imgs)
            return pyramid_images

        # Single scale final image
        pil_images = tensor_to_pil(output, value_range="symmetric")
        if is_single_prompt and isinstance(pil_images, list):
            return pil_images[0]
        return pil_images

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Alias for `__call__`."""
        return self(*args, **kwargs)
