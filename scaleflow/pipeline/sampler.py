"""
Multiscale Flow ODE/Euler sampler for ScaleFlow.

Implements recursive scale-stepping and Flow Matching ODE / Euler numerical
integration from base noise (8x8) up to high resolution (1024x1024).
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from scaleflow.models.scale_embed import SCALE_RESOLUTIONS, scale_to_resolution


class MultiscaleFlowSampler:
    """Multiscale Flow ODE/Euler sampler.
    
    Generates images by recursively stepping through scale levels [0..steps-1],
    starting from 8x8 Gaussian noise and progressively upsampling and refining
    through the flow matching vector field up to 1024x1024.
    """

    def __init__(
        self,
        model: nn.Module,
        base_resolution: int = 8,
        in_channels: int = 3,
        sigma_min: float = 1e-4,
    ) -> None:
        """Initialize MultiscaleFlowSampler.
        
        Args:
            model: ScaleFlowModel instance.
            base_resolution: Base spatial resolution for scale 0 (default: 8 for 8x8).
            in_channels: Number of image channels (default: 3 for RGB).
            sigma_min: Small standard deviation parameter from flow matching.
        """
        self.model = model
        self.base_resolution = base_resolution
        self.in_channels = in_channels
        self.sigma_min = sigma_min

    @torch.no_grad()
    def step_scale(
        self,
        x: torch.Tensor,
        scale: int,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        ode_steps: int = 1,
    ) -> torch.Tensor:
        """Perform recursive upsampling and ODE flow transition from scale s to s+1.
        
        Args:
            x: Image/state tensor at current scale (B, C, H, W).
            scale: Current scale index [0..6].
            prompt_embeds: Text conditioning tensor (B, D).
            negative_prompt_embeds: Optional unconditional text conditioning (B, D).
            guidance_scale: Classifier-free guidance weight (>= 1.0).
            ode_steps: Number of Euler ODE sub-steps (default: 1).
            
        Returns:
            Output image tensor at 2x resolution (B, C, 2H, 2W).
        """
        device = x.device
        dtype = x.dtype
        batch_size = x.shape[0]

        # Case 1: Direct 1-step recursive flow transition (most efficient, standard ScaleFlow)
        if ode_steps <= 1:
            if guidance_scale > 1.0 and negative_prompt_embeds is not None:
                # Combined batch for conditioned and unconditioned passes
                combined_x = torch.cat([x, x], dim=0)
                combined_embeds = torch.cat([prompt_embeds, negative_prompt_embeds], dim=0)
                
                # Get scale embedding
                e_scale = self.model.embed_scale(scale, device=device).to(dtype=dtype)
                combined_scale = torch.cat([e_scale, e_scale], dim=0)
                
                # Run decoder
                out = self.model.decoder(
                    combined_x,
                    e_text=combined_embeds,
                    e_scale=combined_scale,
                    upsample=True,
                )
                out_cond, out_uncond = out.chunk(2, dim=0)
                next_x = out_uncond + guidance_scale * (out_cond - out_uncond)
            else:
                e_scale = self.model.embed_scale(scale, device=device).to(dtype=dtype)
                next_x = self.model.decoder(
                    x,
                    e_text=prompt_embeds,
                    e_scale=e_scale,
                    upsample=True,
                )
            return torch.clamp(next_x, -1.0, 1.0)

        # Case 2: Multi-step Euler ODE integration within current scale, then upsample
        dt = 1.0 / ode_steps
        curr = x
        for step in range(ode_steps):
            t_val = step * dt
            # Model velocity prediction at 1x resolution
            e_scale = self.model.embed_scale(scale, device=device).to(dtype=dtype)
            if guidance_scale > 1.0 and negative_prompt_embeds is not None:
                v_cond = self.model.decoder(curr, e_text=prompt_embeds, e_scale=e_scale, upsample=False)
                v_uncond = self.model.decoder(curr, e_text=negative_prompt_embeds, e_scale=e_scale, upsample=False)
                v = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                v = self.model.decoder(curr, e_text=prompt_embeds, e_scale=e_scale, upsample=False)

            # Euler integration step
            curr = curr + dt * v

        # Finally, recursive step to next scale
        e_scale = self.model.embed_scale(scale, device=device).to(dtype=dtype)
        if guidance_scale > 1.0 and negative_prompt_embeds is not None:
            out_cond = self.model.decoder(curr, e_text=prompt_embeds, e_scale=e_scale, upsample=True)
            out_uncond = self.model.decoder(curr, e_text=negative_prompt_embeds, e_scale=e_scale, upsample=True)
            next_x = out_uncond + guidance_scale * (out_cond - out_uncond)
        else:
            next_x = self.model.decoder(curr, e_text=prompt_embeds, e_scale=e_scale, upsample=True)

        return torch.clamp(next_x, -1.0, 1.0)

    @torch.no_grad()
    def sample(
        self,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        steps: int = 7,
        start_scale: int = 0,
        ode_steps_per_scale: int = 1,
        guidance_scale: float = 1.0,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
        latents: Optional[torch.Tensor] = None,
        return_pyramid: bool = False,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Run full recursive multiscale flow sampling from 8x8 noise up to 1024x1024.
        
        Args:
            prompt_embeds: Text conditioning tensor of shape (B, D).
            negative_prompt_embeds: Optional unconditional conditioning of shape (B, D).
            steps: Number of recursive scale transitions (default: 7 -> 8x8 to 1024x1024).
            start_scale: Scale level index to start from (default: 0).
            ode_steps_per_scale: Euler sub-steps per scale level (default: 1).
            guidance_scale: Classifier-free guidance scale (>= 1.0).
            generator: Optional PyTorch random generator for reproducibility.
            seed: Optional integer seed (used if generator is None).
            latents: Optional initial noise tensor of shape (B, C, 8, 8).
            return_pyramid: If True, returns list of all intermediate scale tensors.
            callback: Optional callback(step_idx, current_scale, current_tensor).
            
        Returns:
            Final generated tensor (B, C, H_final, W_final) normalized in [-1, 1],
            or list of tensors at each scale if return_pyramid=True.
        """
        device = prompt_embeds.device
        dtype = prompt_embeds.dtype
        batch_size = prompt_embeds.shape[0]

        # Setup random generator if seed provided
        if seed is not None and generator is None:
            generator = torch.Generator(device=device).manual_seed(seed)

        # 1. Initialize base noise at scale 0 (8x8)
        base_res = scale_to_resolution(start_scale)
        if latents is None:
            curr = torch.randn(
                (batch_size, self.in_channels, base_res, base_res),
                generator=generator,
                device=device,
                dtype=dtype,
            )
        else:
            curr = latents.to(device=device, dtype=dtype)
            if curr.shape[0] != batch_size:
                curr = curr.repeat(batch_size, 1, 1, 1)

        pyramid: List[torch.Tensor] = [curr]

        # 2. Loop recursively through scales
        for step_idx in range(steps):
            scale_level = start_scale + step_idx
            if scale_level >= 7:
                # Reached maximum standard scale (1024x1024)
                break

            curr = self.step_scale(
                x=curr,
                scale=scale_level,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                guidance_scale=guidance_scale,
                ode_steps=ode_steps_per_scale,
            )
            pyramid.append(curr)

            if callback is not None:
                callback(step_idx, scale_level + 1, curr)

        if return_pyramid:
            return pyramid
        return curr


# Aliases for interface flexibility
EulerFlowSampler = MultiscaleFlowSampler
ScaleFlowSampler = MultiscaleFlowSampler
