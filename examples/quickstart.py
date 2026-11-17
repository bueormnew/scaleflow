"""
ScaleFlow Quickstart Example.

Demonstrates:
1. Creating and configuring a ScaleFlowModel and BPETokenizer.
2. Initializing ScaleFlowPipeline.
3. Encoding prompts and generating images with Multiscale Flow Matching.
4. Generating multi-resolution pyramids and batch image grids.
5. Saving generated images to disk.
6. Pipeline checkpoint saving and loading (.save_pretrained / .from_pretrained).
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from scaleflow import (
    ScaleFlowModel,
    ScaleFlowConfig,
    ScaleFlowPipeline,
    BPETokenizer,
    save_image,
    make_image_grid,
)


def main():
    print("=" * 70)
    print(" ScaleFlow: Recursive Multiscale Flow Matching Quickstart")
    print("=" * 70)

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize BPETokenizer with sample prompts
    print("\n[1/6] Setting up BPETokenizer...")
    sample_texts = [
        "a vibrant mountain landscape at golden hour with reflective lake",
        "a futuristic cybernetic cat in neon city",
        "a cozy wooden cabin in snowy pine forest",
        "a minimalist architectural modern house with warm lighting",
        "a serene tropical beach with crystal turquoise water",
        "an abstract oil painting with bold brushstrokes and vivid colors",
    ]
    tokenizer = BPETokenizer()
    tokenizer.train(sample_texts, vocab_size=256)
    print(f"      Tokenizer trained with vocabulary size: {len(tokenizer.vocab)}")

    # 2. Configure lightweight model for rapid inference
    print("\n[2/6] Initializing ScaleFlowModel...")
    config = ScaleFlowConfig(
        vocab_size=max(256, len(tokenizer.vocab)),
        text_dim=128,
        text_depth=3,
        text_heads=4,
        decoder_dim=64,
        decoder_depth=3,
        decoder_mlp_ratio=2.0,
    )
    model = ScaleFlowModel(config)
    param_info = model.parameter_breakdown()
    print(f"      Model initialized. Total parameters: {param_info['total']:,}")
    print(f"      - Text Encoder: {param_info['text_encoder']:,}")
    print(f"      - Scale Embedding: {param_info['scale_embed']:,}")
    print(f"      - Recursive Decoder: {param_info['decoder']:,}")

    # 3. Create pipeline
    print("\n[3/6] Building ScaleFlowPipeline...")
    pipeline = ScaleFlowPipeline(model=model, tokenizer=tokenizer)
    pipeline.to("cpu")
    print(f"      Pipeline ready on device: {pipeline.device}")

    # 4. Generate single image
    prompt = "a vibrant mountain landscape at golden hour with reflective lake"
    print(f"\n[4/6] Generating image for prompt: '{prompt}'...")
    # Using steps=4 (8 -> 16 -> 32 -> 64 -> 128) for swift execution
    image = pipeline(prompt=prompt, steps=4, seed=42)
    output_path = output_dir / "quickstart_single.png"
    save_image(image, output_path)
    print(f"      Image successfully generated: {image.size[0]}x{image.size[1]} px")
    print(f"      Saved to: {output_path}")

    # 5. Generate multiscale pyramid
    print("\n[5/6] Generating multiscale resolution pyramid...")
    pyramid = pipeline(
        prompt="a futuristic cybernetic cat in neon city",
        steps=4,
        return_all_scales=True,
        seed=101,
    )
    print(f"      Generated {len(pyramid)} scale levels:")
    for idx, scale_img in enumerate(pyramid):
        scale_path = output_dir / f"pyramid_scale_{idx}_{scale_img.size[0]}x{scale_img.size[1]}.png"
        save_image(scale_img, scale_path)
        print(f"      - Scale {idx}: {scale_img.size[0]}x{scale_img.size[1]} -> saved to {scale_path.name}")

    # 6. Batch generation and image grid
    print("\n[6/6] Generating batch of prompts and creating image grid...")
    batch_prompts = [
        "a cozy wooden cabin in snowy pine forest",
        "a minimalist architectural modern house with warm lighting",
        "a serene tropical beach with crystal turquoise water",
        "an abstract oil painting with bold brushstrokes and vivid colors",
    ]
    batch_images = pipeline(prompt=batch_prompts, steps=3, seed=2026)
    grid = make_image_grid(batch_images, rows=2, cols=2, padding=4)
    grid_path = output_dir / "quickstart_grid.png"
    save_image(grid, grid_path)
    print(f"      Batch generated 4 images at {batch_images[0].size[0]}x{batch_images[0].size[1]} px")
    print(f"      Created 2x2 grid ({grid.size[0]}x{grid.size[1]} px), saved to: {grid_path}")

    # Serialization demonstration
    checkpoint_dir = Path("outputs") / "demo_checkpoint"
    print(f"\n[Bonus] Saving pipeline checkpoint to {checkpoint_dir}...")
    pipeline.save_pretrained(checkpoint_dir)
    print("        Loading pipeline from checkpoint...")
    loaded_pipe = ScaleFlowPipeline.from_pretrained(checkpoint_dir)
    test_img = loaded_pipe("a test prompt", steps=2, seed=7)
    print(f"        Loaded pipeline successfully generated {test_img.size} image!")

    print("\n" + "=" * 70)
    print(" ScaleFlow Quickstart completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
