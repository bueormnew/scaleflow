"""
ScaleFlow Toy Training Example.

Demonstrates:
1. Setting up a lightweight ScaleFlowModel and BPETokenizer.
2. Creating a synthetic multiscale image-text training dataset and DataLoader.
3. Configuring and running ScaleFlowTrainer with Multiscale OT-CFM.
4. Logging training metrics (flow matching MSE, velocity cosine similarity).
5. Saving checkpoint and loading into ScaleFlowPipeline for generation.
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
)
from scaleflow.training import (
    MultiscaleOTFlowMatching,
    SyntheticImageTextDataset,
    create_dataloader,
    ScaleFlowTrainer,
    TrainerConfig,
)


def main():
    print("=" * 70)
    print(" ScaleFlow: Toy Training Demonstration (OT-CFM)")
    print("=" * 70)

    checkpoint_dir = Path("outputs") / "toy_checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 1. Train a compact tokenizer on domain vocabulary
    print("\n[1/5] Training BPETokenizer on sample corpus...")
    training_prompts = [
        "a geometric circle on dark background",
        "a colorful radial pattern with gradients",
        "horizontal color stripes with texture",
        "concentric rings with alternating colors",
        "a bright centered square with glow",
        "diagonal light beams intersecting",
    ]
    tokenizer = BPETokenizer()
    tokenizer.train(training_prompts, vocab_size=128)
    print(f"      Tokenizer vocabulary: {len(tokenizer.vocab)} tokens")

    # 2. Initialize compact ScaleFlowModel
    print("\n[2/5] Creating lightweight ScaleFlowModel...")
    model_config = ScaleFlowConfig(
        vocab_size=max(128, len(tokenizer.vocab)),
        text_dim=64,
        text_depth=2,
        text_heads=2,
        decoder_dim=32,
        decoder_depth=2,
        decoder_mlp_ratio=2.0,
        max_scale=4,  # Scales 0..3 (8x8 to 64x64)
    )
    model = ScaleFlowModel(model_config)
    print(f"      Model parameter count: {model.num_parameters():,}")

    # 3. Create synthetic multiscale dataset and dataloaders
    print("\n[3/5] Generating synthetic image-text dataset...")
    train_dataset = SyntheticImageTextDataset(
        num_samples=32,
        image_size=64,
        tokenizer=tokenizer,
        max_seq_len=32,
        seed=42,
    )
    val_dataset = SyntheticImageTextDataset(
        num_samples=8,
        image_size=64,
        tokenizer=tokenizer,
        max_seq_len=32,
        seed=100,
    )
    train_loader = create_dataloader(train_dataset, batch_size=4, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=4, shuffle=False)
    print(f"      Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 4. Configure Multiscale OT Flow Matcher and Trainer
    print("\n[4/5] Initializing Multiscale OT-CFM Trainer...")
    flow_matcher = MultiscaleOTFlowMatching()
    trainer_config = TrainerConfig(
        learning_rate=2e-3,
        weight_decay=0.01,
        max_steps=15,
        min_scale=0,
        max_scale=3,
        log_every_n_steps=3,
        eval_every_n_steps=5,
        save_every_n_steps=15,
        checkpoint_dir=str(checkpoint_dir),
        warmup_steps=2,
        device="cpu",
        seed=42,
    )

    trainer = ScaleFlowTrainer(
        model=model,
        flow_matcher=flow_matcher,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=trainer_config,
        tokenizer=tokenizer,
    )

    print("      Starting training loop...")
    metrics = trainer.train()
    print(f"      Training complete! Final Train Loss: {metrics.get('train_loss', 0.0):.4f}")
    if "val_loss" in metrics:
        print(f"      Final Val Loss: {metrics['val_loss']:.4f}")

    # 5. Load trained model into ScaleFlowPipeline for inference
    print("\n[5/5] Loading trained checkpoint into ScaleFlowPipeline...")
    # Save complete pipeline package
    pipeline = ScaleFlowPipeline(model=model, tokenizer=tokenizer)
    pipeline.save_pretrained(checkpoint_dir)

    loaded_pipe = ScaleFlowPipeline.from_pretrained(checkpoint_dir)
    test_prompt = "a geometric circle on dark background"
    print(f"      Generating inference image for: '{test_prompt}'...")
    generated_img = loaded_pipe(test_prompt, steps=3, seed=777)

    output_path = checkpoint_dir / "toy_inference_sample.png"
    save_image(generated_img, output_path)
    print(f"      Inference output saved: {generated_img.size} to {output_path}")

    print("\n" + "=" * 70)
    print(" ScaleFlow Toy Training completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
