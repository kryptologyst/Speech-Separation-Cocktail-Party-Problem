"""Training script for speech separation models."""

import argparse
import os
import yaml
import torch
from omegaconf import OmegaConf

from speech_separation.data import SyntheticDataGenerator, create_data_loaders
from speech_separation.train import create_trainer
from speech_separation.utils import set_seed, get_device


def main():
    parser = argparse.ArgumentParser(description="Train speech separation model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--model", type=str, choices=["conv_tasnet", "dprnn"], 
                       help="Model type (overrides config)")
    parser.add_argument("--data_dir", type=str, default="data", 
                       help="Data directory")
    parser.add_argument("--meta_file", type=str, default="data/meta/synthetic_metadata.csv",
                       help="Metadata file")
    parser.add_argument("--generate_data", action="store_true",
                       help="Generate synthetic data before training")
    parser.add_argument("--n_samples", type=int, default=1000,
                       help="Number of synthetic samples to generate")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cuda, mps, cpu)")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load config
    config = OmegaConf.load(args.config)
    
    # Override config with command line arguments
    if args.model:
        config.model_name = args.model
    if args.device != "auto":
        config.device = args.device
    
    # Determine device
    if config.device == "auto":
        device = get_device()
    else:
        device = torch.device(config.device)
    
    print(f"Using device: {device}")
    print(f"Config: {config}")
    
    # Generate synthetic data if requested
    if args.generate_data:
        print("Generating synthetic data...")
        generator = SyntheticDataGenerator(
            output_dir=args.data_dir,
            sample_rate=config.data.sample_rate,
            duration_range=(2.0, 8.0),
            snr_range=(0.0, 20.0),
            n_sources=config.data.n_sources,
        )
        generator.generate_sine_waves(n_samples=args.n_samples)
        generator.generate_tts_samples(n_samples=args.n_samples // 2)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=args.data_dir,
        meta_file=args.meta_file,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        sample_rate=config.data.sample_rate,
        segment_length=config.data.segment_length,
        n_sources=config.data.n_sources,
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create trainer
    model_name = getattr(config, 'model_name', 'conv_tasnet')
    trainer = create_trainer(
        model_name=model_name,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()
    
    print("Training completed!")


if __name__ == "__main__":
    main()
