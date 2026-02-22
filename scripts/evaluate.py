"""Evaluation script for speech separation models."""

import argparse
import os
import yaml
import torch
from omegaconf import OmegaConf

from speech_separation.data import create_data_loaders
from speech_separation.eval import evaluate_model
from speech_separation.utils import set_seed, get_device


def main():
    parser = argparse.ArgumentParser(description="Evaluate speech separation model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, 
                       help="Path to model checkpoint")
    parser.add_argument("--model", type=str, choices=["conv_tasnet", "dprnn"], 
                       help="Model type (overrides config)")
    parser.add_argument("--data_dir", type=str, default="data", 
                       help="Data directory")
    parser.add_argument("--meta_file", type=str, default="data/meta/synthetic_metadata.csv",
                       help="Metadata file")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for results")
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
    
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Evaluate model
    model_name = getattr(config, 'model_name', 'conv_tasnet')
    metrics = evaluate_model(
        model_name=model_name,
        checkpoint_path=args.checkpoint,
        test_loader=test_loader,
        config=config,
        device=device,
        output_dir=args.output_dir,
    )
    
    print("Evaluation completed!")
    print("Results:")
    for key, value in metrics.items():
        if key.endswith("_mean"):
            print(f"{key}: {value:.3f}")


if __name__ == "__main__":
    main()
