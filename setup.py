#!/usr/bin/env python3
"""Setup script for speech separation project."""

import os
import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("Setting up Speech Separation Project")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("Error: Python 3.10 or higher is required")
        sys.exit(1)
    
    print(f"Python version: {sys.version}")
    
    # Create necessary directories
    directories = [
        "data/wav",
        "data/meta", 
        "checkpoints",
        "logs",
        "assets",
        "evaluation_results"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Install dependencies
    print("\nInstalling dependencies...")
    
    # Install core dependencies
    if not run_command("pip install -r requirements.txt", "Installing core dependencies"):
        print("Failed to install core dependencies")
        sys.exit(1)
    
    # Install development dependencies (optional)
    if "--dev" in sys.argv:
        if not run_command("pip install -e .[dev]", "Installing development dependencies"):
            print("Warning: Failed to install development dependencies")
    
    # Generate synthetic data
    print("\nGenerating synthetic data...")
    if not run_command(
        "python scripts/train.py --config configs/conv_tasnet.yaml --generate_data --n_samples 100",
        "Generating synthetic data"
    ):
        print("Warning: Failed to generate synthetic data")
    
    # Run tests
    if "--test" in sys.argv:
        print("\nRunning tests...")
        if not run_command("pytest tests/ -v", "Running tests"):
            print("Warning: Some tests failed")
    
    # Setup pre-commit hooks (if dev dependencies installed)
    if "--dev" in sys.argv:
        print("\nSetting up pre-commit hooks...")
        if not run_command("pre-commit install", "Installing pre-commit hooks"):
            print("Warning: Failed to install pre-commit hooks")
    
    print("\n" + "=" * 50)
    print("Setup completed successfully!")
    print("\nNext steps:")
    print("1. Train a model: python scripts/train.py --config configs/conv_tasnet.yaml")
    print("2. Evaluate model: python scripts/evaluate.py --config configs/conv_tasnet.yaml --checkpoint checkpoints/conv_tasnet/best.pth")
    print("3. Run demo: streamlit run demo/app.py")
    print("4. Compare methods: python scripts/compare_methods.py")
    print("\nFor more information, see README.md")


if __name__ == "__main__":
    main()
