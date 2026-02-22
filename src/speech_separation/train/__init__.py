"""Training module for speech separation models."""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, Tuple
import numpy as np
from tqdm import tqdm

from ..models import ConvTasNet, DPRNN
from ..losses import CombinedLoss
from ..metrics import MetricCalculator
from ..utils import get_device, set_seed


class Trainer:
    """Trainer for speech separation models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        save_dir: str = "checkpoints",
        log_dir: str = "logs",
        max_epochs: int = 100,
        patience: int = 10,
        grad_clip: Optional[float] = None,
    ):
        """Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            loss_fn: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to use
            save_dir: Directory to save checkpoints
            log_dir: Directory for logs
            max_epochs: Maximum number of epochs
            patience: Early stopping patience
            grad_clip: Gradient clipping value
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or get_device()
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.max_epochs = max_epochs
        self.patience = patience
        self.grad_clip = grad_clip
        
        # Move model to device
        self.model.to(self.device)
        
        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize logging
        self.writer = SummaryWriter(log_dir)
        
        # Initialize metrics
        self.metric_calculator = MetricCalculator()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        
        print(f"Trainer initialized on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        loss_components = {}
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            mixture = batch["mixture"].to(self.device)
            sources = batch["sources"].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            estimates = self.model(mixture)
            
            # Compute loss
            if isinstance(self.loss_fn, CombinedLoss):
                loss, loss_dict = self.loss_fn(estimates, sources)
            else:
                loss = self.loss_fn(estimates, sources)
                loss_dict = {"total_loss": loss.item()}
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            for key, value in loss_dict.items():
                if key not in loss_components:
                    loss_components[key] = 0.0
                loss_components[key] += value
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss / (batch_idx + 1):.4f}"
            })
        
        # Average losses
        avg_loss = total_loss / num_batches
        avg_loss_components = {k: v / num_batches for k, v in loss_components.items()}
        
        return {
            "train_loss": avg_loss,
            **avg_loss_components
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        loss_components = {}
        all_metrics = {}
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            
            for batch_idx, batch in enumerate(pbar):
                # Move data to device
                mixture = batch["mixture"].to(self.device)
                sources = batch["sources"].to(self.device)
                
                # Forward pass
                estimates = self.model(mixture)
                
                # Compute loss
                if isinstance(self.loss_fn, CombinedLoss):
                    loss, loss_dict = self.loss_fn(estimates, sources)
                else:
                    loss = self.loss_fn(estimates, sources)
                    loss_dict = {"total_loss": loss.item()}
                
                # Accumulate losses
                total_loss += loss.item()
                for key, value in loss_dict.items():
                    if key not in loss_components:
                        loss_components[key] = 0.0
                    loss_components[key] += value
                
                # Compute metrics
                metrics = self.metric_calculator(estimates, sources)
                for key, values in metrics.items():
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].extend(values.flatten().cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Average losses
        avg_loss = total_loss / num_batches
        avg_loss_components = {k: v / num_batches for k, v in loss_components.items()}
        
        # Average metrics
        avg_metrics = {}
        for key, values in all_metrics.items():
            avg_metrics[f"{key}_mean"] = np.mean(values)
            avg_metrics[f"{key}_std"] = np.std(values)
        
        return {
            "val_loss": avg_loss,
            **avg_loss_components,
            **avg_metrics
        }
    
    def save_checkpoint(self, is_best: bool = False) -> None:
        """Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.save_dir, "latest.pth")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.save_dir, "best.pth")
            torch.save(checkpoint, best_path)
            print(f"New best model saved to {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        
        print(f"Checkpoint loaded from {checkpoint_path}")
    
    def train(self) -> None:
        """Train the model."""
        print("Starting training...")
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate_epoch()
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step(val_metrics["val_loss"])
            
            # Log metrics
            for key, value in train_metrics.items():
                self.writer.add_scalar(f"train/{key}", value, epoch)
            
            for key, value in val_metrics.items():
                self.writer.add_scalar(f"val/{key}", value, epoch)
            
            # Store losses
            self.train_losses.append(train_metrics["train_loss"])
            self.val_losses.append(val_metrics["val_loss"])
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{self.max_epochs}")
            print(f"Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"Val SI-SDR: {val_metrics.get('si_sdr_mean', 0):.2f} dB")
            print(f"Val PESQ: {val_metrics.get('pesq_mean', 0):.2f}")
            print(f"Val STOI: {val_metrics.get('stoi_mean', 0):.3f}")
            
            # Check for improvement
            is_best = val_metrics["val_loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["val_loss"]
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(is_best)
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break
        
        # Training completed
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f} seconds")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Close writer
        self.writer.close()


def create_trainer(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict,
    device: Optional[torch.device] = None,
) -> Trainer:
    """Create a trainer with specified configuration.
    
    Args:
        model_name: Name of the model ("conv_tasnet" or "dprnn")
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        device: Device to use
        
    Returns:
        Configured trainer
    """
    device = device or get_device()
    
    # Create model
    if model_name.lower() == "conv_tasnet":
        model = ConvTasNet(**config["model"])
    elif model_name.lower() == "dprnn":
        model = DPRNN(**config["model"])
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Create loss function
    loss_fn = CombinedLoss(**config["loss"])
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["optimizer"]["lr"],
        weight_decay=config["optimizer"].get("weight_decay", 0.0),
    )
    
    # Create scheduler
    scheduler = None
    if config.get("scheduler"):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config["scheduler"]["factor"],
            patience=config["scheduler"]["patience"],
            verbose=True,
        )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=config["save_dir"],
        log_dir=config["log_dir"],
        max_epochs=config["max_epochs"],
        patience=config["patience"],
        grad_clip=config.get("grad_clip"),
    )
    
    return trainer
