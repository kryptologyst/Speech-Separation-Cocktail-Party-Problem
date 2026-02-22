"""Evaluation module for speech separation models."""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from ..models import ConvTasNet, DPRNN
from ..metrics import MetricCalculator, Leaderboard
from ..utils import get_device, save_audio_file


class Evaluator:
    """Evaluator for speech separation models."""
    
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: Optional[torch.device] = None,
        output_dir: str = "evaluation_results",
    ):
        """Initialize evaluator.
        
        Args:
            model: Model to evaluate
            test_loader: Test data loader
            device: Device to use
            output_dir: Directory to save results
        """
        self.model = model
        self.test_loader = test_loader
        self.device = device or get_device()
        self.output_dir = output_dir
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize metrics
        self.metric_calculator = MetricCalculator()
        
        # Initialize leaderboard
        self.leaderboard = Leaderboard()
        
        print(f"Evaluator initialized on device: {self.device}")
    
    def evaluate(self, save_audio: bool = True, save_plots: bool = True) -> Dict[str, float]:
        """Evaluate the model on test set.
        
        Args:
            save_audio: Whether to save separated audio samples
            save_plots: Whether to save evaluation plots
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("Starting evaluation...")
        
        all_metrics = {}
        all_predictions = []
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Evaluating")
            
            for batch_idx, batch in enumerate(pbar):
                # Move data to device
                mixture = batch["mixture"].to(self.device)
                sources = batch["sources"].to(self.device)
                
                # Forward pass
                estimates = self.model(mixture)
                
                # Compute metrics
                metrics = self.metric_calculator(estimates, sources)
                
                # Accumulate metrics
                for key, values in metrics.items():
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].extend(values.flatten().cpu().numpy())
                
                # Store predictions for analysis
                batch_predictions = {
                    "mixture": mixture.cpu(),
                    "sources": sources.cpu(),
                    "estimates": estimates.cpu(),
                    "metrics": {k: v.cpu() for k, v in metrics.items()},
                }
                all_predictions.append(batch_predictions)
                
                # Save audio samples (first few batches)
                if save_audio and batch_idx < 5:
                    self._save_audio_samples(batch_predictions, batch_idx)
                
                # Update progress bar
                avg_si_sdr = np.mean(all_metrics.get("si_sdr", [0]))
                pbar.set_postfix({"avg_si_sdr": f"{avg_si_sdr:.2f}"})
        
        # Compute final metrics
        final_metrics = {}
        for key, values in all_metrics.items():
            final_metrics[f"{key}_mean"] = np.mean(values)
            final_metrics[f"{key}_std"] = np.std(values)
            final_metrics[f"{key}_min"] = np.min(values)
            final_metrics[f"{key}_max"] = np.max(values)
        
        # Save results
        self._save_results(final_metrics, all_predictions)
        
        # Create plots
        if save_plots:
            self._create_plots(all_predictions, final_metrics)
        
        print("\nEvaluation completed!")
        print("="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        for key, value in final_metrics.items():
            if key.endswith("_mean"):
                print(f"{key.replace('_mean', '').upper()}: {value:.3f} ± {final_metrics[key.replace('_mean', '_std')]:.3f}")
        
        return final_metrics
    
    def _save_audio_samples(self, predictions: Dict, batch_idx: int) -> None:
        """Save audio samples for qualitative evaluation."""
        audio_dir = os.path.join(self.output_dir, "audio_samples")
        os.makedirs(audio_dir, exist_ok=True)
        
        batch_size = predictions["mixture"].shape[0]
        
        for i in range(min(batch_size, 3)):  # Save first 3 samples per batch
            sample_id = f"batch_{batch_idx}_sample_{i}"
            
            # Save mixture
            mixture_path = os.path.join(audio_dir, f"{sample_id}_mixture.wav")
            save_audio_file(
                predictions["mixture"][i].numpy(),
                mixture_path,
                sr=16000
            )
            
            # Save sources and estimates
            n_sources = predictions["sources"].shape[1]
            for j in range(n_sources):
                # Original source
                source_path = os.path.join(audio_dir, f"{sample_id}_source_{j}.wav")
                save_audio_file(
                    predictions["sources"][i, j].numpy(),
                    source_path,
                    sr=16000
                )
                
                # Estimated source
                estimate_path = os.path.join(audio_dir, f"{sample_id}_estimate_{j}.wav")
                save_audio_file(
                    predictions["estimates"][i, j].numpy(),
                    estimate_path,
                    sr=16000
                )
    
    def _save_results(self, metrics: Dict[str, float], predictions: List[Dict]) -> None:
        """Save evaluation results."""
        # Save metrics summary
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(
            os.path.join(self.output_dir, "evaluation_metrics.csv"),
            index=False
        )
        
        # Save detailed results
        detailed_results = []
        for batch_idx, batch_preds in enumerate(predictions):
            batch_size = batch_preds["mixture"].shape[0]
            
            for i in range(batch_size):
                result = {
                    "batch_idx": batch_idx,
                    "sample_idx": i,
                }
                
                # Add metrics for each source
                n_sources = batch_preds["sources"].shape[1]
                for j in range(n_sources):
                    for metric_name, values in batch_preds["metrics"].items():
                        result[f"{metric_name}_source_{j}"] = values[i, j].item()
                
                detailed_results.append(result)
        
        detailed_df = pd.DataFrame(detailed_results)
        detailed_df.to_csv(
            os.path.join(self.output_dir, "detailed_results.csv"),
            index=False
        )
        
        print(f"Results saved to {self.output_dir}")
    
    def _create_plots(self, predictions: List[Dict], metrics: Dict[str, float]) -> None:
        """Create evaluation plots."""
        plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Collect all metrics for plotting
        all_si_sdr = []
        all_pesq = []
        all_stoi = []
        all_sdr = []
        
        for batch_preds in predictions:
            for metric_name, values in batch_preds["metrics"].items():
                if metric_name == "si_sdr":
                    all_si_sdr.extend(values.flatten().cpu().numpy())
                elif metric_name == "pesq":
                    all_pesq.extend(values.flatten().cpu().numpy())
                elif metric_name == "stoi":
                    all_stoi.extend(values.flatten().cpu().numpy())
                elif metric_name == "sdr":
                    all_sdr.extend(values.flatten().cpu().numpy())
        
        # Create metric distribution plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Evaluation Metrics Distribution", fontsize=16)
        
        # SI-SDR distribution
        axes[0, 0].hist(all_si_sdr, bins=30, alpha=0.7, color='blue')
        axes[0, 0].set_title(f"SI-SDR Distribution (Mean: {metrics['si_sdr_mean']:.2f})")
        axes[0, 0].set_xlabel("SI-SDR (dB)")
        axes[0, 0].set_ylabel("Frequency")
        
        # PESQ distribution
        axes[0, 1].hist(all_pesq, bins=30, alpha=0.7, color='green')
        axes[0, 1].set_title(f"PESQ Distribution (Mean: {metrics['pesq_mean']:.2f})")
        axes[0, 1].set_xlabel("PESQ")
        axes[0, 1].set_ylabel("Frequency")
        
        # STOI distribution
        axes[1, 0].hist(all_stoi, bins=30, alpha=0.7, color='red')
        axes[1, 0].set_title(f"STOI Distribution (Mean: {metrics['stoi_mean']:.3f})")
        axes[1, 0].set_xlabel("STOI")
        axes[1, 0].set_ylabel("Frequency")
        
        # SDR distribution
        axes[1, 1].hist(all_sdr, bins=30, alpha=0.7, color='orange')
        axes[1, 1].set_title(f"SDR Distribution (Mean: {metrics['sdr_mean']:.2f})")
        axes[1, 1].set_xlabel("SDR (dB)")
        axes[1, 1].set_ylabel("Frequency")
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "metrics_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create correlation plot
        if len(all_si_sdr) > 0 and len(all_pesq) > 0:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            
            # Scatter plot of SI-SDR vs PESQ
            ax.scatter(all_si_sdr, all_pesq, alpha=0.6, s=20)
            ax.set_xlabel("SI-SDR (dB)")
            ax.set_ylabel("PESQ")
            ax.set_title("SI-SDR vs PESQ Correlation")
            
            # Add correlation coefficient
            correlation = np.corrcoef(all_si_sdr, all_pesq)[0, 1]
            ax.text(0.05, 0.95, f"Correlation: {correlation:.3f}", 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "si_sdr_vs_pesq.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Plots saved to {plots_dir}")
    
    def compare_models(self, model_results: Dict[str, Dict[str, float]]) -> None:
        """Compare multiple models.
        
        Args:
            model_results: Dictionary mapping model names to their metrics
        """
        # Create comparison plot
        metrics_to_plot = ["si_sdr_mean", "pesq_mean", "stoi_mean", "sdr_mean"]
        metric_labels = ["SI-SDR (dB)", "PESQ", "STOI", "SDR (dB)"]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Model Comparison", fontsize=16)
        
        for i, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
            ax = axes[i // 2, i % 2]
            
            model_names = list(model_results.keys())
            values = [model_results[name].get(metric, 0) for name in model_names]
            
            bars = ax.bar(model_names, values, alpha=0.7)
            ax.set_title(f"{label} Comparison")
            ax.set_ylabel(label)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f"{value:.3f}", ha='center', va='bottom')
            
            # Rotate x-axis labels if needed
            if len(max(model_names, key=len)) > 8:
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "model_comparison.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Add to leaderboard
        for model_name, metrics in model_results.items():
            self.leaderboard.add_result(model_name, metrics)
        
        # Print leaderboard
        self.leaderboard.print_leaderboard()
        
        # Save leaderboard
        self.leaderboard.save_to_csv(
            os.path.join(self.output_dir, "leaderboard.csv")
        )


def load_model_for_evaluation(
    model_name: str,
    checkpoint_path: str,
    config: Dict,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Load a trained model for evaluation.
    
    Args:
        model_name: Name of the model ("conv_tasnet" or "dprnn")
        checkpoint_path: Path to model checkpoint
        config: Model configuration
        device: Device to use
        
    Returns:
        Loaded model
    """
    device = device or get_device()
    
    # Create model
    if model_name.lower() == "conv_tasnet":
        model = ConvTasNet(**config["model"])
    elif model_name.lower() == "dprnn":
        model = DPRNN(**config["model"])
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Trained for {checkpoint['epoch']} epochs")
    print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
    
    return model


def evaluate_model(
    model_name: str,
    checkpoint_path: str,
    test_loader: DataLoader,
    config: Dict,
    device: Optional[torch.device] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, float]:
    """Evaluate a trained model.
    
    Args:
        model_name: Name of the model
        checkpoint_path: Path to model checkpoint
        test_loader: Test data loader
        config: Model configuration
        device: Device to use
        output_dir: Output directory for results
        
    Returns:
        Evaluation metrics
    """
    device = device or get_device()
    
    # Load model
    model = load_model_for_evaluation(model_name, checkpoint_path, config, device)
    
    # Create evaluator
    if output_dir is None:
        output_dir = f"evaluation_results_{model_name}"
    
    evaluator = Evaluator(model, test_loader, device, output_dir)
    
    # Evaluate
    metrics = evaluator.evaluate()
    
    return metrics
