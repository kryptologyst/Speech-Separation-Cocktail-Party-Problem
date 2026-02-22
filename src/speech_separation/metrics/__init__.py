"""Evaluation metrics for speech separation."""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings

# Suppress pesq warnings
warnings.filterwarnings("ignore", category=UserWarning)


class SISDRMetric:
    """Scale-Invariant Signal-to-Distortion Ratio metric."""
    
    def __init__(self, eps: float = 1e-8):
        """Initialize SI-SDR metric.
        
        Args:
            eps: Small value to avoid division by zero
        """
        self.eps = eps
    
    def __call__(
        self, 
        estimate: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute SI-SDR metric.
        
        Args:
            estimate: Estimated signal [batch, n_src, samples]
            target: Target signal [batch, n_src, samples]
            
        Returns:
            SI-SDR values [batch, n_src]
        """
        # Remove mean
        estimate = estimate - torch.mean(estimate, dim=-1, keepdim=True)
        target = target - torch.mean(target, dim=-1, keepdim=True)
        
        # Compute optimal scaling factor
        alpha = torch.sum(estimate * target, dim=-1, keepdim=True) / (
            torch.sum(target ** 2, dim=-1, keepdim=True) + self.eps
        )
        
        # Scale target
        target_scaled = alpha * target
        
        # Compute SI-SDR
        si_sdr = 10 * torch.log10(
            torch.sum(target_scaled ** 2, dim=-1) / 
            (torch.sum((estimate - target_scaled) ** 2, dim=-1) + self.eps)
        )
        
        return si_sdr


class PESQMetric:
    """PESQ (Perceptual Evaluation of Speech Quality) metric."""
    
    def __init__(self, sample_rate: int = 16000):
        """Initialize PESQ metric.
        
        Args:
            sample_rate: Sample rate of audio
        """
        self.sample_rate = sample_rate
        
        try:
            import pesq
            self.pesq = pesq
        except ImportError:
            print("Warning: pesq not available. PESQ metric will be disabled.")
            self.pesq = None
    
    def __call__(
        self, 
        estimate: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute PESQ metric.
        
        Args:
            estimate: Estimated signal [batch, n_src, samples]
            target: Target signal [batch, n_src, samples]
            
        Returns:
            PESQ values [batch, n_src]
        """
        if self.pesq is None:
            return torch.zeros(estimate.shape[:2])
        
        batch_size, n_src = estimate.shape[:2]
        pesq_values = torch.zeros(batch_size, n_src)
        
        for b in range(batch_size):
            for s in range(n_src):
                try:
                    # Convert to numpy
                    est_np = estimate[b, s].detach().cpu().numpy()
                    tgt_np = target[b, s].detach().cpu().numpy()
                    
                    # Compute PESQ
                    pesq_val = self.pesq.pesq(
                        self.sample_rate, tgt_np, est_np, 'wb'
                    )
                    pesq_values[b, s] = pesq_val
                except:
                    pesq_values[b, s] = 0.0
        
        return pesq_values


class STOIMetric:
    """STOI (Short-Time Objective Intelligibility) metric."""
    
    def __init__(self, sample_rate: int = 16000):
        """Initialize STOI metric.
        
        Args:
            sample_rate: Sample rate of audio
        """
        self.sample_rate = sample_rate
        
        try:
            import pystoi
            self.stoi = pystoi
        except ImportError:
            print("Warning: pystoi not available. STOI metric will be disabled.")
            self.stoi = None
    
    def __call__(
        self, 
        estimate: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute STOI metric.
        
        Args:
            estimate: Estimated signal [batch, n_src, samples]
            target: Target signal [batch, n_src, samples]
            
        Returns:
            STOI values [batch, n_src]
        """
        if self.stoi is None:
            return torch.zeros(estimate.shape[:2])
        
        batch_size, n_src = estimate.shape[:2]
        stoi_values = torch.zeros(batch_size, n_src)
        
        for b in range(batch_size):
            for s in range(n_src):
                try:
                    # Convert to numpy
                    est_np = estimate[b, s].detach().cpu().numpy()
                    tgt_np = target[b, s].detach().cpu().numpy()
                    
                    # Compute STOI
                    stoi_val = self.stoi.stoi(tgt_np, est_np, self.sample_rate)
                    stoi_values[b, s] = stoi_val
                except:
                    stoi_values[b, s] = 0.0
        
        return stoi_values


class SDRMetric:
    """Signal-to-Distortion Ratio metric."""
    
    def __init__(self, eps: float = 1e-8):
        """Initialize SDR metric.
        
        Args:
            eps: Small value to avoid division by zero
        """
        self.eps = eps
    
    def __call__(
        self, 
        estimate: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute SDR metric.
        
        Args:
            estimate: Estimated signal [batch, n_src, samples]
            target: Target signal [batch, n_src, samples]
            
        Returns:
            SDR values [batch, n_src]
        """
        # Compute SDR
        sdr = 10 * torch.log10(
            torch.sum(target ** 2, dim=-1) / 
            (torch.sum((estimate - target) ** 2, dim=-1) + self.eps)
        )
        
        return sdr


class MetricCalculator:
    """Calculator for multiple speech separation metrics."""
    
    def __init__(self, sample_rate: int = 16000):
        """Initialize metric calculator.
        
        Args:
            sample_rate: Sample rate of audio
        """
        self.sample_rate = sample_rate
        
        # Initialize metrics
        self.si_sdr = SISDRMetric()
        self.pesq = PESQMetric(sample_rate)
        self.stoi = STOIMetric(sample_rate)
        self.sdr = SDRMetric()
    
    def __call__(
        self, 
        estimate: torch.Tensor, 
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute all metrics.
        
        Args:
            estimate: Estimated signal [batch, n_src, samples]
            target: Target signal [batch, n_src, samples]
            
        Returns:
            Dictionary of metric values
        """
        metrics = {}
        
        # SI-SDR
        metrics["si_sdr"] = self.si_sdr(estimate, target)
        
        # PESQ
        metrics["pesq"] = self.pesq(estimate, target)
        
        # STOI
        metrics["stoi"] = self.stoi(estimate, target)
        
        # SDR
        metrics["sdr"] = self.sdr(estimate, target)
        
        return metrics
    
    def compute_averages(self, metrics: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute average metrics across batch and sources.
        
        Args:
            metrics: Dictionary of metric values
            
        Returns:
            Dictionary of average metric values
        """
        averages = {}
        
        for name, values in metrics.items():
            averages[f"{name}_mean"] = torch.mean(values).item()
            averages[f"{name}_std"] = torch.std(values).item()
        
        return averages


class Leaderboard:
    """Leaderboard for tracking model performance."""
    
    def __init__(self):
        """Initialize leaderboard."""
        self.results = []
    
    def add_result(
        self,
        model_name: str,
        metrics: Dict[str, float],
        config: Optional[Dict] = None,
    ) -> None:
        """Add a result to the leaderboard.
        
        Args:
            model_name: Name of the model
            metrics: Dictionary of metric values
            config: Model configuration (optional)
        """
        result = {
            "model_name": model_name,
            "metrics": metrics,
            "config": config or {},
        }
        self.results.append(result)
    
    def get_best_model(self, metric: str = "si_sdr_mean") -> Dict:
        """Get the best model for a specific metric.
        
        Args:
            metric: Metric name to optimize
            
        Returns:
            Best model result
        """
        if not self.results:
            return None
        
        best_result = max(self.results, key=lambda x: x["metrics"].get(metric, -float('inf')))
        return best_result
    
    def print_leaderboard(self, top_k: int = 10) -> None:
        """Print the leaderboard.
        
        Args:
            top_k: Number of top results to show
        """
        if not self.results:
            print("No results in leaderboard.")
            return
        
        # Sort by SI-SDR
        sorted_results = sorted(
            self.results, 
            key=lambda x: x["metrics"].get("si_sdr_mean", -float('inf')),
            reverse=True
        )
        
        print("\n" + "="*80)
        print("SPEECH SEPARATION LEADERBOARD")
        print("="*80)
        print(f"{'Rank':<4} {'Model':<20} {'SI-SDR':<8} {'PESQ':<8} {'STOI':<8} {'SDR':<8}")
        print("-"*80)
        
        for i, result in enumerate(sorted_results[:top_k]):
            metrics = result["metrics"]
            print(
                f"{i+1:<4} {result['model_name']:<20} "
                f"{metrics.get('si_sdr_mean', 0):<8.2f} "
                f"{metrics.get('pesq_mean', 0):<8.2f} "
                f"{metrics.get('stoi_mean', 0):<8.2f} "
                f"{metrics.get('sdr_mean', 0):<8.2f}"
            )
        
        print("="*80)
    
    def save_to_csv(self, filepath: str) -> None:
        """Save leaderboard to CSV file.
        
        Args:
            filepath: Path to save CSV file
        """
        import pandas as pd
        
        if not self.results:
            print("No results to save.")
            return
        
        # Flatten results for CSV
        flattened_results = []
        for result in self.results:
            row = {"model_name": result["model_name"]}
            row.update(result["metrics"])
            flattened_results.append(row)
        
        df = pd.DataFrame(flattened_results)
        df.to_csv(filepath, index=False)
        print(f"Leaderboard saved to {filepath}")
