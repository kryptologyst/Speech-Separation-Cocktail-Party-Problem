"""Loss functions for speech separation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SISDRLoss(nn.Module):
    """Scale-Invariant Signal-to-Distortion Ratio Loss.
    
    SI-SDR is a common metric for speech separation that measures the quality
    of separated sources compared to the target sources.
    """
    
    def __init__(self, eps: float = 1e-8):
        """Initialize SI-SDR loss.
        
        Args:
            eps: Small value to avoid division by zero
        """
        super().__init__()
        self.eps = eps
    
    def forward(
        self, 
        estimate: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute SI-SDR loss.
        
        Args:
            estimate: Estimated signal [batch, n_src, samples]
            target: Target signal [batch, n_src, samples]
            
        Returns:
            Negative SI-SDR (to minimize)
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
        
        # Return negative SI-SDR (to minimize)
        return -torch.mean(si_sdr)


class MultiScaleLoss(nn.Module):
    """Multi-scale loss combining SI-SDR and MSE losses."""
    
    def __init__(
        self,
        si_sdr_weight: float = 1.0,
        mse_weight: float = 0.1,
        eps: float = 1e-8,
    ):
        """Initialize multi-scale loss.
        
        Args:
            si_sdr_weight: Weight for SI-SDR loss
            mse_weight: Weight for MSE loss
            eps: Small value to avoid division by zero
        """
        super().__init__()
        self.si_sdr_weight = si_sdr_weight
        self.mse_weight = mse_weight
        self.si_sdr_loss = SISDRLoss(eps)
        self.mse_loss = nn.MSELoss()
    
    def forward(
        self, 
        estimate: torch.Tensor, 
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Compute multi-scale loss.
        
        Args:
            estimate: Estimated signal [batch, n_src, samples]
            target: Target signal [batch, n_src, samples]
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # SI-SDR loss
        si_sdr_loss = self.si_sdr_loss(estimate, target)
        
        # MSE loss
        mse_loss = self.mse_loss(estimate, target)
        
        # Combined loss
        total_loss = self.si_sdr_weight * si_sdr_loss + self.mse_weight * mse_loss
        
        loss_dict = {
            "total_loss": total_loss.item(),
            "si_sdr_loss": si_sdr_loss.item(),
            "mse_loss": mse_loss.item(),
        }
        
        return total_loss, loss_dict


class SpectralLoss(nn.Module):
    """Spectral loss in frequency domain."""
    
    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: Optional[int] = None,
        window: str = "hann",
    ):
        """Initialize spectral loss.
        
        Args:
            n_fft: FFT size
            hop_length: Hop length
            win_length: Window length
            window: Window type
        """
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        
        if window == "hann":
            self.window = torch.hann_window(self.win_length)
        elif window == "hamming":
            self.window = torch.hamming_window(self.win_length)
        else:
            self.window = torch.ones(self.win_length)
    
    def forward(
        self, 
        estimate: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute spectral loss.
        
        Args:
            estimate: Estimated signal [batch, n_src, samples]
            target: Target signal [batch, n_src, samples]
            
        Returns:
            Spectral loss
        """
        # Move window to device
        if estimate.is_cuda:
            self.window = self.window.cuda()
        
        # Compute STFT
        estimate_stft = torch.stft(
            estimate.view(-1, estimate.shape[-1]),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )
        
        target_stft = torch.stft(
            target.view(-1, target.shape[-1]),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )
        
        # Compute magnitude
        estimate_mag = torch.abs(estimate_stft)
        target_mag = torch.abs(target_stft)
        
        # Compute loss
        loss = F.mse_loss(estimate_mag, target_mag)
        
        return loss


class PerceptualLoss(nn.Module):
    """Perceptual loss using pre-trained features."""
    
    def __init__(self, feature_dim: int = 512):
        """Initialize perceptual loss.
        
        Args:
            feature_dim: Dimension of feature representation
        """
        super().__init__()
        self.feature_dim = feature_dim
        
        # Simple feature extractor (can be replaced with pre-trained model)
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 64, 15, stride=4),
            nn.ReLU(),
            nn.Conv1d(64, 128, 15, stride=4),
            nn.ReLU(),
            nn.Conv1d(128, 256, 15, stride=4),
            nn.ReLU(),
            nn.Conv1d(256, feature_dim, 15, stride=4),
            nn.ReLU(),
        )
    
    def forward(
        self, 
        estimate: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute perceptual loss.
        
        Args:
            estimate: Estimated signal [batch, n_src, samples]
            target: Target signal [batch, n_src, samples]
            
        Returns:
            Perceptual loss
        """
        # Extract features
        estimate_features = self.feature_extractor(estimate.view(-1, 1, estimate.shape[-1]))
        target_features = self.feature_extractor(target.view(-1, 1, target.shape[-1]))
        
        # Compute loss
        loss = F.mse_loss(estimate_features, target_features)
        
        return loss


class CombinedLoss(nn.Module):
    """Combined loss function with multiple components."""
    
    def __init__(
        self,
        si_sdr_weight: float = 1.0,
        mse_weight: float = 0.1,
        spectral_weight: float = 0.1,
        perceptual_weight: float = 0.05,
        eps: float = 1e-8,
    ):
        """Initialize combined loss.
        
        Args:
            si_sdr_weight: Weight for SI-SDR loss
            mse_weight: Weight for MSE loss
            spectral_weight: Weight for spectral loss
            perceptual_weight: Weight for perceptual loss
            eps: Small value to avoid division by zero
        """
        super().__init__()
        self.si_sdr_weight = si_sdr_weight
        self.mse_weight = mse_weight
        self.spectral_weight = spectral_weight
        self.perceptual_weight = perceptual_weight
        
        self.si_sdr_loss = SISDRLoss(eps)
        self.mse_loss = nn.MSELoss()
        self.spectral_loss = SpectralLoss()
        self.perceptual_loss = PerceptualLoss()
    
    def forward(
        self, 
        estimate: torch.Tensor, 
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Compute combined loss.
        
        Args:
            estimate: Estimated signal [batch, n_src, samples]
            target: Target signal [batch, n_src, samples]
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Individual losses
        si_sdr_loss = self.si_sdr_loss(estimate, target)
        mse_loss = self.mse_loss(estimate, target)
        spectral_loss = self.spectral_loss(estimate, target)
        perceptual_loss = self.perceptual_loss(estimate, target)
        
        # Combined loss
        total_loss = (
            self.si_sdr_weight * si_sdr_loss +
            self.mse_weight * mse_loss +
            self.spectral_weight * spectral_loss +
            self.perceptual_weight * perceptual_loss
        )
        
        loss_dict = {
            "total_loss": total_loss.item(),
            "si_sdr_loss": si_sdr_loss.item(),
            "mse_loss": mse_loss.item(),
            "spectral_loss": spectral_loss.item(),
            "perceptual_loss": perceptual_loss.item(),
        }
        
        return total_loss, loss_dict
