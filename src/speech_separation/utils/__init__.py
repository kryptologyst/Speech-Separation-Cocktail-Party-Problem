"""Utility functions for speech separation."""

import random
import numpy as np
import torch
from typing import Optional, Tuple, Union


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA -> MPS -> CPU).
    
    Returns:
        torch.device: Available device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def normalize_audio(audio: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize audio to [-1, 1] range.
    
    Args:
        audio: Input audio tensor
        eps: Small value to avoid division by zero
        
    Returns:
        Normalized audio tensor
    """
    max_val = torch.max(torch.abs(audio))
    if max_val > eps:
        return audio / max_val
    return audio


def pad_audio(audio: torch.Tensor, target_length: int) -> torch.Tensor:
    """Pad audio to target length.
    
    Args:
        audio: Input audio tensor
        target_length: Target length in samples
        
    Returns:
        Padded audio tensor
    """
    if audio.shape[-1] >= target_length:
        return audio[..., :target_length]
    
    pad_length = target_length - audio.shape[-1]
    padding = (0, pad_length)
    return torch.nn.functional.pad(audio, padding)


def mix_signals(
    signal1: torch.Tensor, 
    signal2: torch.Tensor, 
    snr_db: float = 10.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mix two signals at specified SNR.
    
    Args:
        signal1: First signal
        signal2: Second signal  
        snr_db: Signal-to-noise ratio in dB
        
    Returns:
        Tuple of (mixed_signal, signal1_scaled, signal2_scaled)
    """
    # Calculate power
    power1 = torch.sum(signal1 ** 2)
    power2 = torch.sum(signal2 ** 2)
    
    # Calculate scaling factor based on SNR
    snr_linear = 10 ** (snr_db / 10)
    scale_factor = torch.sqrt((power1 + power2) / (snr_linear * power2))
    
    # Scale signals
    signal1_scaled = signal1
    signal2_scaled = scale_factor * signal2
    
    # Mix signals
    mixed_signal = signal1_scaled + signal2_scaled
    
    return mixed_signal, signal1_scaled, signal2_scaled


def compute_si_sdr(
    estimate: torch.Tensor, 
    target: torch.Tensor, 
    eps: float = 1e-8
) -> torch.Tensor:
    """Compute Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).
    
    Args:
        estimate: Estimated signal
        target: Target signal
        eps: Small value to avoid division by zero
        
    Returns:
        SI-SDR in dB
    """
    # Remove mean
    estimate = estimate - torch.mean(estimate, dim=-1, keepdim=True)
    target = target - torch.mean(target, dim=-1, keepdim=True)
    
    # Compute optimal scaling factor
    alpha = torch.sum(estimate * target, dim=-1, keepdim=True) / (
        torch.sum(target ** 2, dim=-1, keepdim=True) + eps
    )
    
    # Scale target
    target_scaled = alpha * target
    
    # Compute SI-SDR
    si_sdr = 10 * torch.log10(
        torch.sum(target_scaled ** 2, dim=-1) / 
        (torch.sum((estimate - target_scaled) ** 2, dim=-1) + eps)
    )
    
    return si_sdr


def load_audio_file(file_path: str, sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """Load audio file using librosa.
    
    Args:
        file_path: Path to audio file
        sr: Target sample rate (None to keep original)
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    import librosa
    audio, sr = librosa.load(file_path, sr=sr)
    return audio, sr


def save_audio_file(
    audio: Union[np.ndarray, torch.Tensor], 
    file_path: str, 
    sr: int = 16000
) -> None:
    """Save audio to file.
    
    Args:
        audio: Audio data
        file_path: Output file path
        sr: Sample rate
    """
    import soundfile as sf
    
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    
    sf.write(file_path, audio, sr)
