"""Modernized version of the original speech separation code.

This script demonstrates the evolution from the basic spectral subtraction approach
to modern deep learning-based speech separation.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import stft, istft
import librosa
import soundfile as sf

from speech_separation.models import ConvTasNet, DPRNN
from speech_separation.utils import get_device, mix_signals, normalize_audio
from speech_separation.metrics import MetricCalculator


def load_audio(file_path: str, sr: int = 16000) -> tuple[np.ndarray, int]:
    """Load audio file using librosa.
    
    Args:
        file_path: Path to audio file
        sr: Target sample rate
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    audio, sr = librosa.load(file_path, sr=sr)
    return audio, sr


def mix_speech_signals(signal1: np.ndarray, signal2: np.ndarray, snr_db: float = 10.0) -> np.ndarray:
    """Mix two speech signals at a specified signal-to-noise ratio (SNR).
    
    Args:
        signal1: First speech signal
        signal2: Second speech signal
        snr_db: Signal-to-noise ratio in decibels
        
    Returns:
        Mixed signal
    """
    # Calculate the power of both signals
    power1 = np.sum(signal1**2)
    power2 = np.sum(signal2**2)
    
    # Calculate scaling factor based on desired SNR
    snr_linear = 10 ** (snr_db / 10)
    scale_factor = np.sqrt((power1 + power2) / (snr_linear * power2))
    
    # Mix signals
    mixed_signal = signal1 + scale_factor * signal2
    return mixed_signal


def spectral_subtraction_separation(mixed_signal: np.ndarray, original_signal1: np.ndarray, original_signal2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Perform a basic separation using spectral subtraction.
    
    Args:
        mixed_signal: The mixed signal
        original_signal1: Original first signal (used for estimating noise)
        original_signal2: Original second signal (used for estimating noise)
        
    Returns:
        Tuple of (separated_signal1, separated_signal2)
    """
    # Compute Short-Time Fourier Transform (STFT) of the mixed signal
    f, t, Zxx = stft(mixed_signal, nperseg=1024)
    
    # Estimate the magnitude and phase of the signals
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)
    
    # Estimate noise magnitude using original signals
    f, t, Zxx1 = stft(original_signal1, nperseg=1024)
    f, t, Zxx2 = stft(original_signal2, nperseg=1024)
    mag1 = np.abs(Zxx1)
    mag2 = np.abs(Zxx2)
    
    # Spectral subtraction (very simplified version)
    mag_separated1 = mag - mag1
    mag_separated2 = mag - mag2
    mag_separated1 = np.maximum(mag_separated1, 0)  # Avoid negative values
    mag_separated2 = np.maximum(mag_separated2, 0)  # Avoid negative values
    
    # Reconstruct signals
    _, separated_signal1 = istft(mag_separated1 * np.exp(1j * phase), nperseg=1024)
    _, separated_signal2 = istft(mag_separated2 * np.exp(1j * phase), nperseg=1024)
    
    return separated_signal1, separated_signal2


def modern_deep_learning_separation(mixed_signal: np.ndarray, model_name: str = "conv_tasnet") -> tuple[np.ndarray, np.ndarray]:
    """Perform separation using modern deep learning models.
    
    Args:
        mixed_signal: The mixed signal
        model_name: Name of the model to use ("conv_tasnet" or "dprnn")
        
    Returns:
        Tuple of (separated_signal1, separated_signal2)
    """
    device = get_device()
    
    # Create model
    if model_name.lower() == "conv_tasnet":
        model = ConvTasNet(n_src=2, n_filters=64, kernel_size=8, stride=4)
    elif model_name.lower() == "dprnn":
        model = DPRNN(n_src=2, n_filters=32, kernel_size=8, stride=4)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model.to(device)
    model.eval()
    
    # Convert to tensor
    mixed_tensor = torch.from_numpy(mixed_signal).float().unsqueeze(0).to(device)
    mixed_tensor = normalize_audio(mixed_tensor)
    
    # Forward pass
    with torch.no_grad():
        estimates = model(mixed_tensor)
    
    # Convert back to numpy
    estimates = estimates.squeeze(0).cpu().numpy()
    
    return estimates[0], estimates[1]


def compare_methods():
    """Compare traditional and modern speech separation methods."""
    print("Speech Separation Methods Comparison")
    print("=" * 50)
    
    # Generate synthetic speech signals
    sample_rate = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # Create two different frequency signals
    freq1 = 440  # A4 note
    freq2 = 523  # C5 note
    
    signal1 = np.sin(2 * np.pi * freq1 * t) * 0.5
    signal2 = np.sin(2 * np.pi * freq2 * t) * 0.5
    
    # Add harmonics
    signal1 += 0.2 * np.sin(2 * np.pi * freq1 * 2 * t)
    signal2 += 0.2 * np.sin(2 * np.pi * freq2 * 2 * t)
    
    # Mix signals
    snr_db = 10.0
    mixed_signal = mix_speech_signals(signal1, signal2, snr_db)
    
    print(f"Generated signals of length: {len(mixed_signal)} samples")
    print(f"SNR: {snr_db} dB")
    
    # Method 1: Traditional Spectral Subtraction
    print("\n1. Traditional Spectral Subtraction")
    separated_traditional_1, separated_traditional_2 = spectral_subtraction_separation(
        mixed_signal, signal1, signal2
    )
    
    # Method 2: Modern Deep Learning (Conv-TasNet)
    print("2. Modern Deep Learning (Conv-TasNet)")
    separated_modern_1, separated_modern_2 = modern_deep_learning_separation(
        mixed_signal, "conv_tasnet"
    )
    
    # Method 3: Modern Deep Learning (DPRNN)
    print("3. Modern Deep Learning (DPRNN)")
    separated_dprnn_1, separated_dprnn_2 = modern_deep_learning_separation(
        mixed_signal, "dprnn"
    )
    
    # Evaluate methods
    print("\nEvaluation Results:")
    print("-" * 30)
    
    # Convert to tensors for metric computation
    device = get_device()
    metric_calculator = MetricCalculator()
    
    # Traditional method metrics
    targets = torch.stack([
        torch.from_numpy(signal1).float(),
        torch.from_numpy(signal2).float()
    ]).unsqueeze(0).to(device)
    
    estimates_traditional = torch.stack([
        torch.from_numpy(separated_traditional_1).float(),
        torch.from_numpy(separated_traditional_2).float()
    ]).unsqueeze(0).to(device)
    
    metrics_traditional = metric_calculator(estimates_traditional, targets)
    
    print("Traditional Spectral Subtraction:")
    for name, values in metrics_traditional.items():
        print(f"  {name}: {values.mean().item():.3f}")
    
    # Modern Conv-TasNet metrics
    estimates_modern = torch.stack([
        torch.from_numpy(separated_modern_1).float(),
        torch.from_numpy(separated_modern_2).float()
    ]).unsqueeze(0).to(device)
    
    metrics_modern = metric_calculator(estimates_modern, targets)
    
    print("\nModern Conv-TasNet:")
    for name, values in metrics_modern.items():
        print(f"  {name}: {values.mean().item():.3f}")
    
    # Modern DPRNN metrics
    estimates_dprnn = torch.stack([
        torch.from_numpy(separated_dprnn_1).float(),
        torch.from_numpy(separated_dprnn_2).float()
    ]).unsqueeze(0).to(device)
    
    metrics_dprnn = metric_calculator(estimates_dprnn, targets)
    
    print("\nModern DPRNN:")
    for name, values in metrics_dprnn.items():
        print(f"  {name}: {values.mean().item():.3f}")
    
    # Visualization
    print("\nGenerating visualization...")
    
    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    
    # Original signals
    axes[0, 0].plot(t, signal1, label="Original Speech 1")
    axes[0, 0].set_title("Original Speech 1")
    axes[0, 0].legend()
    
    axes[0, 1].plot(t, signal2, label="Original Speech 2")
    axes[0, 1].set_title("Original Speech 2")
    axes[0, 1].legend()
    
    # Mixed signal
    axes[1, 0].plot(t, mixed_signal, label="Mixed Signal")
    axes[1, 0].set_title("Mixed Signal")
    axes[1, 0].legend()
    
    axes[1, 1].axis('off')  # Empty subplot
    
    # Traditional separation
    axes[2, 0].plot(t, separated_traditional_1, label="Separated Speech 1 (Traditional)")
    axes[2, 0].set_title("Traditional Spectral Subtraction")
    axes[2, 0].legend()
    
    axes[2, 1].plot(t, separated_traditional_2, label="Separated Speech 2 (Traditional)")
    axes[2, 1].set_title("Traditional Spectral Subtraction")
    axes[2, 1].legend()
    
    # Modern separation
    axes[3, 0].plot(t, separated_modern_1, label="Separated Speech 1 (Conv-TasNet)")
    axes[3, 0].set_title("Modern Deep Learning (Conv-TasNet)")
    axes[3, 0].legend()
    
    axes[3, 1].plot(t, separated_modern_2, label="Separated Speech 2 (Conv-TasNet)")
    axes[3, 1].set_title("Modern Deep Learning (Conv-TasNet)")
    axes[3, 1].legend()
    
    plt.tight_layout()
    plt.savefig("speech_separation_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nComparison completed!")
    print("Visualization saved as 'speech_separation_comparison.png'")
    
    # Summary
    print("\nSummary:")
    print("=" * 50)
    print("This comparison demonstrates the evolution from traditional spectral")
    print("subtraction to modern deep learning approaches for speech separation.")
    print("\nKey improvements with deep learning:")
    print("- Better separation quality (higher SI-SDR, PESQ, STOI)")
    print("- More robust to different signal types")
    print("- End-to-end learning without manual feature engineering")
    print("- Scalable to complex real-world scenarios")


if __name__ == "__main__":
    compare_methods()
