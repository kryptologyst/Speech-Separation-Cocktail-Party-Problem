"""Data loading and preprocessing for speech separation."""

import os
import random
import numpy as np
import torch
import torchaudio
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset, DataLoader
import librosa
import soundfile as sf


class SpeechSeparationDataset(Dataset):
    """Dataset for speech separation tasks."""
    
    def __init__(
        self,
        data_dir: str,
        meta_file: str,
        sample_rate: int = 16000,
        segment_length: Optional[int] = None,
        n_sources: int = 2,
        augment: bool = True,
        split: str = "train",
    ):
        """Initialize dataset.
        
        Args:
            data_dir: Directory containing audio files
            meta_file: Path to metadata CSV file
            sample_rate: Target sample rate
            segment_length: Length of audio segments (None for full length)
            n_sources: Number of sources to separate
            augment: Whether to apply data augmentation
            split: Dataset split (train/val/test)
        """
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.n_sources = n_sources
        self.augment = augment and split == "train"
        self.split = split
        
        # Load metadata
        self.meta_df = pd.read_csv(meta_file)
        self.meta_df = self.meta_df[self.meta_df["split"] == split].reset_index(drop=True)
        
        # Filter for multi-source files
        self.meta_df = self.meta_df[self.meta_df["n_sources"] >= n_sources]
        
        print(f"Loaded {len(self.meta_df)} samples for {split} split")
    
    def __len__(self) -> int:
        return len(self.meta_df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset."""
        row = self.meta_df.iloc[idx]
        
        # Load mixture
        mixture_path = os.path.join(self.data_dir, row["mixture_path"])
        mixture, sr = librosa.load(mixture_path, sr=self.sample_rate)
        
        # Load sources
        sources = []
        for i in range(self.n_sources):
            source_path = os.path.join(self.data_dir, row[f"source_{i}_path"])
            source, _ = librosa.load(source_path, sr=self.sample_rate)
            sources.append(source)
        
        # Ensure same length
        min_length = min(len(mixture), *[len(s) for s in sources])
        mixture = mixture[:min_length]
        sources = [s[:min_length] for s in sources]
        
        # Convert to tensors
        mixture = torch.from_numpy(mixture).float()
        sources = torch.stack([torch.from_numpy(s).float() for s in sources])
        
        # Segment if needed
        if self.segment_length is not None and len(mixture) > self.segment_length:
            start_idx = random.randint(0, len(mixture) - self.segment_length)
            mixture = mixture[start_idx:start_idx + self.segment_length]
            sources = sources[:, start_idx:start_idx + self.segment_length]
        
        # Data augmentation
        if self.augment:
            mixture, sources = self._augment(mixture, sources)
        
        return {
            "mixture": mixture,
            "sources": sources,
            "metadata": {
                "sample_id": row["sample_id"],
                "n_sources": row["n_sources"],
                "snr": row.get("snr", 0.0),
            }
        }
    
    def _augment(self, mixture: torch.Tensor, sources: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply data augmentation."""
        # Random gain
        if random.random() < 0.5:
            gain = random.uniform(0.8, 1.2)
            mixture = mixture * gain
            sources = sources * gain
        
        # Add noise
        if random.random() < 0.3:
            noise_level = random.uniform(0.01, 0.05)
            noise = torch.randn_like(mixture) * noise_level
            mixture = mixture + noise
        
        return mixture, sources


class SyntheticDataGenerator:
    """Generate synthetic speech separation data."""
    
    def __init__(
        self,
        output_dir: str,
        sample_rate: int = 16000,
        duration_range: Tuple[float, float] = (2.0, 8.0),
        snr_range: Tuple[float, float] = (0.0, 20.0),
        n_sources: int = 2,
    ):
        """Initialize synthetic data generator.
        
        Args:
            output_dir: Output directory for generated data
            sample_rate: Sample rate for generated audio
            duration_range: Range of audio durations in seconds
            snr_range: Range of SNR values in dB
            n_sources: Number of sources to mix
        """
        self.output_dir = output_dir
        self.sample_rate = sample_rate
        self.duration_range = duration_range
        self.snr_range = snr_range
        self.n_sources = n_sources
        
        # Create output directories
        os.makedirs(os.path.join(output_dir, "wav"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "meta"), exist_ok=True)
    
    def generate_sine_waves(self, n_samples: int = 100) -> None:
        """Generate synthetic data using sine waves."""
        print(f"Generating {n_samples} synthetic samples with sine waves...")
        
        metadata = []
        
        for i in range(n_samples):
            # Generate random parameters
            duration = random.uniform(*self.duration_range)
            snr = random.uniform(*self.snr_range)
            
            # Generate sources
            sources = []
            for j in range(self.n_sources):
                # Random frequency and phase
                freq = random.uniform(200, 2000)
                phase = random.uniform(0, 2 * np.pi)
                
                # Generate sine wave
                t = np.linspace(0, duration, int(duration * self.sample_rate))
                source = np.sin(2 * np.pi * freq * t + phase)
                
                # Add some harmonics
                if random.random() < 0.5:
                    harmonic_freq = freq * random.uniform(2, 4)
                    harmonic_amp = random.uniform(0.1, 0.3)
                    source += harmonic_amp * np.sin(2 * np.pi * harmonic_freq * t + phase)
                
                sources.append(source)
            
            # Mix sources
            mixture = self._mix_sources(sources, snr)
            
            # Save files
            sample_id = f"synthetic_{i:04d}"
            
            # Save mixture
            mixture_path = f"wav/{sample_id}_mixture.wav"
            sf.write(
                os.path.join(self.output_dir, mixture_path),
                mixture, self.sample_rate
            )
            
            # Save sources
            source_paths = []
            for j, source in enumerate(sources):
                source_path = f"wav/{sample_id}_source_{j}.wav"
                sf.write(
                    os.path.join(self.output_dir, source_path),
                    source, self.sample_rate
                )
                source_paths.append(source_path)
            
            # Add to metadata
            metadata.append({
                "sample_id": sample_id,
                "mixture_path": mixture_path,
                "source_0_path": source_paths[0],
                "source_1_path": source_paths[1],
                "n_sources": self.n_sources,
                "snr": snr,
                "duration": duration,
                "split": "train" if i < n_samples * 0.7 else "val" if i < n_samples * 0.85 else "test",
            })
        
        # Save metadata
        meta_df = pd.DataFrame(metadata)
        meta_df.to_csv(
            os.path.join(self.output_dir, "meta", "synthetic_metadata.csv"),
            index=False
        )
        
        print(f"Generated {n_samples} samples. Metadata saved to synthetic_metadata.csv")
    
    def generate_tts_samples(self, n_samples: int = 50) -> None:
        """Generate synthetic data using TTS (if available)."""
        try:
            import pyttsx3
            
            print(f"Generating {n_samples} synthetic samples with TTS...")
            
            # Initialize TTS engine
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)  # Speed of speech
            
            # Sample texts
            texts = [
                "Hello, how are you today?",
                "The weather is nice today.",
                "I love machine learning.",
                "Speech separation is fascinating.",
                "Deep learning models are powerful.",
                "Audio processing requires careful attention.",
                "Neural networks can learn complex patterns.",
                "Signal processing is fundamental to audio.",
            ]
            
            metadata = []
            
            for i in range(n_samples):
                # Generate random parameters
                duration = random.uniform(*self.duration_range)
                snr = random.uniform(*self.snr_range)
                
                # Generate sources using TTS
                sources = []
                for j in range(self.n_sources):
                    text = random.choice(texts)
                    
                    # Generate TTS audio
                    engine.setProperty('voice', random.choice(engine.getProperty('voices')).id)
                    
                    # Save to temporary file
                    temp_path = f"/tmp/temp_source_{i}_{j}.wav"
                    engine.save_to_file(text, temp_path)
                    engine.runAndWait()
                    
                    # Load and resample
                    source, sr = librosa.load(temp_path, sr=self.sample_rate)
                    
                    # Trim or pad to desired duration
                    target_length = int(duration * self.sample_rate)
                    if len(source) > target_length:
                        source = source[:target_length]
                    else:
                        source = np.pad(source, (0, target_length - len(source)))
                    
                    sources.append(source)
                    
                    # Clean up temp file
                    os.remove(temp_path)
                
                # Mix sources
                mixture = self._mix_sources(sources, snr)
                
                # Save files
                sample_id = f"tts_{i:04d}"
                
                # Save mixture
                mixture_path = f"wav/{sample_id}_mixture.wav"
                sf.write(
                    os.path.join(self.output_dir, mixture_path),
                    mixture, self.sample_rate
                )
                
                # Save sources
                source_paths = []
                for j, source in enumerate(sources):
                    source_path = f"wav/{sample_id}_source_{j}.wav"
                    sf.write(
                        os.path.join(self.output_dir, source_path),
                        source, self.sample_rate
                    )
                    source_paths.append(source_path)
                
                # Add to metadata
                metadata.append({
                    "sample_id": sample_id,
                    "mixture_path": mixture_path,
                    "source_0_path": source_paths[0],
                    "source_1_path": source_paths[1],
                    "n_sources": self.n_sources,
                    "snr": snr,
                    "duration": duration,
                    "split": "train" if i < n_samples * 0.7 else "val" if i < n_samples * 0.85 else "test",
                })
            
            # Save metadata
            meta_df = pd.DataFrame(metadata)
            meta_df.to_csv(
                os.path.join(self.output_dir, "meta", "tts_metadata.csv"),
                index=False
            )
            
            print(f"Generated {n_samples} TTS samples. Metadata saved to tts_metadata.csv")
            
        except ImportError:
            print("pyttsx3 not available. Skipping TTS generation.")
    
    def _mix_sources(self, sources: List[np.ndarray], snr: float) -> np.ndarray:
        """Mix sources with specified SNR."""
        if len(sources) < 2:
            return sources[0]
        
        # Use first source as target, others as interference
        target = sources[0]
        interference = np.sum(sources[1:], axis=0)
        
        # Calculate power
        target_power = np.sum(target ** 2)
        interference_power = np.sum(interference ** 2)
        
        if interference_power > 0:
            # Calculate scaling factor for SNR
            snr_linear = 10 ** (snr / 10)
            scale_factor = np.sqrt(target_power / (snr_linear * interference_power))
            interference = interference * scale_factor
        
        return target + interference


def create_data_loaders(
    data_dir: str,
    meta_file: str,
    batch_size: int = 16,
    num_workers: int = 4,
    sample_rate: int = 16000,
    segment_length: Optional[int] = None,
    n_sources: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for train, validation, and test sets.
    
    Args:
        data_dir: Directory containing audio files
        meta_file: Path to metadata CSV file
        batch_size: Batch size
        num_workers: Number of worker processes
        sample_rate: Target sample rate
        segment_length: Length of audio segments
        n_sources: Number of sources to separate
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = SpeechSeparationDataset(
        data_dir=data_dir,
        meta_file=meta_file,
        sample_rate=sample_rate,
        segment_length=segment_length,
        n_sources=n_sources,
        augment=True,
        split="train",
    )
    
    val_dataset = SpeechSeparationDataset(
        data_dir=data_dir,
        meta_file=meta_file,
        sample_rate=sample_rate,
        segment_length=segment_length,
        n_sources=n_sources,
        augment=False,
        split="val",
    )
    
    test_dataset = SpeechSeparationDataset(
        data_dir=data_dir,
        meta_file=meta_file,
        sample_rate=sample_rate,
        segment_length=segment_length,
        n_sources=n_sources,
        augment=False,
        split="test",
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    return train_loader, val_loader, test_loader
