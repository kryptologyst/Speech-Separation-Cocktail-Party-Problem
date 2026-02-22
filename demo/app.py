"""Streamlit demo for speech separation."""

import streamlit as st
import torch
import numpy as np
import librosa
import soundfile as sf
import io
import os
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from speech_separation.models import ConvTasNet, DPRNN
from speech_separation.metrics import MetricCalculator
from speech_separation.utils import get_device, normalize_audio, mix_signals


# Page configuration
st.set_page_config(
    page_title="Speech Separation Demo",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Privacy disclaimer
PRIVACY_DISCLAIMER = """
**PRIVACY DISCLAIMER**

This is a research demonstration tool for educational purposes only. 

- **No biometric identification**: This tool is not designed for biometric identification or voice recognition
- **No data storage**: Audio files are processed locally and not stored or transmitted
- **Research use only**: This tool is intended for research and educational purposes
- **No production use**: Do not use this tool for production biometric applications
- **Voice cloning prohibition**: Misuse of this technology for voice cloning or impersonation is strictly prohibited

By using this tool, you acknowledge and agree to these terms.
"""


@st.cache_resource
def load_model(model_name: str, checkpoint_path: Optional[str] = None) -> torch.nn.Module:
    """Load a pre-trained model."""
    device = get_device()
    
    if model_name == "Conv-TasNet":
        model = ConvTasNet(n_src=2, n_filters=256, kernel_size=16, stride=8)
    elif model_name == "DPRNN":
        model = DPRNN(n_src=2, n_filters=64, kernel_size=16, stride=8)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model.to(device)
    model.eval()
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        st.success(f"Loaded checkpoint from {checkpoint_path}")
    else:
        st.warning("No checkpoint provided. Using randomly initialized model.")
    
    return model


def process_audio_file(audio_file, sample_rate: int = 16000) -> np.ndarray:
    """Process uploaded audio file."""
    # Read audio file
    audio_bytes = audio_file.read()
    
    # Load with librosa
    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=sample_rate)
    
    return audio


def create_synthetic_mixture(signal1: np.ndarray, signal2: np.ndarray, snr_db: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create synthetic mixture from two signals."""
    # Convert to tensors
    signal1_tensor = torch.from_numpy(signal1).float()
    signal2_tensor = torch.from_numpy(signal2).float()
    
    # Mix signals
    mixed, s1_scaled, s2_scaled = mix_signals(signal1_tensor, signal2_tensor, snr_db)
    
    return mixed.numpy(), s1_scaled.numpy(), s2_scaled.numpy()


def separate_speech(model: torch.nn.Module, mixture: np.ndarray, device: torch.device) -> np.ndarray:
    """Separate speech using the model."""
    # Convert to tensor
    mixture_tensor = torch.from_numpy(mixture).float().unsqueeze(0).to(device)
    
    # Normalize
    mixture_tensor = normalize_audio(mixture_tensor)
    
    # Forward pass
    with torch.no_grad():
        estimates = model(mixture_tensor)
    
    # Convert back to numpy
    estimates = estimates.squeeze(0).cpu().numpy()
    
    return estimates


def compute_metrics(estimates: np.ndarray, targets: np.ndarray) -> dict:
    """Compute separation metrics."""
    device = get_device()
    
    # Convert to tensors
    estimates_tensor = torch.from_numpy(estimates).float().unsqueeze(0).to(device)
    targets_tensor = torch.from_numpy(targets).float().unsqueeze(0).to(device)
    
    # Compute metrics
    metric_calculator = MetricCalculator()
    metrics = metric_calculator(estimates_tensor, targets_tensor)
    
    # Convert to numpy and average
    result = {}
    for key, values in metrics.items():
        result[key] = values.mean().item()
    
    return result


def plot_audio_signals(signals: dict, sample_rate: int = 16000):
    """Plot audio signals."""
    fig = make_subplots(
        rows=len(signals), cols=1,
        subplot_titles=list(signals.keys()),
        vertical_spacing=0.1
    )
    
    time_axis = np.linspace(0, len(list(signals.values())[0]) / sample_rate, len(list(signals.values())[0]))
    
    for i, (name, signal) in enumerate(signals.items(), 1):
        fig.add_trace(
            go.Scatter(x=time_axis, y=signal, name=name, line=dict(width=1)),
            row=i, col=1
        )
    
    fig.update_layout(
        height=200 * len(signals),
        showlegend=False,
        title="Audio Signals"
    )
    
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Amplitude")
    
    return fig


def main():
    # Header
    st.markdown('<h1 class="main-header">🎵 Speech Separation Demo</h1>', unsafe_allow_html=True)
    
    # Privacy disclaimer
    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
    st.markdown(PRIVACY_DISCLAIMER)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Model selection
    model_name = st.sidebar.selectbox(
        "Model",
        ["Conv-TasNet", "DPRNN"],
        help="Choose the speech separation model"
    )
    
    # Checkpoint selection
    checkpoint_path = st.sidebar.text_input(
        "Checkpoint Path (optional)",
        help="Path to trained model checkpoint"
    )
    
    # Audio parameters
    sample_rate = st.sidebar.selectbox(
        "Sample Rate",
        [8000, 16000, 22050, 44100],
        index=1,
        help="Audio sample rate"
    )
    
    snr_db = st.sidebar.slider(
        "Signal-to-Noise Ratio (dB)",
        min_value=0.0,
        max_value=30.0,
        value=10.0,
        step=1.0,
        help="SNR for synthetic mixtures"
    )
    
    # Load model
    try:
        model = load_model(model_name, checkpoint_path if checkpoint_path else None)
        device = get_device()
        st.sidebar.success(f"Model loaded on {device}")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🎤 Upload Audio", "🎵 Synthetic Demo", "📊 Analysis"])
    
    with tab1:
        st.header("Upload Audio Files")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Source 1")
            audio_file1 = st.file_uploader(
                "Upload first audio file",
                type=['wav', 'mp3', 'flac', 'm4a'],
                key="audio1"
            )
        
        with col2:
            st.subheader("Source 2")
            audio_file2 = st.file_uploader(
                "Upload second audio file",
                type=['wav', 'mp3', 'flac', 'm4a'],
                key="audio2"
            )
        
        if audio_file1 and audio_file2:
            try:
                # Process audio files
                signal1 = process_audio_file(audio_file1, sample_rate)
                signal2 = process_audio_file(audio_file2, sample_rate)
                
                # Create mixture
                mixed, s1_scaled, s2_scaled = create_synthetic_mixture(signal1, signal2, snr_db)
                
                # Separate
                estimates = separate_speech(model, mixed, device)
                
                # Compute metrics
                targets = np.stack([s1_scaled, s2_scaled])
                metrics = compute_metrics(estimates, targets)
                
                # Display results
                st.success("Separation completed!")
                
                # Audio playback
                st.subheader("Audio Playback")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.audio(signal1, sample_rate=sample_rate)
                    st.caption("Original Source 1")
                
                with col2:
                    st.audio(signal2, sample_rate=sample_rate)
                    st.caption("Original Source 2")
                
                with col3:
                    st.audio(mixed, sample_rate=sample_rate)
                    st.caption("Mixed Signal")
                
                with col4:
                    st.audio(estimates[0], sample_rate=sample_rate)
                    st.caption("Separated Source 1")
                
                # Metrics
                st.subheader("Separation Quality")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("SI-SDR", f"{metrics['si_sdr']:.2f} dB")
                
                with col2:
                    st.metric("PESQ", f"{metrics['pesq']:.2f}")
                
                with col3:
                    st.metric("STOI", f"{metrics['stoi']:.3f}")
                
                with col4:
                    st.metric("SDR", f"{metrics['sdr']:.2f} dB")
                
                # Visualization
                st.subheader("Waveform Visualization")
                signals_to_plot = {
                    "Original Source 1": signal1,
                    "Original Source 2": signal2,
                    "Mixed Signal": mixed,
                    "Separated Source 1": estimates[0],
                    "Separated Source 2": estimates[1]
                }
                
                fig = plot_audio_signals(signals_to_plot, sample_rate)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error processing audio: {e}")
    
    with tab2:
        st.header("Synthetic Speech Separation Demo")
        
        st.write("This demo generates synthetic speech-like signals and demonstrates separation.")
        
        if st.button("Generate Synthetic Demo"):
            try:
                # Generate synthetic signals
                duration = 3.0  # seconds
                t = np.linspace(0, duration, int(duration * sample_rate))
                
                # Generate two different frequency signals
                freq1 = 440  # A4 note
                freq2 = 523  # C5 note
                
                signal1 = np.sin(2 * np.pi * freq1 * t) * 0.5
                signal2 = np.sin(2 * np.pi * freq2 * t) * 0.5
                
                # Add some harmonics
                signal1 += 0.2 * np.sin(2 * np.pi * freq1 * 2 * t)
                signal2 += 0.2 * np.sin(2 * np.pi * freq2 * 2 * t)
                
                # Create mixture
                mixed, s1_scaled, s2_scaled = create_synthetic_mixture(signal1, signal2, snr_db)
                
                # Separate
                estimates = separate_speech(model, mixed, device)
                
                # Compute metrics
                targets = np.stack([s1_scaled, s2_scaled])
                metrics = compute_metrics(estimates, targets)
                
                st.success("Synthetic separation completed!")
                
                # Audio playback
                st.subheader("Audio Playback")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.audio(signal1, sample_rate=sample_rate)
                    st.caption("Synthetic Source 1 (440 Hz)")
                
                with col2:
                    st.audio(signal2, sample_rate=sample_rate)
                    st.caption("Synthetic Source 2 (523 Hz)")
                
                with col3:
                    st.audio(mixed, sample_rate=sample_rate)
                    st.caption("Mixed Signal")
                
                with col4:
                    st.audio(estimates[0], sample_rate=sample_rate)
                    st.caption("Separated Source 1")
                
                # Metrics
                st.subheader("Separation Quality")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("SI-SDR", f"{metrics['si_sdr']:.2f} dB")
                
                with col2:
                    st.metric("PESQ", f"{metrics['pesq']:.2f}")
                
                with col3:
                    st.metric("STOI", f"{metrics['stoi']:.3f}")
                
                with col4:
                    st.metric("SDR", f"{metrics['sdr']:.2f} dB")
                
                # Visualization
                st.subheader("Waveform Visualization")
                signals_to_plot = {
                    "Synthetic Source 1": signal1,
                    "Synthetic Source 2": signal2,
                    "Mixed Signal": mixed,
                    "Separated Source 1": estimates[0],
                    "Separated Source 2": estimates[1]
                }
                
                fig = plot_audio_signals(signals_to_plot, sample_rate)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error in synthetic demo: {e}")
    
    with tab3:
        st.header("Model Analysis")
        
        st.subheader("Model Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Model Type:** {model_name}")
            st.write(f"**Device:** {device}")
            st.write(f"**Sample Rate:** {sample_rate} Hz")
        
        with col2:
            st.write(f"**Parameters:** {sum(p.numel() for p in model.parameters()):,}")
            st.write(f"**SNR:** {snr_db} dB")
        
        st.subheader("About Speech Separation")
        st.write("""
        Speech separation, also known as the cocktail party problem, involves separating 
        multiple overlapping speech signals from a mixture. This demo showcases two popular 
        deep learning approaches:
        
        - **Conv-TasNet**: A fully convolutional time-domain audio separation network
        - **DPRNN**: Dual-Path RNN for efficient long sequence modeling
        
        The models are evaluated using several metrics:
        - **SI-SDR**: Scale-Invariant Signal-to-Distortion Ratio
        - **PESQ**: Perceptual Evaluation of Speech Quality
        - **STOI**: Short-Time Objective Intelligibility
        - **SDR**: Signal-to-Distortion Ratio
        """)


if __name__ == "__main__":
    main()
