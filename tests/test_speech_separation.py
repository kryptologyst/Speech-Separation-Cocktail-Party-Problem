"""Tests for speech separation package."""

import pytest
import torch
import numpy as np
from speech_separation.models import ConvTasNet, DPRNN
from speech_separation.losses import SISDRLoss, CombinedLoss
from speech_separation.metrics import SISDRMetric, MetricCalculator
from speech_separation.utils import get_device, normalize_audio, mix_signals


class TestModels:
    """Test model implementations."""
    
    def test_conv_tasnet_forward(self):
        """Test Conv-TasNet forward pass."""
        model = ConvTasNet(n_src=2, n_filters=64, kernel_size=8, stride=4)
        batch_size = 2
        samples = 16000
        
        # Create dummy input
        x = torch.randn(batch_size, samples)
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        assert output.shape == (batch_size, 2, samples)
        assert not torch.isnan(output).any()
    
    def test_dprnn_forward(self):
        """Test DPRNN forward pass."""
        model = DPRNN(n_src=2, n_filters=32, kernel_size=8, stride=4)
        batch_size = 2
        samples = 16000
        
        # Create dummy input
        x = torch.randn(batch_size, samples)
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        assert output.shape == (batch_size, 2, samples)
        assert not torch.isnan(output).any()
    
    def test_model_parameters(self):
        """Test that models have reasonable number of parameters."""
        conv_tasnet = ConvTasNet(n_src=2, n_filters=64, kernel_size=8, stride=4)
        dprnn = DPRNN(n_src=2, n_filters=32, kernel_size=8, stride=4)
        
        # Check parameter counts are reasonable
        assert conv_tasnet.num_parameters() > 1000
        assert dprnn.num_parameters() > 1000


class TestLosses:
    """Test loss functions."""
    
    def test_si_sdr_loss(self):
        """Test SI-SDR loss computation."""
        loss_fn = SISDRLoss()
        batch_size = 2
        n_src = 2
        samples = 16000
        
        # Create dummy data
        estimate = torch.randn(batch_size, n_src, samples)
        target = torch.randn(batch_size, n_src, samples)
        
        # Compute loss
        loss = loss_fn(estimate, target)
        
        # Check loss is scalar and finite
        assert loss.dim() == 0
        assert torch.isfinite(loss)
    
    def test_combined_loss(self):
        """Test combined loss function."""
        loss_fn = CombinedLoss()
        batch_size = 2
        n_src = 2
        samples = 16000
        
        # Create dummy data
        estimate = torch.randn(batch_size, n_src, samples)
        target = torch.randn(batch_size, n_src, samples)
        
        # Compute loss
        loss, loss_dict = loss_fn(estimate, target)
        
        # Check loss is scalar and finite
        assert loss.dim() == 0
        assert torch.isfinite(loss)
        
        # Check loss dictionary
        assert "total_loss" in loss_dict
        assert "si_sdr_loss" in loss_dict
        assert "mse_loss" in loss_dict


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_si_sdr_metric(self):
        """Test SI-SDR metric computation."""
        metric = SISDRMetric()
        batch_size = 2
        n_src = 2
        samples = 16000
        
        # Create dummy data
        estimate = torch.randn(batch_size, n_src, samples)
        target = torch.randn(batch_size, n_src, samples)
        
        # Compute metric
        si_sdr = metric(estimate, target)
        
        # Check output shape
        assert si_sdr.shape == (batch_size, n_src)
        assert torch.isfinite(si_sdr).all()
    
    def test_metric_calculator(self):
        """Test metric calculator."""
        calculator = MetricCalculator()
        batch_size = 2
        n_src = 2
        samples = 16000
        
        # Create dummy data
        estimate = torch.randn(batch_size, n_src, samples)
        target = torch.randn(batch_size, n_src, samples)
        
        # Compute metrics
        metrics = calculator(estimate, target)
        
        # Check metrics
        assert "si_sdr" in metrics
        assert "pesq" in metrics
        assert "stoi" in metrics
        assert "sdr" in metrics
        
        # Check shapes
        for metric_name, values in metrics.items():
            assert values.shape == (batch_size, n_src)


class TestUtils:
    """Test utility functions."""
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
    
    def test_normalize_audio(self):
        """Test audio normalization."""
        # Test normal case
        audio = torch.randn(1000) * 2.0
        normalized = normalize_audio(audio)
        
        assert torch.max(torch.abs(normalized)) <= 1.0 + 1e-6
        
        # Test zero case
        audio_zero = torch.zeros(1000)
        normalized_zero = normalize_audio(audio_zero)
        assert torch.allclose(normalized_zero, audio_zero)
    
    def test_mix_signals(self):
        """Test signal mixing."""
        signal1 = torch.randn(1000)
        signal2 = torch.randn(1000)
        snr_db = 10.0
        
        mixed, s1_scaled, s2_scaled = mix_signals(signal1, signal2, snr_db)
        
        # Check shapes
        assert mixed.shape == signal1.shape
        assert s1_scaled.shape == signal1.shape
        assert s2_scaled.shape == signal2.shape
        
        # Check that mixed signal is sum of scaled signals
        assert torch.allclose(mixed, s1_scaled + s2_scaled)


if __name__ == "__main__":
    pytest.main([__file__])
