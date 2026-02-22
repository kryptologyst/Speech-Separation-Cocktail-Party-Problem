"""Speech separation models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class Conv1dBlock(nn.Module):
    """1D Convolutional block with normalization and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        norm_type: str = "batch",
        activation: str = "relu",
    ):
        super().__init__()
        
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding, 
            dilation, groups, bias
        )
        
        if norm_type == "batch":
            self.norm = nn.BatchNorm1d(out_channels)
        elif norm_type == "layer":
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = None
            
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class ConvTranspose1dBlock(nn.Module):
    """1D Transpose Convolutional block with normalization and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        norm_type: str = "batch",
        activation: str = "relu",
    ):
        super().__init__()
        
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride, padding,
            output_padding, dilation, groups, bias
        )
        
        if norm_type == "batch":
            self.norm = nn.BatchNorm1d(out_channels)
        elif norm_type == "layer":
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = None
            
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_transpose(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class ConvTasNet(nn.Module):
    """Conv-TasNet model for speech separation.
    
    Conv-TasNet: A Fully Convolutional Time-Domain Audio Separation Network
    https://arxiv.org/abs/1809.07454
    """
    
    def __init__(
        self,
        n_src: int = 2,
        n_filters: int = 256,
        kernel_size: int = 16,
        stride: int = 8,
        n_repeats: int = 3,
        n_blocks: int = 8,
        bn_chan: int = 128,
        hid_chan: int = 512,
        skip_chan: int = 128,
        causal: bool = False,
        norm_type: str = "gLN",
        mask_act: str = "relu",
    ):
        super().__init__()
        
        self.n_src = n_src
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = stride
        
        # Encoder
        self.encoder = nn.Conv1d(1, n_filters, kernel_size, stride=stride)
        
        # Separator
        self.separator = Separator(
            n_filters=n_filters,
            n_repeats=n_repeats,
            n_blocks=n_blocks,
            bn_chan=bn_chan,
            hid_chan=hid_chan,
            skip_chan=skip_chan,
            causal=causal,
            norm_type=norm_type,
        )
        
        # Decoder
        self.decoder = nn.ConvTranspose1d(
            n_filters, 1, kernel_size, stride=stride
        )
        
        # Mask activation
        if mask_act == "relu":
            self.mask_act = nn.ReLU()
        elif mask_act == "softmax":
            self.mask_act = nn.Softmax(dim=1)
        else:
            self.mask_act = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input mixture [batch, samples]
            
        Returns:
            Separated sources [batch, n_src, samples]
        """
        batch_size = x.shape[0]
        
        # Encoder
        w = self.encoder(x.unsqueeze(1))  # [batch, n_filters, frames]
        
        # Separator
        masks = self.separator(w)  # [batch, n_src, n_filters, frames]
        
        # Apply masks
        masked_w = w.unsqueeze(1) * masks  # [batch, n_src, n_filters, frames]
        
        # Decoder
        separated = []
        for i in range(self.n_src):
            separated.append(
                self.decoder(masked_w[:, i]).squeeze(1)
            )
        
        return torch.stack(separated, dim=1)


class Separator(nn.Module):
    """Separator network for Conv-TasNet."""
    
    def __init__(
        self,
        n_filters: int,
        n_repeats: int,
        n_blocks: int,
        bn_chan: int,
        hid_chan: int,
        skip_chan: int,
        causal: bool = False,
        norm_type: str = "gLN",
    ):
        super().__init__()
        
        self.n_repeats = n_repeats
        self.n_blocks = n_blocks
        
        # Bottleneck
        self.bottleneck = nn.Conv1d(n_filters, bn_chan, 1)
        
        # TCN blocks
        self.tcn_blocks = nn.ModuleList()
        for r in range(n_repeats):
            for b in range(n_blocks):
                dilation = 2 ** b
                self.tcn_blocks.append(
                    TCNBlock(
                        bn_chan, hid_chan, skip_chan, dilation, causal, norm_type
                    )
                )
        
        # Output layer
        self.output = nn.Conv1d(bn_chan, n_filters, 1)
        
        # Mask layer
        self.mask_layer = nn.Conv1d(n_filters, n_filters, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input features [batch, n_filters, frames]
            
        Returns:
            Masks [batch, n_src, n_filters, frames]
        """
        # Bottleneck
        x = self.bottleneck(x)
        
        # TCN blocks
        skip_connections = []
        for block in self.tcn_blocks:
            x, skip = block(x)
            skip_connections.append(skip)
        
        # Sum skip connections
        x = sum(skip_connections)
        
        # Output
        x = self.output(x)
        
        # Generate masks
        masks = self.mask_layer(x)
        
        return masks.unsqueeze(1)  # Add source dimension


class TCNBlock(nn.Module):
    """Temporal Convolutional Network block."""
    
    def __init__(
        self,
        in_chan: int,
        hid_chan: int,
        skip_chan: int,
        dilation: int,
        causal: bool = False,
        norm_type: str = "gLN",
    ):
        super().__init__()
        
        self.causal = causal
        
        # Depthwise convolution
        if causal:
            padding = dilation
        else:
            padding = dilation
        
        self.depthwise_conv = nn.Conv1d(
            in_chan, in_chan, 3, padding=padding, dilation=dilation, groups=in_chan
        )
        
        # Normalization
        if norm_type == "gLN":
            self.norm = GlobalLayerNorm(in_chan)
        elif norm_type == "cLN":
            self.norm = ChannelwiseLayerNorm(in_chan)
        else:
            self.norm = nn.BatchNorm1d(in_chan)
        
        # Pointwise convolutions
        self.pointwise_conv1 = nn.Conv1d(in_chan, hid_chan, 1)
        self.pointwise_conv2 = nn.Conv1d(hid_chan, in_chan, 1)
        self.skip_conv = nn.Conv1d(hid_chan, skip_chan, 1)
        
        # Activation
        self.activation = nn.PReLU()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input [batch, in_chan, frames]
            
        Returns:
            Tuple of (output, skip_connection)
        """
        residual = x
        
        # Depthwise convolution
        x = self.depthwise_conv(x)
        
        # Apply causal padding if needed
        if self.causal:
            x = x[..., :-self.depthwise_conv.padding[0]]
        
        # Normalization and activation
        x = self.norm(x)
        x = self.activation(x)
        
        # Pointwise convolutions
        x = self.pointwise_conv1(x)
        x = self.activation(x)
        skip = self.skip_conv(x)
        x = self.pointwise_conv2(x)
        
        # Residual connection
        x = x + residual
        
        return x, skip


class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization."""
    
    def __init__(self, num_features: int, eps: float = 1e-8):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Global normalization
        mean = torch.mean(x, dim=[1, 2], keepdim=True)
        var = torch.var(x, dim=[1, 2], keepdim=True)
        
        x = (x - mean) / torch.sqrt(var + self.eps)
        
        # Apply learnable parameters
        x = x * self.weight.unsqueeze(0).unsqueeze(-1)
        x = x + self.bias.unsqueeze(0).unsqueeze(-1)
        
        return x


class ChannelwiseLayerNorm(nn.Module):
    """Channel-wise Layer Normalization."""
    
    def __init__(self, num_features: int, eps: float = 1e-8):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Channel-wise normalization
        mean = torch.mean(x, dim=2, keepdim=True)
        var = torch.var(x, dim=2, keepdim=True)
        
        x = (x - mean) / torch.sqrt(var + self.eps)
        
        # Apply learnable parameters
        x = x * self.weight.unsqueeze(0).unsqueeze(-1)
        x = x + self.bias.unsqueeze(0).unsqueeze(-1)
        
        return x


class DPRNN(nn.Module):
    """Dual-Path RNN model for speech separation.
    
    Dual-Path RNN: Efficient Long Sequence Modeling for Speech Separation
    https://arxiv.org/abs/1910.06379
    """
    
    def __init__(
        self,
        n_src: int = 2,
        n_filters: int = 64,
        kernel_size: int = 16,
        stride: int = 8,
        n_repeats: int = 6,
        n_blocks: int = 3,
        rnn_hidden: int = 128,
        norm_type: str = "gLN",
        mask_act: str = "relu",
    ):
        super().__init__()
        
        self.n_src = n_src
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = stride
        
        # Encoder
        self.encoder = nn.Conv1d(1, n_filters, kernel_size, stride=stride)
        
        # Separator
        self.separator = DPRNNSeparator(
            n_filters=n_filters,
            n_repeats=n_repeats,
            n_blocks=n_blocks,
            rnn_hidden=rnn_hidden,
            norm_type=norm_type,
        )
        
        # Decoder
        self.decoder = nn.ConvTranspose1d(
            n_filters, 1, kernel_size, stride=stride
        )
        
        # Mask activation
        if mask_act == "relu":
            self.mask_act = nn.ReLU()
        elif mask_act == "softmax":
            self.mask_act = nn.Softmax(dim=1)
        else:
            self.mask_act = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch_size = x.shape[0]
        
        # Encoder
        w = self.encoder(x.unsqueeze(1))
        
        # Separator
        masks = self.separator(w)
        
        # Apply masks
        masked_w = w.unsqueeze(1) * masks
        
        # Decoder
        separated = []
        for i in range(self.n_src):
            separated.append(
                self.decoder(masked_w[:, i]).squeeze(1)
            )
        
        return torch.stack(separated, dim=1)


class DPRNNSeparator(nn.Module):
    """DPRNN Separator network."""
    
    def __init__(
        self,
        n_filters: int,
        n_repeats: int,
        n_blocks: int,
        rnn_hidden: int,
        norm_type: str = "gLN",
    ):
        super().__init__()
        
        self.n_repeats = n_repeats
        self.n_blocks = n_blocks
        
        # Intra-RNN and Inter-RNN blocks
        self.intra_rnn_blocks = nn.ModuleList()
        self.inter_rnn_blocks = nn.ModuleList()
        
        for r in range(n_repeats):
            for b in range(n_blocks):
                self.intra_rnn_blocks.append(
                    DPRNNBlock(n_filters, rnn_hidden, norm_type, "intra")
                )
                self.inter_rnn_blocks.append(
                    DPRNNBlock(n_filters, rnn_hidden, norm_type, "inter")
                )
        
        # Output layer
        self.output = nn.Conv1d(n_filters, n_filters, 1)
        
        # Mask layer
        self.mask_layer = nn.Conv1d(n_filters, n_filters, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch_size, n_filters, frames = x.shape
        
        # Reshape for dual-path processing
        x = x.view(batch_size, n_filters, frames // self.n_blocks, self.n_blocks)
        
        # Process through dual-path blocks
        for r in range(self.n_repeats):
            for b in range(self.n_blocks):
                # Intra-RNN (within segments)
                x = self.intra_rnn_blocks[r * self.n_blocks + b](x)
                
                # Inter-RNN (across segments)
                x = self.inter_rnn_blocks[r * self.n_blocks + b](x)
        
        # Reshape back
        x = x.view(batch_size, n_filters, frames)
        
        # Output
        x = self.output(x)
        
        # Generate masks
        masks = self.mask_layer(x)
        
        return masks.unsqueeze(1)


class DPRNNBlock(nn.Module):
    """DPRNN Block for intra or inter processing."""
    
    def __init__(
        self,
        n_filters: int,
        rnn_hidden: int,
        norm_type: str = "gLN",
        path_type: str = "intra",
    ):
        super().__init__()
        
        self.path_type = path_type
        
        # Normalization
        if norm_type == "gLN":
            self.norm = GlobalLayerNorm(n_filters)
        elif norm_type == "cLN":
            self.norm = ChannelwiseLayerNorm(n_filters)
        else:
            self.norm = nn.BatchNorm1d(n_filters)
        
        # RNN
        if path_type == "intra":
            # Process within segments
            self.rnn = nn.LSTM(n_filters, rnn_hidden, 1, batch_first=True)
            self.proj = nn.Linear(rnn_hidden, n_filters)
        else:
            # Process across segments
            self.rnn = nn.LSTM(n_filters, rnn_hidden, 1, batch_first=True)
            self.proj = nn.Linear(rnn_hidden, n_filters)
        
        # Activation
        self.activation = nn.PReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch_size, n_filters, segments, blocks = x.shape
        
        if self.path_type == "intra":
            # Process within segments
            x = x.permute(0, 2, 3, 1)  # [batch, segments, blocks, n_filters]
            x = x.contiguous().view(-1, blocks, n_filters)
            
            # Normalization
            x = self.norm(x.transpose(1, 2)).transpose(1, 2)
            
            # RNN
            x, _ = self.rnn(x)
            x = self.proj(x)
            
            # Reshape back
            x = x.view(batch_size, segments, blocks, n_filters)
            x = x.permute(0, 3, 1, 2)
            
        else:
            # Process across segments
            x = x.permute(0, 3, 2, 1)  # [batch, blocks, segments, n_filters]
            x = x.contiguous().view(-1, segments, n_filters)
            
            # Normalization
            x = self.norm(x.transpose(1, 2)).transpose(1, 2)
            
            # RNN
            x, _ = self.rnn(x)
            x = self.proj(x)
            
            # Reshape back
            x = x.view(batch_size, blocks, segments, n_filters)
            x = x.permute(0, 3, 2, 1)
        
        return x
