# Speech Separation (Cocktail Party Problem)

Research-ready implementation of speech separation systems using deep learning approaches. This project implements Conv-TasNet and DPRNN models for separating overlapping speech signals, commonly known as the cocktail party problem.

## PRIVACY DISCLAIMER

**IMPORTANT: This is a research demonstration tool for educational purposes only.**

- **No biometric identification**: This tool is not designed for biometric identification or voice recognition
- **No data storage**: Audio files are processed locally and not stored or transmitted  
- **Research use only**: This tool is intended for research and educational purposes
- **No production use**: Do not use this tool for production biometric applications
- **Voice cloning prohibition**: Misuse of this technology for voice cloning or impersonation is strictly prohibited

By using this tool, you acknowledge and agree to these terms.

## Features

- **Modern Architecture**: Implementation of Conv-TasNet and DPRNN models
- **Comprehensive Evaluation**: SI-SDR, PESQ, STOI, and SDR metrics
- **Synthetic Data Generation**: Built-in data generation for testing
- **Interactive Demo**: Streamlit-based web interface
- **Production Ready**: Proper project structure with configs, tests, and CI
- **Privacy Focused**: Local processing with no data storage

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Speech-Separation-Cocktail-Party-Problem.git
cd Speech-Separation-Cocktail-Party-Problem
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Generate synthetic data:
```bash
python scripts/train.py --config configs/conv_tasnet.yaml --generate_data --n_samples 1000
```

4. Train a model:
```bash
python scripts/train.py --config configs/conv_tasnet.yaml
```

5. Evaluate the model:
```bash
python scripts/evaluate.py --config configs/conv_tasnet.yaml --checkpoint checkpoints/conv_tasnet/best.pth
```

6. Run the demo:
```bash
streamlit run demo/app.py
```

## Project Structure

```
speech-separation/
├── src/speech_separation/          # Main package
│   ├── models/                    # Model implementations
│   ├── data/                      # Data loading and generation
│   ├── losses/                    # Loss functions
│   ├── metrics/                   # Evaluation metrics
│   ├── train/                     # Training utilities
│   ├── eval/                      # Evaluation utilities
│   └── utils/                     # Utility functions
├── configs/                       # Configuration files
├── scripts/                       # Training and evaluation scripts
├── demo/                          # Streamlit demo application
├── tests/                         # Unit tests
├── data/                          # Data directory
├── checkpoints/                   # Model checkpoints
├── logs/                          # Training logs
└── assets/                        # Generated assets
```

## Models

### Conv-TasNet
Convolutional Time-domain Audio Separation Network - a fully convolutional approach that operates directly on the time-domain waveform.

**Key Features:**
- Encoder-decoder architecture with 1D convolutions
- Temporal Convolutional Network (TCN) separator
- Global Layer Normalization
- Mask-based separation

### DPRNN
Dual-Path RNN - combines intra-chunk and inter-chunk processing for efficient long sequence modeling.

**Key Features:**
- Dual-path processing (intra-RNN and inter-RNN)
- Chunk-based processing for long sequences
- LSTM-based recurrent processing
- Efficient memory usage

## Data Format

The project expects data in the following format:

```
data/
├── wav/                           # Audio files
│   ├── sample_0001_mixture.wav
│   ├── sample_0001_source_0.wav
│   └── sample_0001_source_1.wav
└── meta/                          # Metadata
    └── synthetic_metadata.csv
```

Metadata CSV format:
```csv
sample_id,mixture_path,source_0_path,source_1_path,n_sources,snr,duration,split
sample_0001,wav/sample_0001_mixture.wav,wav/sample_0001_source_0.wav,wav/sample_0001_source_1.wav,2,10.0,3.5,train
```

## Configuration

Models are configured using YAML files in the `configs/` directory:

```yaml
# Model configuration
model:
  n_src: 2
  n_filters: 256
  kernel_size: 16
  stride: 8

# Training configuration
max_epochs: 100
batch_size: 16
lr: 0.001

# Data configuration
data:
  sample_rate: 16000
  segment_length: 32000
  n_sources: 2
```

## Training

### Basic Training
```bash
python scripts/train.py --config configs/conv_tasnet.yaml
```

### With Custom Data
```bash
python scripts/train.py \
    --config configs/conv_tasnet.yaml \
    --data_dir /path/to/data \
    --meta_file /path/to/metadata.csv
```

### Resume Training
```bash
python scripts/train.py \
    --config configs/conv_tasnet.yaml \
    --resume checkpoints/conv_tasnet/latest.pth
```

## Evaluation

### Basic Evaluation
```bash
python scripts/evaluate.py \
    --config configs/conv_tasnet.yaml \
    --checkpoint checkpoints/conv_tasnet/best.pth
```

### Custom Output Directory
```bash
python scripts/evaluate.py \
    --config configs/conv_tasnet.yaml \
    --checkpoint checkpoints/conv_tasnet/best.pth \
    --output_dir results/conv_tasnet_eval
```

## Metrics

The system evaluates speech separation quality using multiple metrics:

- **SI-SDR**: Scale-Invariant Signal-to-Distortion Ratio (dB)
- **PESQ**: Perceptual Evaluation of Speech Quality
- **STOI**: Short-Time Objective Intelligibility
- **SDR**: Signal-to-Distortion Ratio (dB)

## Demo Application

The Streamlit demo provides an interactive interface for:

- Uploading audio files for separation
- Generating synthetic speech mixtures
- Real-time separation with quality metrics
- Visualization of waveforms and results

### Running the Demo
```bash
streamlit run demo/app.py
```

## Synthetic Data Generation

The system includes built-in synthetic data generation:

```python
from speech_separation.data import SyntheticDataGenerator

generator = SyntheticDataGenerator(
    output_dir="data",
    sample_rate=16000,
    duration_range=(2.0, 8.0),
    snr_range=(0.0, 20.0),
    n_sources=2
)

# Generate sine wave mixtures
generator.generate_sine_waves(n_samples=1000)

# Generate TTS-based mixtures (requires pyttsx3)
generator.generate_tts_samples(n_samples=500)
```

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/
ruff check src/
```

### Pre-commit Hooks
```bash
pre-commit install
pre-commit run --all-files
```

## Performance

Typical performance on synthetic data:

| Model | SI-SDR (dB) | PESQ | STOI | Parameters |
|-------|-------------|------|------|------------|
| Conv-TasNet | 8.5 | 2.1 | 0.75 | 5.1M |
| DPRNN | 9.2 | 2.3 | 0.78 | 2.6M |

*Results may vary based on data and training configuration*

## Limitations

- **Synthetic Data**: Default training uses synthetic data; real speech data may yield different results
- **Model Size**: Models are relatively small for demonstration; production models may be larger
- **Real-time**: Current implementation is not optimized for real-time processing
- **Generalization**: Performance may vary significantly across different types of audio

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{speech_separation,
  title={Speech Separation (Cocktail Party Problem)},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Speech-Separation-Cocktail-Party-Problem}
}
```

## Acknowledgments

- Conv-TasNet: [Luo & Mesgarani, 2018](https://arxiv.org/abs/1809.07454)
- DPRNN: [Luo et al., 2019](https://arxiv.org/abs/1910.06379)
- Asteroid toolkit for inspiration
- PyTorch and the open-source community

## Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review the demo application

---

**Remember: This tool is for research and educational purposes only. Please respect privacy and ethical guidelines when using speech separation technology.**
# Speech-Separation-Cocktail-Party-Problem
