# Mirokaï DOA Research Robotics

A state-of-the-art Direction of Arrival (DOA) estimation system for robotics applications using deep learning models with multichannel audio processing.

## ⚠️ Project Status

**This project is currently in progress.** Only DOA basics have been tested and validated. The system is under active development.

## Overview

This project implements deep learning models for estimating the direction of arrival of speech signals using a 4-channel microphone array. The system processes multichannel audio signals and predicts the azimuth angle of sound sources in real-time.

### Key Features

- **State-of-the-art DOA models**: SCATTiny, FiLMMixerSRP, and ReTiNDoA architectures
- **SRP-PHAT baseline**: Steered Response Power with Phase Transform for comparison
- **Real-time processing**: Optimized for low-latency inference
- **Feature extraction**: Multichannel STFT and advanced feature computation
- **VAD integration**: Voice Activity Detection for improved accuracy
- **Flexible training**: Support for on-the-fly data generation and precomputed features

## Architecture

The project supports three main model architectures:

1. **SCATTiny**: SRP-conditioned additive cross-transformer with attention mechanisms
2. **FiLMMixerSRP**: Time-only Mixer with SRP Feature-wise Linear Modulation (FiLM) conditioning
3. **ReTiNDoA**: Retentive cell-based architecture for temporal modeling

All models use a common backbone that pools channel-frequency features and applies MLP layers to produce per-time embeddings.

## Prerequisites

- Python 3.8+
- PyTorch
- NumPy, SciPy
- See `requirements.txt` for full dependencies

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd doa-research-robotics
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. **Important**: Generate Room Impulse Responses (RIR) using the separate `rir_generator` project:
   - The RIR generator should be set up and run to create the RIR bank
   - Update the `rir_root` path in your configuration files to point to the generated RIR bank

## Project Structure

```
doa-research-robotics/
├── configs/              # Configuration YAML files
│   ├── train.yaml        # Training configuration
│   ├── train_full.yaml   # Full training configuration
│   ├── synth.yaml        # Synthetic data configuration
│   ├── realtime.yaml     # Real-time inference configuration
│   └── constraint.yaml   # Constraint-based training
├── scripts/              # Shell scripts
│   ├── train_synth.sh    # Train on synthetic data
│   ├── make_synth.sh     # Generate synthetic mixtures
│   ├── gen_pseudolabels.sh  # Generate pseudo labels
│   └── rt_demo.sh        # Real-time demonstration
├── src/
│   └── mirokai_doa/      # Main source code
│       ├── train.py      # Training script
│       ├── models.py     # Model definitions
│       ├── features.py   # Feature extraction
│       ├── srp.py        # SRP-PHAT implementation
│       ├── losses.py    # Loss functions
│       ├── train_utils.py # Training utilities
│       ├── mix_batcher.py # Data batching
│       ├── precompute_features.py # Feature preprocessing
│       ├── realtime.py   # Real-time inference
│       ├── vad.py        # Voice Activity Detection
│       └── models/
│           └── silero_vad.onnx  # VAD model
├── tests/                # Test files and sample data
├── models/               # Trained model checkpoints (empty initially)
└── requirements.txt      # Python dependencies
```

## Configuration

The project uses YAML configuration files located in `configs/`. Key configuration sections:

- **dataset**: Paths to RIR bank, speech, noise, and ambiance datasets
- **features**: STFT parameters, sampling rate, azimuth resolution (K bins)
- **microphone**: Array geometry (4-channel Seeed microphone array v2)
- **model**: Architecture-specific hyperparameters

Example configuration structure:
```yaml
dataset:
  rir_root: "/path/to/rir_generator/rir_bank"
  speech_root: "/path/to/LibriSpeech"
  noise_root: "/path/to/noise"
  batch_size: 4

features:
  sr: 16000
  win_s: 0.032
  hop_s: 0.010
  K: 72  # 5° resolution (360/72)

microphone:
  positions:
    - [0.0277, 0.0]    # Mic 0: 0°
    - [0.0, 0.0277]    # Mic 1: 90°
    - [-0.0277, 0.0]   # Mic 2: 180°
    - [0.0, -0.0277]   # Mic 3: 270°
```

## Usage

### Training

Train a model using one of the available architectures:

```bash
python src/mirokai_doa/train.py \
    --model scat \
    --cfg configs/train.yaml \
    --save-root models
```

Available models: `scat`, `film`, `retin`

### Feature Precomputation

Precompute features for faster training:

```bash
python src/mirokai_doa/precompute_features.py \
    --cfg configs/train.yaml \
    --output-dir features_v1
```

### Real-time Inference

Run real-time DOA estimation:

```bash
python src/mirokai_doa/realtime.py \
    --cfg configs/realtime.yaml \
    --checkpoint models/scat/checkpoint.pth
```

### Using Shell Scripts

```bash
# Train on synthetic data
./scripts/train_synth.sh

# Generate synthetic mixtures
./scripts/make_synth.sh

# Real-time demo
./scripts/rt_demo.sh
```

## RIR Generation

**Important**: Room Impulse Responses (RIR) must be generated using the separate `rir_generator` project before training:

1. Set up and run the `rir_generator` project
2. Generate the RIR bank with appropriate room configurations
3. Update the `rir_root` path in your configuration files
4. Ensure the RIR bank structure matches the expected format

The RIR generator project should be pushed/available separately and is required for proper dataset generation.

## Testing

Run tests to verify feature extraction and SRP functionality:

```bash
# Test feature extraction
python -m pytest tests/test_features.py

# Test SRP-PHAT
python -m pytest tests/test_srp.py
```

## Model Details

### SCATTiny
- SRP-conditioned cross-attention mechanism
- Pooled channel-frequency features with MLP backbone
- Additive cross-attention between SRP prototypes and feature tokens

### FiLMMixerSRP
- Feature-wise Linear Modulation (FiLM) from SRP features
- Temporal 1D mixer blocks with residual connections
- Per-time classification head

### ReTiNDoA
- Retentive cell-based temporal modeling
- Unrolled over time for efficient inference
- Optional delta angle regression

## Development Status

- ✅ DOA basics implemented and tested
- ✅ Model architectures defined
- ✅ Training pipeline functional
- ✅ Feature extraction working
- ⚠️ RIR generation requires external `rir_generator` project
- 🚧 Advanced features in development
- 🚧 Full evaluation pipeline pending

## Contributing

This is a research project in active development. Contributions and feedback are welcome.

## License


## Citation

If you use this code in your research, please cite:

```bibtex
@software{mirokai_doa,
  title = {Mirokaï DOA Research Robotics},
  author = {[Your Name]},
  year = {2025},
  url = {[Repository URL]}
}
```

## Acknowledgments

- Seeed Studio for microphone array hardware specifications
- Silero VAD for voice activity detection
- LibriSpeech for speech datasets
