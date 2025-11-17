# BeatSwap

A PyTorch-based deep learning model for separating audio into voice and beat components. Takes one audio file as input and returns two separate outputs: isolated voice and isolated beat tracks.

## Features

- **Modular Architecture**: Separated encoder-decoder architecture with reusable components
- **LeakyReLU Activation**: Uses LeakyReLU throughout the network for better gradient flow
- **Adam Optimizer**: Efficient Adam optimizer for training
- **Binary Cross-Entropy Loss**: BCELoss for training the separation task
- **Single File Processing**: Easy inference on individual audio files
- **Flexible Training**: Supports custom datasets and hyperparameters

## Project Structure

```
Beatswap/
├── models/              # Model components
│   ├── encoder.py      # AudioEncoder module
│   ├── decoder.py      # AudioDecoder module
│   └── beatswap.py     # Main BeatSwapModel
├── data/               # Data handling
│   ├── dataset.py      # AudioDataset class
│   └── preprocessing.py # Audio preprocessing utilities
├── train.py            # Training script
├── inference.py        # Inference script
├── config.py           # Configuration constants
├── utils.py            # Utility functions
└── quick_start.py      # Quick start example script
```

## Installation

1. **Clone or navigate to the project directory:**
```bash
cd Beatswap
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Setup (Optional)

Create necessary directories:
```bash
python quick_start.py --mode setup
```

This creates: `checkpoints/`, `outputs/`, `train_audio/`, `val_audio/`

### 2. Training on a Single Audio File

If you have a single audio file for training:

**Option A: Using quick_start.py (Recommended)**
```bash
# Place your audio file in a directory
mkdir audio_files
# Move your audio file to audio_files/

# Train
python quick_start.py --mode train --audio-dir ./audio_files --epochs 50 --batch-size 1
```

**Option B: Using train.py directly**
```bash
python train.py --train-dir ./audio_files --num-epochs 50 --batch-size 1
```

The model will use this single file for training. For better results, you'll want multiple audio files.

### Training with Multiple Audio Files

1. **Prepare your training data:**
   - Create a directory with your training audio files (e.g., `train_audio/`)
   - Supported formats: `.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`
   - Optionally create a validation directory: `val_audio/`

2. **Start training:**
```bash
python train.py --train-dir ./train_audio --val-dir ./val_audio
```

**Training options:**
- `--train-dir`: Directory containing training audio files (required)
- `--val-dir`: Directory containing validation audio files (optional)
- `--batch-size`: Batch size (default: 8)
- `--learning-rate`: Learning rate (default: 0.001)
- `--num-epochs`: Number of training epochs (default: 100)
- `--device`: Device to use - 'cuda' or 'cpu' (default: auto-detect)
- `--num-workers`: Number of data loading workers (default: 4)
- `--save-every`: Save checkpoint every N epochs (default: 10)

**Example:**
```bash
python train.py \
    --train-dir ./train_audio \
    --val-dir ./val_audio \
    --batch-size 4 \
    --learning-rate 0.0001 \
    --num-epochs 200 \
    --device cuda
```

Training checkpoints are saved in the `checkpoints/` directory. The best model (based on validation loss) is saved as `best_model.pth`.

### Inference (Separating Audio)

Once you have a trained model, separate audio into voice and beat:

```bash
python inference.py \
    --input-audio song.wav \
    --checkpoint checkpoints/best_model.pth \
    --output-dir outputs/
```

**Inference options:**
- `--input-audio`: Path to input audio file (required)
- `--checkpoint`: Path to model checkpoint (optional, but recommended)
- `--output-dir`: Output directory (default: `outputs/`)
- `--device`: Device to use - 'cuda' or 'cpu' (default: auto-detect)
- `--denormalize`: Denormalize output audio (default: True)

**Example:**
```bash
python inference.py \
    --input-audio my_song.wav \
    --checkpoint checkpoints/best_model.pth \
    --output-dir separated_audio/
```

This will create two files in the output directory:
- `my_song_voice.wav` - Isolated voice track
- `my_song_beat.wav` - Isolated beat track

### Quick Start Script

For convenience, use the `quick_start.py` script:

```bash
python quick_start.py --mode train --audio-dir ./audio_files
python quick_start.py --mode infer --input song.wav --checkpoint checkpoints/best_model.pth
```

## Configuration

Edit `config.py` to modify default settings:

```python
MODEL_CONFIG = {
    'input_channels': 1,
    'hidden_dim': 256,
    'leaky_relu_slope': 0.2,
}

TRAINING_CONFIG = {
    'batch_size': 8,
    'learning_rate': 0.001,
    'num_epochs': 100,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

AUDIO_CONFIG = {
    'sample_rate': 22050,
    'num_samples': 22050 * 10,  # 10 seconds default
    'normalize': True,
}
```

## Model Architecture

The BeatSwap model consists of:
- **AudioEncoder**: Processes input audio into a latent representation using 1D convolutions
- **VoiceDecoder**: Reconstructs voice component from latent representation
- **BeatDecoder**: Reconstructs beat component from latent representation

Both decoders share the same encoder output, allowing the model to learn separate voice and beat representations.

## Training Tips

1. **Data Preparation**: 
   - Use high-quality audio files (preferably WAV format)
   - Ensure consistent sample rates (model uses 22050 Hz by default)
   - More training data generally leads to better results

2. **Hyperparameters**:
   - Start with default learning rate (0.001) and adjust if training is unstable
   - Use smaller batch sizes (1-4) if you have limited memory
   - Training may take many epochs to converge

3. **Hardware**:
   - GPU recommended for faster training
   - CPU works but will be slower

## Output

- **Training**: Checkpoints saved in `checkpoints/` directory
- **Inference**: Separated audio files saved in `outputs/` directory (or specified output directory)

## Troubleshooting

**Issue: "No audio files found"**
- Make sure audio files are in the specified directory
- Check that file extensions are supported (.wav, .mp3, .flac, .m4a, .ogg)

**Issue: CUDA out of memory**
- Reduce batch size (`--batch-size 1` or `--batch-size 2`)
- Reduce `num_samples` in `config.py` if using fixed-length audio

**Issue: Poor separation quality**
- Train for more epochs
- Use more diverse training data
- Adjust learning rate if loss is not decreasing

## License

[Add your license here]

## Contributing

[Add contributing guidelines here]

