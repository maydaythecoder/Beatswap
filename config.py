"""Configuration constants for BeatSwap model training and inference."""

MODEL_CONFIG = {
    'input_channels': 1,
    'hidden_dim': 256,
    'leaky_relu_slope': 0.2,
}

TRAINING_CONFIG = {
    'batch_size': 8,
    'learning_rate': 0.001,
    'num_epochs': 100,
    'device': 'cuda' if __import__('torch').cuda.is_available() else 'cpu',
}

AUDIO_CONFIG = {
    'sample_rate': 22050,
    'num_samples': 22050 * 10,  # 10 seconds default
    'normalize': True,
}

PATHS = {
    'checkpoints_dir': 'checkpoints',
    'outputs_dir': 'outputs',
}

