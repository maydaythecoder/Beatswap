"""Dataset class for loading audio files."""

import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
from .preprocessing import preprocess_audio


class AudioDataset(Dataset):
    """Dataset for loading audio files with voice and beat pairs."""
    
    def __init__(self, data_dir, target_sample_rate=22050, target_length=None, mono=True, normalize=True):
        """
        Args:
            data_dir: Directory containing audio files
            target_sample_rate: Target sample rate for audio
            target_length: Target length in samples (None for original length)
            mono: Convert to mono if True
            normalize: Normalize to [0, 1] range if True
        """
        self.data_dir = Path(data_dir)
        self.target_sample_rate = target_sample_rate
        self.target_length = target_length
        self.mono = mono
        self.normalize = normalize
        
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
        self.audio_files = [
            f for f in os.listdir(data_dir) 
            if Path(f).suffix.lower() in audio_extensions
        ]
        
        if len(self.audio_files) == 0:
            raise ValueError(f"No audio files found in {data_dir}")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        audio_path = self.data_dir / audio_file
        
        waveform, _ = preprocess_audio(
            str(audio_path),
            target_sample_rate=self.target_sample_rate,
            target_length=self.target_length,
            mono=self.mono,
            normalize=self.normalize
        )
        
        return waveform

