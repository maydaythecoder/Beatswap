"""Utility functions for BeatSwap project."""

import os
import torch
import torchaudio


def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)


def load_audio(file_path, target_sample_rate=22050, target_length=None, mono=True):
    """
    Load and preprocess audio file.
    
    Args:
        file_path: Path to audio file
        target_sample_rate: Target sample rate (default: 22050)
        target_length: Target length in samples (None for original length)
        mono: Convert to mono if True
    
    Returns:
        audio_tensor: Audio tensor of shape (channels, samples)
        sample_rate: Actual sample rate
    """
    waveform, sample_rate = torchaudio.load(file_path)
    
    if mono and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
        waveform = resampler(waveform)
    
    if target_length is not None:
        if waveform.shape[1] > target_length:
            waveform = waveform[:, :target_length]
        elif waveform.shape[1] < target_length:
            padding = target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
    
    return waveform, target_sample_rate


def normalize_audio(audio_tensor):
    """Normalize audio to [0, 1] range."""
    audio_min = audio_tensor.min()
    audio_max = audio_tensor.max()
    if audio_max - audio_min > 0:
        return (audio_tensor - audio_min) / (audio_max - audio_min)
    return audio_tensor


def save_audio(audio_tensor, file_path, sample_rate=22050):
    """
    Save audio tensor to file.
    
    Args:
        audio_tensor: Audio tensor of shape (channels, samples) or (batch, channels, samples)
        file_path: Output file path
        sample_rate: Sample rate for saved audio
    """
    if audio_tensor.dim() == 3:
        audio_tensor = audio_tensor.squeeze(0)
    
    torchaudio.save(file_path, audio_tensor, sample_rate)


def denormalize_audio(audio_tensor, original_min=0, original_max=1):
    """Denormalize audio from [0, 1] range back to original range."""
    return audio_tensor * (original_max - original_min) + original_min

