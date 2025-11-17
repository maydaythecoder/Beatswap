"""Audio preprocessing utilities."""

import torch
import torchaudio


def preprocess_audio(audio_path, target_sample_rate=22050, target_length=None, mono=True, normalize=True):
    """
    Preprocess audio file for model input.
    
    Args:
        audio_path: Path to audio file
        target_sample_rate: Target sample rate
        target_length: Target length in samples (None for original length)
        mono: Convert to mono if True
        normalize: Normalize to [0, 1] range if True
    
    Returns:
        audio_tensor: Preprocessed audio tensor of shape (channels, samples)
        original_stats: Dictionary with original min/max for denormalization
    """
    waveform, sample_rate = torchaudio.load(audio_path)
    
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
    
    original_stats = {
        'min': waveform.min().item(),
        'max': waveform.max().item(),
        'sample_rate': target_sample_rate,
    }
    
    if normalize:
        audio_min = waveform.min()
        audio_max = waveform.max()
        if audio_max - audio_min > 0:
            waveform = (waveform - audio_min) / (audio_max - audio_min)
    
    return waveform, original_stats


def postprocess_audio(audio_tensor, original_stats=None, denormalize=True):
    """
    Postprocess model output audio.
    
    Args:
        audio_tensor: Model output tensor of shape (channels, samples) or (1, channels, samples)
        original_stats: Dictionary with original min/max for denormalization
        denormalize: Denormalize from [0, 1] if True
    
    Returns:
        audio_tensor: Postprocessed audio tensor
    """
    if audio_tensor.dim() == 3 and audio_tensor.shape[0] == 1:
        audio_tensor = audio_tensor.squeeze(0)
    
    if denormalize and original_stats is not None:
        audio_min = original_stats['min']
        audio_max = original_stats['max']
        audio_tensor = audio_tensor * (audio_max - audio_min) + audio_min
    
    return audio_tensor

