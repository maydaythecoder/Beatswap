"""Inference script for processing single audio files."""

import argparse
import torch
from pathlib import Path

from models import BeatSwapModel
from data.preprocessing import preprocess_audio, postprocess_audio
from config import MODEL_CONFIG, AUDIO_CONFIG, PATHS
from utils import ensure_dir, save_audio


def separate_audio(model, audio_path, output_dir, device, denormalize=True):
    """
    Separate audio into voice and beat components.
    
    Args:
        model: Trained BeatSwapModel
        audio_path: Path to input audio file
        output_dir: Directory to save output files
        device: Device to run inference on
        denormalize: Whether to denormalize output audio
    """
    model.eval()
    
    ensure_dir(output_dir)
    
    input_path = Path(audio_path)
    audio_tensor, original_stats = preprocess_audio(
        str(audio_path),
        target_sample_rate=AUDIO_CONFIG['sample_rate'],
        target_length=AUDIO_CONFIG.get('num_samples'),
        normalize=AUDIO_CONFIG['normalize']
    )
    
    audio_tensor = audio_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        voice_pred, beat_pred = model(audio_tensor)
    
    voice_output = voice_pred.squeeze(0).cpu()
    beat_output = beat_pred.squeeze(0).cpu()
    
    if denormalize:
        voice_output = postprocess_audio(voice_output, original_stats, denormalize=True)
        beat_output = postprocess_audio(beat_output, original_stats, denormalize=True)
    
    sample_rate = original_stats['sample_rate']
    
    voice_path = Path(output_dir) / f"{input_path.stem}_voice{input_path.suffix}"
    beat_path = Path(output_dir) / f"{input_path.stem}_beat{input_path.suffix}"
    
    save_audio(voice_output, str(voice_path), sample_rate)
    save_audio(beat_output, str(beat_path), sample_rate)
    
    print(f"Voice saved to: {voice_path}")
    print(f"Beat saved to: {beat_path}")


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = BeatSwapModel(**MODEL_CONFIG).to(device)
    
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        print("Warning: No checkpoint provided. Using randomly initialized model.")
    
    separate_audio(
        model,
        args.input_audio,
        args.output_dir,
        device,
        denormalize=args.denormalize
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Separate audio into voice and beat')
    parser.add_argument('--input-audio', type=str, required=True, help='Path to input audio file')
    parser.add_argument('--output-dir', type=str, default=PATHS['outputs_dir'], help='Output directory')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--denormalize', action='store_true', default=True, help='Denormalize output audio')
    
    args = parser.parse_args()
    main(args)

