"""Quick start script for BeatSwap - easy training and inference."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def train_quick(audio_dir, epochs=50, batch_size=1):
    """Quick training setup with a single directory of audio files."""
    print(f"Starting training with audio files in: {audio_dir}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}\n")
    
    train_dir = Path(audio_dir)
    if not train_dir.exists():
        print(f"Error: Directory {audio_dir} does not exist!")
        return
    
    cmd = [
        sys.executable, "train.py",
        "--train-dir", str(train_dir),
        "--num-epochs", str(epochs),
        "--batch-size", str(batch_size),
    ]
    
    print(f"Running: {' '.join(cmd)}\n")
    subprocess.run(cmd)


def infer_quick(input_audio, checkpoint=None, output_dir="outputs"):
    """Quick inference on a single audio file."""
    print(f"Separating audio: {input_audio}")
    
    input_path = Path(input_audio)
    if not input_path.exists():
        print(f"Error: File {input_audio} does not exist!")
        return
    
    cmd = [
        sys.executable, "inference.py",
        "--input-audio", str(input_path),
        "--output-dir", output_dir,
    ]
    
    if checkpoint:
        checkpoint_path = Path(checkpoint)
        if checkpoint_path.exists():
            cmd.extend(["--checkpoint", str(checkpoint_path)])
        else:
            print(f"Warning: Checkpoint {checkpoint} not found. Using untrained model.")
    
    print(f"Running: {' '.join(cmd)}\n")
    subprocess.run(cmd)


def setup_directories():
    """Create necessary directories if they don't exist."""
    dirs = ["checkpoints", "outputs", "train_audio", "val_audio"]
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
    print("Created directories: checkpoints, outputs, train_audio, val_audio")


def main():
    parser = argparse.ArgumentParser(
        description='Quick start script for BeatSwap',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Setup directories
  python quick_start.py --setup

  # Train with audio files in a directory
  python quick_start.py --mode train --audio-dir ./audio_files

  # Train with custom parameters
  python quick_start.py --mode train --audio-dir ./audio_files --epochs 100 --batch-size 4

  # Separate audio (with checkpoint)
  python quick_start.py --mode infer --input song.wav --checkpoint checkpoints/best_model.pth

  # Separate audio (without checkpoint - uses untrained model)
  python quick_start.py --mode infer --input song.wav
        """
    )
    
    parser.add_argument('--mode', type=str, choices=['train', 'infer', 'setup'], required=True,
                       help='Mode: train, infer, or setup directories')
    parser.add_argument('--audio-dir', type=str, 
                       help='Directory containing audio files for training')
    parser.add_argument('--input', type=str, 
                       help='Input audio file for inference')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Model checkpoint path for inference')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory for inference results')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for training (default: 1)')
    
    args = parser.parse_args()
    
    if args.mode == 'setup':
        setup_directories()
        print("\nNext steps:")
        print("1. Place your training audio files in 'train_audio/' directory")
        print("2. Optionally place validation files in 'val_audio/' directory")
        print("3. Run: python quick_start.py --mode train --audio-dir ./train_audio")
    elif args.mode == 'train':
        if not args.audio_dir:
            print("Error: --audio-dir is required for training mode")
            parser.print_help()
            return
        train_quick(args.audio_dir, args.epochs, args.batch_size)
    elif args.mode == 'infer':
        if not args.input:
            print("Error: --input is required for inference mode")
            parser.print_help()
            return
        infer_quick(args.input, args.checkpoint, args.output_dir)


if __name__ == '__main__':
    main()

