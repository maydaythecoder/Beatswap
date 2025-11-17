"""Training script for BeatSwap model."""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import BeatSwapModel
from data import AudioDataset
from config import MODEL_CONFIG, TRAINING_CONFIG, AUDIO_CONFIG, PATHS
from utils import ensure_dir


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch_idx, audio in enumerate(tqdm(dataloader, desc="Training")):
        audio = audio.to(device)
        
        optimizer.zero_grad()
        
        voice_pred, beat_pred = model(audio)
        
        loss_voice = criterion(voice_pred, audio)
        loss_beat = criterion(beat_pred, audio)
        loss = loss_voice + loss_beat
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for audio in tqdm(dataloader, desc="Validating"):
            audio = audio.to(device)
            
            voice_pred, beat_pred = model(audio)
            
            loss_voice = criterion(voice_pred, audio)
            loss_beat = criterion(beat_pred, audio)
            loss = loss_voice + loss_beat
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    ensure_dir(PATHS['checkpoints_dir'])
    
    model = BeatSwapModel(**MODEL_CONFIG).to(device)
    
    train_dataset = AudioDataset(
        args.train_dir,
        target_sample_rate=AUDIO_CONFIG['sample_rate'],
        target_length=AUDIO_CONFIG.get('num_samples'),
        normalize=AUDIO_CONFIG['normalize']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = None
    if args.val_dir:
        val_dataset = AudioDataset(
            args.val_dir,
            target_sample_rate=AUDIO_CONFIG['sample_rate'],
            target_length=AUDIO_CONFIG.get('num_samples'),
            normalize=AUDIO_CONFIG['normalize']
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}")
        
        if val_loader:
            val_loss = validate(model, val_loader, criterion, device)
            print(f"Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(PATHS['checkpoints_dir'], 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                }, checkpoint_path)
                print(f"Saved best model to {checkpoint_path}")
        
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(PATHS['checkpoints_dir'], f'checkpoint_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, checkpoint_path)
    
    print("\nTraining complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train BeatSwap model')
    parser.add_argument('--train-dir', type=str, required=True, help='Directory containing training audio files')
    parser.add_argument('--val-dir', type=str, default=None, help='Directory containing validation audio files')
    parser.add_argument('--batch-size', type=int, default=TRAINING_CONFIG['batch_size'], help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=TRAINING_CONFIG['learning_rate'], help='Learning rate')
    parser.add_argument('--num-epochs', type=int, default=TRAINING_CONFIG['num_epochs'], help='Number of epochs')
    parser.add_argument('--device', type=str, default=TRAINING_CONFIG['device'], help='Device to use (cuda/cpu)')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--save-every', type=int, default=10, help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    main(args)

