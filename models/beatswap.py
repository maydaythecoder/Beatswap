import torch.nn as nn
import torch.nn.functional as F
from .encoder import AudioEncoder
from .decoder import AudioDecoder


class BeatSwapModel(nn.Module):
    """
    BeatSwap model that separates audio into voice and beat components.
    
    Takes 1 audio file as input and returns 2 separate audio files (voice and beat).
    """
    
    def __init__(self, input_channels=1, hidden_dim=256, leaky_relu_slope=0.2):
        super(BeatSwapModel, self).__init__()
        
        self.encoder = AudioEncoder(input_channels, hidden_dim, leaky_relu_slope)
        self.voice_decoder = AudioDecoder(hidden_dim, input_channels, leaky_relu_slope, use_sigmoid=True)
        self.beat_decoder = AudioDecoder(hidden_dim, input_channels, leaky_relu_slope, use_sigmoid=True)
    
    def forward(self, x):
        """
        Args:
            x: Input audio tensor of shape (batch, channels, samples)
        
        Returns:
            voice, beat: Voice and beat audio tensors, each of same shape as input
                        Outputs are in [0, 1] range (Sigmoid activation) for BCELoss
        """
        encoded = self.encoder(x)
        
        voice = self.voice_decoder(encoded)
        beat = self.beat_decoder(encoded)
        
        if voice.shape[-1] != x.shape[-1]:
            voice = F.interpolate(voice, size=x.shape[-1], mode='linear', align_corners=False)
        if beat.shape[-1] != x.shape[-1]:
            beat = F.interpolate(beat, size=x.shape[-1], mode='linear', align_corners=False)
        
        return voice, beat

