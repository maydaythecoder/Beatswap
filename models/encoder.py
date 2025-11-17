import torch.nn as nn


class AudioEncoder(nn.Module):
    """Encoder module that processes input audio into a latent representation."""
    
    def __init__(self, input_channels=1, hidden_dim=256, leaky_relu_slope=0.2):
        super(AudioEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=leaky_relu_slope),
            nn.Conv1d(64, 128, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=leaky_relu_slope),
            nn.Conv1d(128, hidden_dim, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(negative_slope=leaky_relu_slope),
        )
    
    def forward(self, x):
        """
        Args:
            x: Input audio tensor of shape (batch, channels, samples)
        
        Returns:
            encoded: Encoded representation of shape (batch, hidden_dim, samples')
        """
        return self.encoder(x)

