import torch.nn as nn


class AudioDecoder(nn.Module):
    """Decoder module that reconstructs audio from latent representation."""
    
    def __init__(self, hidden_dim=256, output_channels=1, leaky_relu_slope=0.2, use_sigmoid=True):
        super(AudioDecoder, self).__init__()
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, 128, kernel_size=5, padding=2, stride=2, output_padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=leaky_relu_slope),
            nn.ConvTranspose1d(128, 64, kernel_size=5, padding=2, stride=2, output_padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=leaky_relu_slope),
            nn.Conv1d(64, output_channels, kernel_size=7, padding=3),
        )
        
        if use_sigmoid:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Tanh()
    
    def forward(self, x):
        """
        Args:
            x: Encoded representation of shape (batch, hidden_dim, samples')
        
        Returns:
            decoded: Decoded audio tensor of shape (batch, output_channels, samples'')
        """
        return self.activation(self.decoder(x))

