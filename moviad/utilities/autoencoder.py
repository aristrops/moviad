import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureAutoencoder(nn.Module):
    def __init__(self, in_channels, compression_ratio=0.5):
        super().__init__()

        latent_channels = max(1, int(in_channels * compression_ratio))

        self.norm = nn.BatchNorm2d(in_channels)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, latent_channels, 3, stride=2, padding=1),
            nn.GELU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 32, 3, stride=2,
                               padding=1, output_padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, in_channels, 3, stride=2,
                               padding=1, output_padding=1)
        )

    def compress(self, x):
        x = self.norm(x)
        return self.encoder(x)

    def decompress(self, z):
        return self.decoder(z)

    def forward(self, x):
        recon = self.decompress(self.compress(x))
        if recon.shape[-2:] != x.shape[-2:]: # Crop to match patch dimensions
            recon = recon[..., :x.shape[-2], :x.shape[-1]]
        return recon


class ViTFeatureAutoencoder(nn.Module):
    def __init__(self, in_channels, compression_ratio=0.5, hidden_channels=None):
        super().__init__()

        latent_channels = max(1, int(in_channels * compression_ratio))
        if hidden_channels is None:
            hidden_channels = max(latent_channels, in_channels // 2)

        self.norm = BatchNorm1d(in_channels)

        self.encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, latent_channels),
            nn.GELU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, in_channels)
        )

    def compress(self, x):
        # x: [B, N, C]
        x = self.norm(x)
        return self.encoder(x)

    def decompress(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decompress(self.compress(x))


class BatchNorm1d(nn.BatchNorm1d):
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = super(BatchNorm1d, self).forward(x)
        x = x.permute(0, 2, 1)
        return x