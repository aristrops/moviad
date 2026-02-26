import torch
import torch.nn as nn
import torch.nn.functional as F


# class FeatureAutoencoder(nn.Module):
#     def __init__(self, in_channels, compression_ratio=0.5):
#         super().__init__()
#
#         latent_channels = max(1, int(in_channels * compression_ratio))
#
#         self.norm = nn.BatchNorm2d(in_channels)
#
#         # Encoder
#         self.encoder = nn.Sequential(
#             nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
#             nn.GELU(),
#             nn.Conv2d(32, latent_channels, 3, stride=2, padding=1),
#             nn.GELU()
#         )
#
#         # Decoder
#         self.decoder = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
#             nn.ConvTranspose2d(latent_channels, 32, 3, stride=2,
#                                padding=1, output_padding=1),
#             nn.GELU(),
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
#             nn.ConvTranspose2d(32, in_channels, 3, stride=2,
#                                padding=1, output_padding=1)
#         )
#
#     def compress(self, x):
#         x = self.norm(x)
#         return self.encoder(x)
#
#     def decompress(self, z):
#         return self.decoder(z)
#
#     def forward(self, x):
#         return F.interpolate(self.decompress(self.compress(x)), size=x.shape[-2:])
    
    
# --- First version --- (uses tensor crop instead of upscaling+interpolation) 
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