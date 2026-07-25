import torch
import torch.nn as nn


class LatentPredictor(nn.Module):
    """
    Легковесный предиктор для прогноза латентного состояния через 6 часов.
    """

    def __init__(self, latent_channels=14, hidden_channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(
            latent_channels, hidden_channels, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.conv3 = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm2d(hidden_channels)
        self.conv4 = nn.Conv2d(
            hidden_channels, latent_channels, kernel_size=3, padding=1
        )

        self.activation = nn.ReLU()

    def forward(self, z_t):
        z = self.activation(self.bn1(self.conv1(z_t)))
        z = self.activation(self.bn2(self.conv2(z)))
        z = self.activation(self.bn3(self.conv3(z)))
        z = self.conv4(z)
        return z
