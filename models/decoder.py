from torch import nn


class Decoder(nn.Module):
    def __init__(self, latent_channels=14, out_channels=28):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(
            latent_channels, 20, kernel_size=5, stride=2, padding=2, output_padding=1
        )
        self.bnorm = nn.BatchNorm2d(20)
        self.deconv2 = nn.ConvTranspose2d(
            20, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        z = self.bnorm(self.deconv1(z))
        z = torch.relu(z)
        z = self.deconv2(z)
        # z = self.sigmoid(z)
        return z
