from torch import nn

from models.utils import quantize_ste


class EncoderWithQuantization(nn.Module):
    def __init__(self, in_channels=28, latent_channels=14, num_levels=15):
        super().__init__()
        self.num_levels = num_levels

        self.encoder_net = nn.Sequential(
            nn.Conv2d(in_channels, 20, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Conv2d(20, latent_channels, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(latent_channels),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder_net(x)

        # Масштабируем интервал [-1, 1] в диапазон целых чисел
        x_scaled = (x + 1) * (self.num_levels / 2)

        # Квантование с помощью STE
        x_quantized = quantize_ste(x_scaled)

        # (Опционально) Возвращаем к исходному масштабу для декодера
        x_normalized = (x_quantized / (self.num_levels / 2)) - 1
        return x_normalized
