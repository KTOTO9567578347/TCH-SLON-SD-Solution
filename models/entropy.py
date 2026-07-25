import torch
import struct
import torchac
from typing import Tuple


def encode_latents_with_scale(latents_int: torch.Tensor, scale: float) -> bytes:
    min_val, max_val = -128, 127

    latents_cpu = latents_int.detach().cpu()

    symbols = (latents_cpu.flatten().to(torch.int32) - min_val).clamp(
        0, max_val - min_val
    )
    symbols_batch = symbols.unsqueeze(0).to(torch.int16)
    num_symbols = symbols_batch.shape[1]

    alphabet = torch.arange(min_val, max_val + 1, dtype=torch.float32)
    probs = 0.5 * torch.exp(-torch.abs(alphabet) / scale) + 1e-10
    probs = probs / probs.sum()
    cdf_single = torch.cat([torch.tensor([0.0]), torch.cumsum(probs, dim=0)]).unsqueeze(
        0
    )

    cdf_batch = cdf_single.unsqueeze(1).expand(1, num_symbols, -1)

    encoded = torchac.encode_float_cdf(
        cdf_batch, symbols_batch, check_input_bounds=True
    )
    return encoded


def decode_latents_with_scale(
    encoded_bytes: bytes, shape: Tuple[int, int, int], scale: float
) -> torch.Tensor:
    """Декодирование с исправленными размерностями CDF"""
    min_val, max_val = -128, 127
    num_symbols = shape[0] * shape[1] * shape[2]

    alphabet = torch.arange(min_val, max_val + 1, dtype=torch.float32)
    probs = 0.5 * torch.exp(-torch.abs(alphabet) / scale) + 1e-10
    probs = probs / probs.sum()
    cdf_single = torch.cat([torch.tensor([0.0]), torch.cumsum(probs, dim=0)]).unsqueeze(
        0
    )

    cdf_batch = cdf_single.unsqueeze(1).expand(1, num_symbols, -1)

    decoded_symbols = torchac.decode_float_cdf(cdf_batch, encoded_bytes)
    latents_flat = decoded_symbols + min_val
    return latents_flat.reshape(shape).to(torch.int32)


def compute_B(
    encoder: EncoderWithQuantization,
    decoder: Decoder,
    image: torch.Tensor,
    scale: float = 1.0,
) -> int:
    """
    Принимает модель и изображение (1, 28, H, W).
    Возвращает размер B в байтах.
    """
    with torch.no_grad():
        latents_float = encoder(image)
        latents_int = latents_float.to(torch.int32)

        C, H, W = latents_int.shape[1], latents_int.shape[2], latents_int.shape[3]

        header = bytearray()
        header.append(1)
        header.extend(struct.pack(">HHH", C, H, W))
        header.extend(struct.pack(">f", scale))

        encoded_latents = encode_latents_with_scale(latents_int, scale)

        bitstream = bytes(header) + encoded_latents
        return len(bitstream)
