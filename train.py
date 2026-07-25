import torch
import torch.nn as nn
from tqdm import tqdm

import os, datetime

from models import EncoderWithQuantization, Decoder


def train_model(device, encoder, decoder, optimizer, criterion, n_epochs):
    encoder.train()
    decoder.train()

    print(f"Запуск тестового цикла на устройстве: {device} на {n_epochs} эпох")
    losses = []
    i = 0

    for X_batch, _ in tqdm(loader, total=len(loader), desc="Testing Autoencoder"):
        optimizer.zero_grad()

        latent_features = encoder(X_batch)
        reconstructed_X = decoder(latent_features)

        loss = criterion(reconstructed_X, X_batch)

        losses.append(loss.item())
        loss.backward()

        # клиппинг градиентов
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)

        optimizer.step()

        if i % 10 == 0:
            print(" -", sum(losses) / len(losses))

        i += 1

        if i == n_epochs:
            break

    print(f"Тест успешно завершен! Финальный Loss: {loss.item():.4f}")
    return encoder, decoder


def save_model(encoder, decoder):
    dirname = f"weights_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}"
    print(f"Сохранение модели в {dirname}")
    os.mkdir(f"checkpoints/{dirname}")

    torch.save(encoder.state_dict(), f"./checkpoints/{dirname}/encoder.pt")
    torch.save(decoder.state_dict(), f"./checkpoints/{dirname}/decoder.pt")


def load_model(dirname, device):
    print(f"Загрузка модели из {dirname}")
    encoder = EncoderWithQuantization().to(device)
    encoder.load_state_dict(
        torch.load(f"./checkpoints/{dirname}/encoder.pt", weights_only=True)
    )
    encoder.eval()

    decoder = Decoder().to(device)
    decoder.load_state_dict(
        torch.load(f"./checkpoints/{dirname}/decoder.pt", weights_only=True)
    )
    decoder.eval()

    return encoder, decoder
