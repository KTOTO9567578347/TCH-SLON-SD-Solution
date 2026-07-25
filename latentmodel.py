from models import freeze_model, unfreeze_model, LatentPredictor


import torch
from torch import nn, optim


def train_latent_predictor(
    encoder,
    decoder,
    predictor,
    train_pairs,
    val_pairs,
    num_steps=5000,
    batch_size=32,
    lr=1e-3,
    device="cuda",
):
    """
    Обучение probe-модели на 1024 парах, не более 5000 шагов
    """

    # Заморозка Энкодера и Декодера для обучения латентной модели
    freeze_model(encoder)
    freeze_model(decoder)

    predictor.train()
    optimizer = optim.Adam(predictor.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Подготовка данных
    train_loader = DataLoader(
        TensorDataset(train_pairs[:, 0], train_pairs[:, 1]),
        batch_size=batch_size,
        shuffle=True,
    )

    best_val_loss = float("inf")
    step = 0

    while step < num_steps:
        for z_t, z_t_plus_6 in train_loader:
            z_t = z_t.to(device)
            z_t_plus_6 = z_t_plus_6.to(device)

            # Прогноз
            z_pred = predictor(z_t)

            # Loss в латентном пространстве
            loss = criterion(z_pred, z_t_plus_6)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            optimizer.step()

            step += 1

            # Валидация
            if step % 100 == 0:
                predictor.eval()
                with torch.no_grad():
                    val_loss = validate_predictor(
                        predictor, val_pairs, criterion, device
                    )
                predictor.train()

                print(
                    f"Step {step}: Train Loss = {loss.item():.6f}, Val Loss = {val_loss:.6f}"
                )

                # Сохраняем лучшую модель
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(predictor.state_dict(), "checkpoints/probe_best.pt")

            if step >= num_steps:
                break

    return predictor


def save_latent(latent):
    dirname = f"weights_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}"
    print(f"Сохранение модели в {dirname}")
    os.mkdir(f"checkpoints/{dirname}")

    torch.save(latent.state_dict(), f"./checkpoints/{dirname}/latent.pt")


def load_latent(dirname, device):
    print(f"Загрузка модели из {dirname}")
    latent = LatentPredictor().to(device)
    latent.load_state_dict(
        torch.load(f"./checkpoints/{dirname}/latent.pt", weights_only=True)
    )
    latent.eval()
    return latent
