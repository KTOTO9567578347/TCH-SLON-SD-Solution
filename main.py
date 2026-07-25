import torch
from torch import optim, nn

device = "cuda" if torch.cuda.is_available() else "cpu"
print("работаем на", device)

import dask

dask.config.set(
    scheduler="threads", num_workers=4
)  # Настраиваем dask на параллельную фоновую загрузку через потоки (threads)


from data import *
from models import *
from train import train_model, save_model

ds = get_dataset025()

bgen = get_batch_generator(ds, device)
loader = get_loader(bgen, *get_precomputed_stats(), device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = EncoderWithQuantization(latent_channels=14).to(device)
decoder = Decoder(latent_channels=14).to(device)

optimizer = optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()),
    lr=1e-3
)

criterion = nn.SmoothL1Loss(beta=1.0)

save_model(encoder, decoder)