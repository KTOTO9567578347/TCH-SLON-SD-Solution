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
from autoencoder import (
    train_autoencoder,
    save_autoencoder,
    load_autoencoder,
)
from latentmodel import train_latent_predictor, save_latent, load_latent

ds = get_dataset025()

bgen = get_batch_generator(ds, device)
loader = get_loader(bgen, *get_precomputed_stats(), device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = EncoderWithQuantization().to(device)
decoder = Decoder().to(device)

optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)

criterion = nn.SmoothL1Loss(beta=1.0)

encoder, decoder = load_autoencoder("weights", device)

save_autoencoder(encoder, decoder)

latent = LatentPredictor()
latent = load_latent("weights", device)
