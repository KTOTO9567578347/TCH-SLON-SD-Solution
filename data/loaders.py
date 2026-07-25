import torch
import numpy as np
import xbatcher as xb
import dask

from data.datasets import get_dataset025, get_dataset05


class WeatherAutoencoderLoader:
    def __init__(self, batch_generator, means_dict, stds_dict, device="cuda"):
        self.bgen = batch_generator
        self.var_names = None
        self.device = torch.device(device)
        self.means_dict = means_dict
        self.stds_dict = stds_dict

    def __iter__(self):
        for batch in self.bgen:
            if self.var_names is None:
                self.var_names = list(batch.data_vars)
                self.means_tensor = torch.tensor(
                    [self.means_dict[name] for name in self.var_names],
                    dtype=torch.float32,
                ).view(-1, 1, 1, 1)
                self.stds_tensor = torch.tensor(
                    [self.stds_dict[name] for name in self.var_names],
                    dtype=torch.float32,
                ).view(-1, 1, 1, 1)

                if self.device.type == "cuda":
                    self.means_tensor = self.means_tensor.to(self.device)
                    self.stds_tensor = self.stds_tensor.to(self.device)

            target_dims = ("time", "latitude", "longitude")

            data_arrays = [
                batch[var].transpose(*target_dims).values for var in self.var_names
            ]
            data = np.stack(data_arrays, axis=0)

            # ЗАЩИТА: Пропускаем неполные батчи с краев карты Земли
            if data.shape[2] != 128 or data.shape[3] != 128:
                continue

            tensor_batch = torch.as_tensor(data, dtype=torch.float32)
            if self.device.type == "cuda":
                tensor_batch = tensor_batch.pin_memory().to(
                    self.device, non_blocking=True
                )

            tensor_batch = torch.nan_to_num(
                tensor_batch, nan=0.0, posinf=1.0, neginf=-1.0
            )

            # Z-score нормализация (вход и таргет масштабированы одинаково)
            tensor_batch = (tensor_batch - self.means_tensor) / self.stds_tensor

            # Time, Channels, Lat, Lon
            tensor_batch = tensor_batch.permute(1, 0, 2, 3)

            yield tensor_batch, tensor_batch

    def __len__(self):
        return len(self.bgen)


def get_batch_generator(device):
    dask.config.set(
        scheduler="threads", num_workers=4
    )  # Настраиваем dask на параллельную фоновую загрузку через потоки (threads)

    print("Загрузка датасета 0.25*")
    ds025 = get_dataset025()

    print("Создание батчгенератора")
    return xb.BatchGenerator(
        ds025,
        input_dims={"time": 8, "latitude": 128, "longitude": 128},
        input_overlap={"time": 0, "latitude": 0, "longitude": 0},
        preload_batch=True,
    )

def get_loader(bgen, means_dict, stds_dict, device):
    print("Создание даталоадера")
    return WeatherAutoencoderLoader(bgen, means_dict, stds_dict, device=device)