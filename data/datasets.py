import xarray as xr
import xarray_regrid
import numpy as np


def get_dataset025():
    "Датасет с сеткой 0.25 градусов"
    rename_dict_ground = {
        "2m_temperature": "2t",
        "mean_sea_level_pressure": "msl",
        "10m_u_component_of_wind": "10u",
        "10m_v_component_of_wind": "10v",
        "total_precipitation_6hr": "tp",
        "sea_surface_temperature": "sst",
        "total_column_water_vapour": "tcwv",
        "total_cloud_cover": "tcc",
    }
    rename_dict_air = {
        "temperature": "t",
        "u_component_of_wind": "u",
        "v_component_of_wind": "v",
        "geopotential": "z",
        "specific_humidity": "q",
    }

    url = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"

    ds = xr.open_dataset(
        url,
        engine="zarr",
        consolidated=True,
        storage_options={"token": "anon"},
        chunks="auto",
    )

    # наземные показатели
    near_ground = [
        "2m_temperature",
        "mean_sea_level_pressure",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "total_precipitation_6hr",
        "sea_surface_temperature",
        "total_column_water_vapour",
        "total_cloud_cover",
    ]

    isobar_types = [1000, 925, 850, 700]  # изобарные значения

    air_types = [  # воздушные показатели
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "geopotential",
        "specific_humidity",
    ]

    # Общий временной срез для обоих типов данных
    START_TIME = "2014-01-01"
    END_TIME = "2019-12-31"
    time_slice = slice(START_TIME, END_TIME)

    # выделяем приземные переменные
    ds_ground = ds[near_ground].sel(time=time_slice).rename(rename_dict_ground)

    # выделяем атмосферные переменные
    ds_air = ds[air_types].sel(level=isobar_types, time=time_slice)

    air_vars = {}
    for air in air_types:
        for bar in isobar_types:
            da = ds_air[air].sel(level=bar).drop_vars("level")  # срез по уровням
            da.name = f"{rename_dict_air[air]}_{bar}"
            air_vars[da.name] = da

    # объединение
    ds_ground_dict = {name: ds_ground[name] for name in ds_ground.data_vars}
    all_vars = {**ds_ground_dict, **air_vars}
    ds = xr.merge(list(all_vars.values()))

    return ds


def get_dataset05():
    "Датасет с сеткой 0.5 градусов"
    ds = get_dataset025().to_dataset(dim="variable")

    target_grid = xr.Dataset(
        {
            "latitude": (["latitude"], np.arange(-89.75, 90.0, 0.5)),
            "longitude": (["longitude"], np.arange(0.0, 360.0, 0.5)),
        }
    )

    ds = ds.regrid.conservative(target_grid)
    return ds
