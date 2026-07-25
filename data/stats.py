import numpy as np


def compute_dataset_stats(ds, sample_fraction=0.02):
    """
    Быстро считает среднее и std для каждой переменной по случайной временной подвыборке.
    Учитывает веса широт, чтобы избежать искажений на полюсах.
    """

    # Случайная подвыборка по time
    total_times = len(ds.time)
    n_samples = max(1, int(total_times * sample_fraction))
    random_indices = np.random.choice(total_times, size=n_samples, replace=False)
    random_indices.sort()

    ds_sampled = ds.isel(time=random_indices)

    weights = np.cos(np.deg2rad(ds_sampled.latitude))
    weights /= weights.mean()

    means = {}
    stds = {}

    for var_name in ds_sampled.data_vars:
        weighted_var = ds_sampled[var_name].weighted(weights)

        v_mean = float(weighted_var.mean().values)

        v_variance = float(
            ((ds_sampled[var_name] - v_mean) ** 2).weighted(weights).mean().values
        )
        v_std = np.sqrt(v_variance)

        if v_std == 0:
            v_std = 1.0

        means[var_name] = v_mean
        stds[var_name] = v_std
        print(f"Переменная {var_name:8} | Среднее: {v_mean:12.4f} | Std: {v_std:12.4f}")

    return means, stds


def get_precomputed_stats():
    means_dict = {
        "2t": 287.54571533203125,
        "msl": 101144.828125,
        "10u": -0.37171676754951477,
        "10v": 0.14705954492092133,
        "tp": 0.0007431305712088943,
        "sst": 291.5942687988281,
        "tcwv": 24.714500427246094,
        "tcc": 0.6294848918914795,
        "t_1000": 288.50189208984375,
        "t_925": 284.2601318359375,
        "t_850": 281.25579833984375,
        "t_700": 273.9064636230469,
        "u_1000": -0.3982798755168915,
        "u_925": 0.2242080569267273,
        "u_850": 1.0792275667190552,
        "u_700": 3.2283222675323486,
        "v_1000": 0.1467936784029007,
        "v_925": 0.1286812722682953,
        "v_850": 0.03933039307594299,
        "v_700": -0.02824198640882969,
        "z_1000": 942.7501831054688,
        "z_925": 7371.5751953125,
        "z_850": 14263.0771484375,
        "z_700": 29794.1171875,
        "q_1000": 0.009408126585185528,
        "q_925": 0.008069725707173347,
        "q_850": 0.00611522002145648,
        "q_700": 0.0032571181654930115,
    }
    stds_dict = {
        "2t": np.float64(15.416738534020974),
        "msl": np.float64(1155.7307212322428),
        "10u": np.float64(5.568044406702828),
        "10v": np.float64(4.567121908139054),
        "tp": np.float64(0.002447679867059621),
        "sst": np.float64(10.330162505268438),
        "tcwv": np.float64(17.24232112969287),
        "tcc": np.float64(0.36417275425138707),
        "t_1000": np.float64(13.3624105652117),
        "t_925": np.float64(12.732162863842351),
        "t_850": np.float64(12.366676074193236),
        "t_700": np.float64(11.621001181354009),
        "u_1000": np.float64(6.230015657506282),
        "u_925": np.float64(8.052301151410559),
        "u_850": np.float64(8.340163470891408),
        "u_700": np.float64(9.354396712455802),
        "v_1000": np.float64(5.151321737803057),
        "v_925": np.float64(6.186798980766869),
        "v_850": np.float64(5.871119739629767),
        "v_700": np.float64(6.425596026965733),
        "z_1000": np.float64(936.1515302022424),
        "z_925": np.float64(1059.7839992187087),
        "z_850": np.float64(1253.2425443624231),
        "z_700": np.float64(1801.0211686707073),
        "q_1000": np.float64(0.005841925212812825),
        "q_925": np.float64(0.005067701844307394),
        "q_850": np.float64(0.004285320445588393),
        "q_700": np.float64(0.0028702109939330775),
    }
    return means_dict, stds_dict


def calc_stats(ds):
    return compute_dataset_stats(ds, sample_fraction=0.005)
