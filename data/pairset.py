import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
from datetime import datetime
import time
import tqdm

import multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

TARGET_CHANNELS = ['z50', 'z100', 'z150', 'z200', 'z250', 'z300', 'z400', 'z500',
               'z600', 'z700', 'z850', 'z925', 'z1000', 't50', 't100', 't150',
               't200', 't250', 't300', 't400', 't500', 't600', 't700', 't850',
               't925', 't1000', 'u50', 'u100', 'u150', 'u200', 'u250', 'u300',
               'u400', 'u500', 'u600', 'u700', 'u850', 'u925', 'u1000', 'v50',
               'v100', 'v150', 'v200', 'v250', 'v300', 'v400', 'v500', 'v600',
               'v700', 'v850', 'v925', 'v1000', 'r50', 'r100', 'r150', 'r200',
               'r250', 'r300', 'r400', 'r500', 'r600', 'r700', 'r850', 'r925',
               'r1000', 't2m', 'u10m', 'v10m', 'msl', 'tp']

START_TIME = "2021-01-01 00:00:00"
END_TIME = "2024-12-31 18:00:00"

ERA5_PATH = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/huangqiusheng/datasets/era5.rtm.02_25.6h.c109.new3/"
GFS_PATH = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/database/gfs_2020_2024_c70_normalized"

BAD_TIMES = {
    pd.Timestamp("202401010000"),
    pd.Timestamp("202501010000"),
}

# Source domain registry: each source gets a unique integer index.
# The model's source embedding table uses these indices to look up a learned
# vector that encodes the domain identity (GFS bias, HRES resolution, etc.).
# Extend this dict when adding new source domains.
SOURCE_REGISTRY = {
    "gfs": 0,
    "hres": 1,
}

class Any2ERA5Dataset(Dataset):
    """Multi-source to ERA5 dataset.

    Each dataset instance corresponds to exactly one source domain (e.g. GFS,
    HRES, CMA). Multiple instances can be combined via torch.utils.data.ConcatDataset
    to train a single model on all sources simultaneously. The source_idx returned
    by __getitem__ tells the model which domain the sample belongs to.
    """

    def __init__(
        self,
        target_channels=None,
        start: str | None = None,
        end: str | None = None,
        x_path: str | None = None,
        y_path: str | None = None,
        target_mode: str = "era5",
        source_name: str = "gfs",
        source_idx: int | None = None,
        # Validation: select N whole days per month in a given year
        val_sample_per_month: int | None = None,
        val_sample_year: int | None = None,
        # Training: cap samples per year for fast iteration
        max_samples_per_year: int | None = None,
        sample_seed: int = 42,
    ):
        self.x_path = GFS_PATH if x_path is None else x_path
        self.y_path = ERA5_PATH if y_path is None else y_path
        self.target_channels = TARGET_CHANNELS if target_channels is None else target_channels
        self.target_mode = str(target_mode).lower()
        if self.target_mode not in {"era5", "diff"}:
            raise ValueError(f"target_mode must be 'era5' or 'diff', got: {target_mode}")

        self.source_name = str(source_name).lower()
        self.source_idx = SOURCE_REGISTRY.get(self.source_name, 0) if source_idx is None else int(source_idx)

        self.start_time = pd.to_datetime(START_TIME if start is None else start)
        self.end_time = pd.to_datetime(END_TIME if end is None else end)

        self.val_sample_per_month = val_sample_per_month
        self.val_sample_year = val_sample_year
        self.max_samples_per_year = max_samples_per_year
        self.sample_seed = int(sample_seed)

        self.ds_x = xr.open_zarr(self.x_path, consolidated=False)
        self.ds_y = xr.open_zarr(self.y_path, consolidated=False)

        x_times = pd.DatetimeIndex(self.ds_x.time.values)
        y_times = pd.DatetimeIndex(self.ds_y.time.values)

        x_times_in_range = x_times[(x_times >= self.start_time) & (x_times <= self.end_time)]
        y_times_in_range = y_times[(y_times >= self.start_time) & (y_times <= self.end_time)]

        common_times = x_times_in_range.intersection(y_times_in_range)

        if BAD_TIMES:
            mask = ~common_times.isin(BAD_TIMES)
            common_times = common_times[mask]

        if self.val_sample_per_month is not None and self.val_sample_year is not None:
            rng = np.random.default_rng(self.sample_seed)
            times_year = common_times[common_times.year == self.val_sample_year]

            selected_ts: list[pd.Timestamp] = []
            for month in range(1, 13):
                month_times = times_year[times_year.month == month]
                if len(month_times) == 0:
                    continue

                days = month_times.normalize().unique()
                if len(days) == 0:
                    continue

                k = min(self.val_sample_per_month, len(days))
                chosen_days = rng.choice(days, size=k, replace=False)

                for d in chosen_days:
                    mask_d = month_times.normalize() == d
                    selected_ts.extend(month_times[mask_d].tolist())

            if selected_ts:
                common_times = pd.DatetimeIndex(sorted(selected_ts))

        if self.max_samples_per_year is not None and self.max_samples_per_year > 0:
            rng = np.random.default_rng(self.sample_seed)
            selected_ts: list[pd.Timestamp] = []

            for year in sorted(common_times.year.unique()):
                year_times = common_times[common_times.year == year]
                n = len(year_times)
                if n <= self.max_samples_per_year:
                    selected_ts.extend(year_times.tolist())
                else:
                    idx = rng.choice(n, size=self.max_samples_per_year, replace=False)
                    selected_ts.extend(year_times.sort_values().to_series().iloc[idx].tolist())

            if selected_ts:
                common_times = pd.DatetimeIndex(sorted(selected_ts))

        self.time_list = common_times.tolist()

        self.align_ch()

        self.lat_size = len(self.ds_x["lat"])
        self.lon_size = len(self.ds_x["lon"])
        self.chan_size = len(self.target_channels)


    def align_ch(self):
        self.x_all_channels = [str(c).strip() for c in self.ds_x["channel"].values]
        self.y_all_channels = [str(c).strip() for c in self.ds_y["channel"].values]

        self.x_c_idx = {name: idx for idx, name in enumerate(self.x_all_channels)}
        self.y_c_idx = {name: idx for idx, name in enumerate(self.y_all_channels)}

        self.x_target_idx = []
        self.y_target_idx = []
        for ch in self.target_channels:
            self.x_target_idx.append(self.x_c_idx[ch])
            self.y_target_idx.append(self.y_c_idx[ch])


    def __len__(self):
        return len(self.time_list)

    def __getitem__(self, idx):
        current_time = self.time_list[idx]

        x_data = self.ds_x["data"].sel(time=current_time).isel(channel=self.x_target_idx)
        y_data = self.ds_y["data"].sel(time=current_time).isel(channel=self.y_target_idx)

        x_np = x_data.values.astype(np.float32)
        y_np = y_data.values.astype(np.float32)

        if self.target_mode == "diff":
            y_np = y_np - x_np

        x_tensor = torch.from_numpy(x_np)
        y_tensor = torch.from_numpy(y_np)

        return x_tensor, y_tensor, self.source_idx, str(current_time)
