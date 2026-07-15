import os
import warnings
from collections import defaultdict
from typing import Dict, Iterable, Literal, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xarray as xr

try:
    from zarr.errors import ZarrUserWarning
except Exception:
    ZarrUserWarning = UserWarning

from fuxi.fuxi_grad import time_encoding

TARGET_RMSE_CHANNELS = ["z500", "z850", "u200", "t2m", "u10m", "v10m", "msl", "tp", "t300"]


DEFAULT_CHANNEL_WEIGHTS = {
    "z500": 0.05,
    "z850": 0.05,
    "u200": 0.1,
    "t2m": 0.1,
    "u10m": 0.1,
    "v10m": 0.1,
    "msl": 0.05,
    "tp": 0.1,
    "t300": 0.1,
}


FUXI_ERA5_REFERENCE_RMSE = {
    "z500": 21.8412,
    "z850": 22.2562,
    "u200": 1.0605,
    "t2m": 0.6205,
    "u10m": 0.4522,
    "v10m": 0.4659,
    "msl": 31.0368,
    "tp": 0.6198,
    "t300": 0.2347,
}


ChannelRMSELossMode = Literal["manual_weighted", "raw_mean", "reference_norm"]


def _open_dataarray_robust(path: str) -> xr.DataArray:
    try:
        return xr.open_dataarray(path)
    except Exception:
        ds = xr.open_dataset(path)
        if len(ds.data_vars) == 0:
            raise ValueError(f"No data_vars found in {path}")
        first = list(ds.data_vars)[0]
        return ds[first]


def _configure_warning_filters():
    warnings.filterwarnings(
        "ignore",
        message=r"Both zarr\.json \(Zarr format 3\) and \.zgroup \(Zarr format 2\) metadata objects exist.*",
        category=ZarrUserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Object at .* is not recognized as a component of a Zarr hierarchy\.",
        category=ZarrUserWarning,
    )


def _to_channel_first_stats(arr: np.ndarray, expected_c: int) -> np.ndarray:
    # 输出 [C,H,W]
    if arr.ndim == 1:
        return arr[:, None, None]
    if arr.ndim == 3:
        if arr.shape[0] == expected_c:
            return arr
        if arr.shape[-1] == expected_c:
            return np.transpose(arr, (2, 0, 1))
    raise ValueError(f"Unsupported stats shape {arr.shape}, expected channel dim={expected_c}")


def _decode_channel_values(values) -> list[str]:
    out = []
    for v in values:
        if isinstance(v, bytes):
            out.append(v.decode())
        else:
            out.append(str(v))
    return out


class FuXiRMSEInterface:
    """FuXi 一步推理 + 通道 RMSE 计算接口（不落盘 nc）。

    - 输入：G2E 输出的 ERA5（物理量，单时刻）
    - 内部：构造 [t-6h, t] 双帧输入，调用 FuXi 一步预报
    - 输出：目标通道 RMSE、加权 RMSE（raw）以及训练用的归一化 RMSE loss
    """

    def __init__(
        self,
        fuxi_model: torch.nn.Module,
        era5_zarr_path: str,
        channel_names: Iterable[str],
        device: torch.device,
        target_channels: Optional[Iterable[str]] = None,
        channel_weights: Optional[Dict[str, float]] = None,
        reference_rmse: Optional[Dict[str, float]] = None,
        loss_mode: ChannelRMSELossMode = "manual_weighted",
        lead_hours: int = 6,
    ):
        _configure_warning_filters()

        self.fuxi_model = fuxi_model
        self.era5_zarr_path = era5_zarr_path
        self.device = device
        self.lead_hours = int(lead_hours)

        self.channel_names = [str(c) for c in channel_names]
        self.channel_to_idx = {c: i for i, c in enumerate(self.channel_names)}

        self.target_channels = list(target_channels) if target_channels is not None else list(TARGET_RMSE_CHANNELS)
        self.channel_weights = dict(channel_weights) if channel_weights is not None else dict(DEFAULT_CHANNEL_WEIGHTS)
        self.reference_rmse = dict(reference_rmse) if reference_rmse is not None else dict(FUXI_ERA5_REFERENCE_RMSE)
        self.loss_mode = str(loss_mode)
        if self.loss_mode not in {"manual_weighted", "raw_mean", "reference_norm"}:
            raise ValueError(
                "loss_mode must be one of: manual_weighted, raw_mean, reference_norm; "
                f"got {self.loss_mode!r}"
            )

        # 过滤不存在通道
        self.target_channels = [c for c in self.target_channels if c in self.channel_to_idx]
        if len(self.target_channels) == 0:
            raise ValueError("None of target_channels exist in channel_names")

        for ch in self.target_channels:
            self.channel_weights.setdefault(ch, 1.0 / len(self.target_channels))

        wsum = sum(float(self.channel_weights[ch]) for ch in self.target_channels)
        if wsum <= 0:
            raise ValueError("Sum of channel weights must be positive")

        if self.loss_mode == "reference_norm":
            missing_ref = [ch for ch in self.target_channels if ch not in self.reference_rmse]
            if missing_ref:
                raise ValueError(f"reference_rmse missing target channels: {missing_ref}")
            non_positive_ref = [ch for ch in self.target_channels if float(self.reference_rmse[ch]) <= 0]
            if non_positive_ref:
                raise ValueError(f"reference_rmse must be positive for channels: {non_positive_ref}")

        self.ds_era5 = xr.open_zarr(self.era5_zarr_path, consolidated=False)
        self.era5_times = pd.DatetimeIndex(self.ds_era5["time"].values)
        self.era5_time_to_index = {pd.Timestamp(t): i for i, t in enumerate(self.era5_times)}

        if "channel" in self.ds_era5.coords:
            era5_channel_values = _decode_channel_values(self.ds_era5["channel"].values)
            self._era5_channel_dim = "channel"
        elif "channel" in self.ds_era5.dims:
            era5_channel_values = _decode_channel_values(self.ds_era5["channel"].values)
            self._era5_channel_dim = "channel"
        elif "level" in self.ds_era5.coords:
            era5_channel_values = _decode_channel_values(self.ds_era5["level"].values)
            self._era5_channel_dim = "level"
        else:
            raise ValueError("ERA5 zarr missing channel/level dimension")

        era5_cidx = {c: i for i, c in enumerate(era5_channel_values)}
        missing = [c for c in self.channel_names if c not in era5_cidx]
        if missing:
            raise ValueError(f"ERA5 zarr missing target channels (first 10): {missing[:10]}")
        self.era5_channel_indices = [era5_cidx[c] for c in self.channel_names]

        mean_da = _open_dataarray_robust(os.path.join(self.era5_zarr_path, "mean.nc"))
        std_da = _open_dataarray_robust(os.path.join(self.era5_zarr_path, "std.nc"))

        if "channel" in mean_da.dims:
            mean_da = mean_da.sel(channel=self.channel_names)
            std_da = std_da.sel(channel=self.channel_names)

        mean_np = mean_da.values.astype(np.float32)
        std_np = std_da.values.astype(np.float32)

        mean_np = _to_channel_first_stats(mean_np, expected_c=len(self.channel_names))
        std_np = _to_channel_first_stats(std_np, expected_c=len(self.channel_names))

        self.era5_mean = torch.from_numpy(mean_np).to(self.device)
        self.era5_std = torch.from_numpy(std_np).to(self.device)

        lat_vals = self.ds_era5["lat"].values.astype(np.float32)
        lat_weights = np.cos(np.deg2rad(np.abs(lat_vals))).astype(np.float32)
        self.lat_weights = torch.from_numpy(lat_weights).to(self.device).view(-1, 1)

    def close(self):
        try:
            self.ds_era5.close()
        except Exception:
            pass

    def _fetch_era5_denorm(self, ts: pd.Timestamp) -> Optional[torch.Tensor]:
        ts = pd.Timestamp(ts)
        tidx = self.era5_time_to_index.get(ts)
        if tidx is None:
            return None

        arr = (
            self.ds_era5["data"]
            .isel(time=tidx)
            .isel({self._era5_channel_dim: self.era5_channel_indices})
            .values
            .astype(np.float32)
        )  # [C,H,W], C == len(channel_names)
        t = torch.from_numpy(arr).to(self.device)
        t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        t = t * self.era5_std + self.era5_mean
        t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        return t

    def _spatial_rmse(self, pred_2d: torch.Tensor, true_2d: torch.Tensor) -> torch.Tensor:
        pred_2d = torch.nan_to_num(pred_2d, nan=0.0, posinf=0.0, neginf=0.0)
        true_2d = torch.nan_to_num(true_2d, nan=0.0, posinf=0.0, neginf=0.0)

        if pred_2d.shape != true_2d.shape:
            true_2d = F.interpolate(
                true_2d[None, None], size=pred_2d.shape[-2:], mode="bilinear", align_corners=False
            ).squeeze(0).squeeze(0)

        err2 = (pred_2d - true_2d) ** 2
        w = self.lat_weights.expand_as(err2)
        weighted_mean = (err2 * w).sum() / (w.sum() + 1e-12)
        return torch.sqrt(weighted_mean + 1e-12)

    def compute_batch_loss(
        self,
        g2e_current_denorm: torch.Tensor,
        times,
        requires_grad: bool = True,
    ):
        """返回:
        - rmse_loss_norm: 用于反向传播的加权归一化 RMSE
        - channel_rmse_mean: 各通道 raw RMSE 的 batch 均值
        - weighted_rmse_raw: 加权 raw RMSE（日志可读）
        - valid_count: 有效样本数
        """
        batch_size = g2e_current_denorm.shape[0]
        rmse_loss_list = []
        rmse_sum = defaultdict(float)
        valid_count = 0
        era5_cache: dict[pd.Timestamp, Optional[torch.Tensor]] = {}

        def _fetch_cached(ts: pd.Timestamp) -> Optional[torch.Tensor]:
            if ts in era5_cache:
                return era5_cache[ts]
            v = self._fetch_era5_denorm(ts)
            era5_cache[ts] = v
            return v

        for b in range(batch_size):
            t0 = pd.Timestamp(str(times[b]))
            t_prev = t0 - pd.Timedelta(hours=self.lead_hours)
            t_next = t0 + pd.Timedelta(hours=self.lead_hours)

            prev_true = _fetch_cached(t_prev)
            next_true = _fetch_cached(t_next)
            if prev_true is None or next_true is None:
                continue

            curr = torch.nan_to_num(g2e_current_denorm[b], nan=0.0, posinf=0.0, neginf=0.0)
            hist = torch.stack([prev_true, curr], dim=0)  # [2,C,H,W]

            temb = time_encoding(t0, 1).to(device=self.device, dtype=torch.float32)[0]  # [1,12]

            if requires_grad:
                _, pred_next = self.fuxi_model.inference_normal((hist, temb))
            else:
                with torch.no_grad():
                    _, pred_next = self.fuxi_model.inference_normal((hist, temb))

            pred_next = torch.nan_to_num(pred_next.squeeze(0), nan=0.0, posinf=0.0, neginf=0.0)

            rmse_ch_list = []
            for ch in self.target_channels:
                cidx = self.channel_to_idx[ch]
                rmse_ch = self._spatial_rmse(pred_next[cidx], next_true[cidx])
                rmse_sum[ch] += float(rmse_ch.detach().cpu())
                rmse_ch_list.append((ch, rmse_ch))

            if self.loss_mode == "manual_weighted":
                sample_loss = sum(float(self.channel_weights[ch]) * rmse_ch for ch, rmse_ch in rmse_ch_list)
            elif self.loss_mode == "raw_mean":
                sample_loss = torch.stack([rmse_ch for _, rmse_ch in rmse_ch_list]).mean()
            elif self.loss_mode == "reference_norm":
                sample_loss = torch.stack(
                    [rmse_ch / (float(self.reference_rmse[ch]) + 1e-12) for ch, rmse_ch in rmse_ch_list]
                ).mean()
            else:
                raise RuntimeError(f"Unexpected loss_mode: {self.loss_mode!r}")

            rmse_loss_list.append(sample_loss)
            valid_count += 1

        if valid_count == 0:
            zero = torch.tensor(0.0, device=self.device)
            return zero, {ch: 0.0 for ch in self.target_channels}, 0.0, 0

        rmse_loss_norm = torch.stack(rmse_loss_list).mean()
        channel_rmse_mean = {ch: rmse_sum[ch] / valid_count for ch in self.target_channels}
        weighted_rmse_raw = float(sum(self.channel_weights[ch] * channel_rmse_mean[ch] for ch in self.target_channels))

        return rmse_loss_norm, channel_rmse_mean, weighted_rmse_raw, valid_count
