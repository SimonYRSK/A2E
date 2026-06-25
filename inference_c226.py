"""
A2EC226 inference output writer.

This script only does one thing:
  source normalized C226 data -> A2EC226 165-channel output -> expand to the
  full 226-channel C226 layout -> save NetCDF.

The 61 channels not predicted by A2EC226 (cc*, clwc*, ciwc*, tp) are filled
with zeros in the saved 226-channel tensor.
"""

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from tqdm import tqdm

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True


# ============================================================
# Constants
# ============================================================
DEFAULT_DATES = pd.date_range("2025-01-01", "2025-01-31", freq="D").strftime("%Y%m%d").tolist()

# Full 226-channel order from A2E/data/varlist.md.
CHANNELS_226 = [
    "t10", "t20", "t50", "t70", "t100", "t150", "t200", "t250", "t300", "t400",
    "t500", "t600", "t700", "t750", "t800", "t850", "t900", "t925", "t950", "t1000",
    "u10", "u20", "u50", "u70", "u100", "u150", "u200", "u250", "u300", "u400",
    "u500", "u600", "u700", "u750", "u800", "u850", "u900", "u925", "u950", "u1000",
    "v10", "v20", "v50", "v70", "v100", "v150", "v200", "v250", "v300", "v400",
    "v500", "v600", "v700", "v750", "v800", "v850", "v900", "v925", "v950", "v1000",
    "z10", "z20", "z50", "z70", "z100", "z150", "z200", "z250", "z300", "z400",
    "z500", "z600", "z700", "z750", "z800", "z850", "z900", "z925", "z950", "z1000",
    "q10", "q20", "q50", "q70", "q100", "q150", "q200", "q250", "q300", "q400",
    "q500", "q600", "q700", "q750", "q800", "q850", "q900", "q925", "q950", "q1000",
    "r10", "r20", "r50", "r70", "r100", "r150", "r200", "r250", "r300", "r400",
    "r500", "r600", "r700", "r750", "r800", "r850", "r900", "r925", "r950", "r1000",
    "cc10", "cc20", "cc50", "cc70", "cc100", "cc150", "cc200", "cc250", "cc300", "cc400",
    "cc500", "cc600", "cc700", "cc750", "cc800", "cc850", "cc900", "cc925", "cc950", "cc1000",
    "clwc10", "clwc20", "clwc50", "clwc70", "clwc100", "clwc150", "clwc200", "clwc250", "clwc300", "clwc400",
    "clwc500", "clwc600", "clwc700", "clwc750", "clwc800", "clwc850", "clwc900", "clwc925", "clwc950", "clwc1000",
    "ciwc10", "ciwc20", "ciwc50", "ciwc70", "ciwc100", "ciwc150", "ciwc200", "ciwc250", "ciwc300", "ciwc400",
    "ciwc500", "ciwc600", "ciwc700", "ciwc750", "ciwc800", "ciwc850", "ciwc900", "ciwc925", "ciwc950", "ciwc1000",
    "vo10", "vo20", "vo50", "vo70", "vo100", "vo150", "vo200", "vo250", "vo300", "vo400",
    "vo500", "vo600", "vo700", "vo750", "vo800", "vo850", "vo900", "vo925", "vo950", "vo1000",
    "d10", "d20", "d50", "d70", "d100", "d150", "d200", "d250", "d300", "d400",
    "d500", "d600", "d700", "d750", "d800", "d850", "d900", "d925", "d950", "d1000",
    "t2m", "d2m", "u10m", "v10m", "msl", "tp",
]

# A2EC226 in mainc226_res.py was trained on these three source domains.
SOURCE_REGISTRY = {
    "gfs": 0,
    "hres": 1,
    "cma": 2,
}

DEFAULT_GFS_PATH = "/cpfs01/projects-HDD/cfff-aad9fa3a0781_HDD/public/ximutian/dataset/gfs_2020_2025_c226_0p25_norm.zarr"
DEFAULT_HRES_PATH = "/cpfs01/projects-HDD/cfff-aad9fa3a0781_HDD/public/ximutian/dataset/hres_2024_2025_c226_0p25_norm.zarr"
DEFAULT_CMA_PATH = "/cpfs01/projects-HDD/cfff-aad9fa3a0781_HDD/public/ximutian/dataset/cma/cma_gfs_2020_2026.c226.norm.zarr"
DEFAULT_ERA5_PATH = "/cpfs01/projects-HDD/cfff-aad9fa3a0781_HDD/public/dataset/era5.2010_2025.c226.zarr"
DEFAULT_A2E_CKPT = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/A2E/checkpoints/A2E_0603/checkpoint_epoch_33.pth"
DEFAULT_OUTPUT_DIR = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/A2E/inference_results/A2EC226_165_to_226_nc"


# ============================================================
# Model
# ============================================================
def build_a2e_model(device: torch.device, checkpoint_path: str, num_sources: int):
    from models.swinUNET_res import A2E

    model = A2E(
        img_size=(721, 1440),
        patch_size=(4, 4),
        in_chans=165,
        out_chans=165,
        embed_dim=384,
        num_groups=32,
        num_heads=8,
        num_stages=3,
        window_size=9,
        depth=[0, 0, 1],
        using_checkpoints=True,
        using_time_embedding=True,
        using_source_embedding=True,
        num_sources=num_sources,
        res_per_stage=[1, 1, 1],
        channels=[384, 768, 1536],
        using_kl=False,
        dropout_rate=0.1,
        use_skip_connections=True,
        use_residual_blocks=True,
        using_dann=False,
    )

    ckpt = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    clean_state_dict = {}
    for k, v in state_dict.items():
        clean_state_dict[k.replace("_orig_mod.", "").replace("module.", "")] = v

    missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)
    if missing:
        print(f"A2E missing keys ({len(missing)}): {missing[:20]}{'...' if len(missing) > 20 else ''}")
    if unexpected:
        print(f"A2E unexpected keys ({len(unexpected)}): {unexpected[:20]}{'...' if len(unexpected) > 20 else ''}")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)
    return model


# ============================================================
# Data helpers
# ============================================================
def _decode_channel_values(values) -> list[str]:
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


def _get_channel_dim(ds: xr.Dataset) -> str:
    if "channel" in ds.dims or "channel" in ds.coords:
        return "channel"
    if "level" in ds.dims or "level" in ds.coords:
        return "level"
    raise KeyError(f"Dataset missing channel/level dim. dims: {list(ds.dims)}")


def _open_dataarray_robust(path: str) -> xr.DataArray:
    try:
        return xr.open_dataarray(path)
    except Exception:
        ds = xr.open_dataset(path)
        if len(ds.data_vars) == 0:
            raise ValueError(f"No data_vars found in {path}")
        return ds[list(ds.data_vars)[0]]


def _to_channel_first_stats(arr: np.ndarray, expected_c: int) -> np.ndarray:
    if arr.ndim == 1:
        return arr[:, None, None]
    if arr.ndim == 3:
        if arr.shape[0] == expected_c:
            return arr
        if arr.shape[-1] == expected_c:
            return np.transpose(arr, (2, 0, 1))
    raise ValueError(f"Unsupported stats shape {arr.shape}, expected channel dim={expected_c}")


def _align_stats_dataarray(da: xr.DataArray, channels: list[str], name: str) -> xr.DataArray:
    if "channel" in da.dims or "channel" in da.coords:
        da_channels = _decode_channel_values(da["channel"].values)
        da = da.assign_coords(channel=da_channels)
        missing = [ch for ch in channels if ch not in da_channels]
        if missing:
            raise ValueError(f"{name} missing channels: {missing[:20]}{'...' if len(missing) > 20 else ''}")
        return da.sel(channel=channels)
    return da


def load_era5_stats(era5_zarr_path: str, channels: list[str]):
    mean_da = _align_stats_dataarray(
        _open_dataarray_robust(os.path.join(era5_zarr_path, "mean.nc")), channels, "ERA5 mean"
    )
    std_da = _align_stats_dataarray(
        _open_dataarray_robust(os.path.join(era5_zarr_path, "std.nc")), channels, "ERA5 std"
    )

    mean_np = _to_channel_first_stats(mean_da.values.astype(np.float32), expected_c=len(channels))
    std_np = _to_channel_first_stats(std_da.values.astype(np.float32), expected_c=len(channels))
    return torch.from_numpy(mean_np), torch.from_numpy(std_np)


def read_source_data(zarr_path: str, ts: pd.Timestamp, channels: list[str]) -> torch.Tensor:
    ds = xr.open_zarr(zarr_path, consolidated=False)
    chan_dim = _get_channel_dim(ds)
    src_channels = _decode_channel_values(ds[chan_dim].values)
    missing = [ch for ch in channels if ch not in src_channels]
    if missing:
        ds.close()
        raise ValueError(f"Source zarr missing channels: {missing[:20]}{'...' if len(missing) > 20 else ''}")
    chan_indices = [src_channels.index(ch) for ch in channels]
    data = ds["data"].sel(time=ts).isel({chan_dim: chan_indices}).values.astype(np.float32)
    ds.close()
    return torch.from_numpy(data)


def read_lat_lon(zarr_path: str):
    ds = xr.open_zarr(zarr_path, consolidated=False)
    lat_name = "lat" if "lat" in ds.coords or "lat" in ds.dims else "latitude"
    lon_name = "lon" if "lon" in ds.coords or "lon" in ds.dims else "longitude"
    lat = ds[lat_name].values.astype(np.float32)
    lon = ds[lon_name].values.astype(np.float32)
    ds.close()
    return lat, lon


def denormalize(data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    out = data * std + mean
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def expand_channels_tensor(data: torch.Tensor, source_channels: list[str], target_channels: list[str]) -> torch.Tensor:
    """Scatter [C,H,W] into the 226-channel order. Missing target channels stay zero."""
    src_to_idx = {ch: i for i, ch in enumerate(source_channels)}
    target_set = set(target_channels)
    missing = [ch for ch in source_channels if ch not in target_set]
    if missing:
        raise ValueError(f"Source channels not present in target channel list: {missing}")

    out = data.new_zeros((len(target_channels), *data.shape[1:]))
    for target_idx, ch in enumerate(target_channels):
        src_idx = src_to_idx.get(ch)
        if src_idx is not None:
            out[target_idx] = data[src_idx]
    return out


def save_a2e_c226_nc(
    data_226: np.ndarray,
    path: str | Path,
    init_time: pd.Timestamp,
    channels: list[str],
    lat,
    lon,
    space: str,
    source_name: str,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        {"data": (["time", "channel", "lat", "lon"], data_226[None].astype(np.float32, copy=False))},
        coords={
            "time": pd.DatetimeIndex([init_time]),
            "channel": channels,
            "lat": np.asarray(lat, dtype=np.float32),
            "lon": np.asarray(lon, dtype=np.float32),
        },
        attrs={
            "source": source_name,
            "space": space,
            "description": "A2EC226 165-channel output expanded to 226 channels; missing channels zero-filled.",
            "zero_filled_channels": "cc*, clwc*, ciwc*, tp",
        },
    )
    ds.to_netcdf(path)


# ============================================================
# Inference
# ============================================================
@torch.no_grad()
def run_a2e_c226_once(
    init_time: pd.Timestamp,
    source_idx: int,
    source_path: str,
    source_name: str,
    a2e_model: torch.nn.Module,
    channels_165: list[str],
    lat,
    lon,
    output_path: Path,
    device: torch.device,
    save_space: str,
    era5_mean_165: torch.Tensor | None = None,
    era5_std_165: torch.Tensor | None = None,
):
    src_norm = read_source_data(source_path, init_time, channels_165).to(device)
    domains_tensor = torch.tensor([source_idx], dtype=torch.long, device=device)
    times_arr = np.array([str(init_time)])

    output = a2e_model(src_norm.unsqueeze(0), times=times_arr, domains=domains_tensor)
    a2e_norm_165 = output[0].squeeze(0) if isinstance(output, tuple) else output.squeeze(0)

    if save_space == "normalized":
        data_165 = a2e_norm_165
    elif save_space == "physical":
        if era5_mean_165 is None or era5_std_165 is None:
            raise ValueError("physical save_space requires ERA5 mean/std")
        data_165 = denormalize(a2e_norm_165, era5_mean_165, era5_std_165)
    else:
        raise ValueError(f"Unknown save_space: {save_space}")

    data_226 = expand_channels_tensor(data_165, channels_165, CHANNELS_226)
    save_a2e_c226_nc(
        data_226.float().cpu().numpy(),
        output_path,
        init_time,
        CHANNELS_226,
        lat,
        lon,
        save_space,
        source_name,
    )


# ============================================================
# Entry point
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="A2EC226 165-channel output -> 226-channel NetCDF")
    parser.add_argument("--gfs_path", type=str, default=DEFAULT_GFS_PATH)
    parser.add_argument("--hres_path", type=str, default=DEFAULT_HRES_PATH)
    parser.add_argument("--cma_path", type=str, default=DEFAULT_CMA_PATH)
    parser.add_argument("--era5_dir", type=str, default=DEFAULT_ERA5_PATH,
                        help="Only used when --save_space physical")
    parser.add_argument("--a2e_ckpt", type=str, default=DEFAULT_A2E_CKPT)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sources", type=str, nargs="+", default=["gfs", "hres", "cma"],
                        help="Source domains to run")
    parser.add_argument("--dates", type=str, nargs="+", default=DEFAULT_DATES,
                        help="Init dates YYYYMMDD. Default: 20250101-20250131")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_sources", type=int, default=len(SOURCE_REGISTRY),
                        help="Use 3 for mainc226_res.py multi-source checkpoints; use 1 for single-source checkpoints")
    parser.add_argument("--save_space", choices=["normalized", "physical"], default="normalized",
                        help="Save A2E output in normalized model space or denormalized physical space")
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args()

    from data.pairsetc226 import TARGET_CHANNELS as C165
    channels_165 = list(C165)

    source_paths = {
        "gfs": args.gfs_path,
        "hres": args.hres_path,
        "cma": args.cma_path,
    }

    active_sources = []
    for src_name in args.sources:
        if src_name not in SOURCE_REGISTRY:
            print(f"Warning: unknown source '{src_name}', skipping")
            continue
        src_path = source_paths.get(src_name)
        if src_path is None:
            print(f"Warning: no path configured for '{src_name}', skipping")
            continue
        source_idx = 0 if args.num_sources == 1 else SOURCE_REGISTRY[src_name]
        active_sources.append((src_name, source_idx, src_path))

    if not active_sources:
        raise ValueError("No valid sources configured. Use --sources gfs hres cma")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Building A2EC226 model (num_sources={args.num_sources})...")
    a2e_model = build_a2e_model(device, args.a2e_ckpt, num_sources=args.num_sources)
    print(f"A2E loaded from {args.a2e_ckpt}")

    era5_mean_165 = era5_std_165 = None
    if args.save_space == "physical":
        era5_mean_165, era5_std_165 = load_era5_stats(args.era5_dir, channels_165)
        era5_mean_165 = era5_mean_165.to(device)
        era5_std_165 = era5_std_165.to(device)

    for source_name, source_idx, source_path in active_sources:
        lat, lon = read_lat_lon(source_path)
        out_dir = Path(args.output_dir) / source_name / f"a2e_c226_{args.save_space}_nc"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"Source: {source_name} (idx={source_idx})")
        print(f"Data path: {source_path}")
        print(f"Save space: {args.save_space}")
        print(f"Output dir: {out_dir}")
        print(f"Dates: {len(args.dates)}")
        print(f"{'=' * 60}")

        for date_str in tqdm(args.dates, desc=f"A2EC226 [{source_name}]"):
            init_time = pd.Timestamp(f"{date_str} 00:00:00")
            output_path = out_dir / f"{date_str}.nc"
            if output_path.exists() and not args.overwrite:
                tqdm.write(f"  [{source_name}] {date_str}: exists, skipping")
                continue

            try:
                run_a2e_c226_once(
                    init_time=init_time,
                    source_idx=source_idx,
                    source_path=source_path,
                    source_name=source_name,
                    a2e_model=a2e_model,
                    channels_165=channels_165,
                    lat=lat,
                    lon=lon,
                    output_path=output_path,
                    device=device,
                    save_space=args.save_space,
                    era5_mean_165=era5_mean_165,
                    era5_std_165=era5_std_165,
                )
                tqdm.write(f"  [{source_name}] {date_str}: saved {output_path}")
            except Exception as e:
                tqdm.write(f"  [{source_name}] {date_str}: FAILED - {e}")
                import traceback
                traceback.print_exc()

    torch.cuda.empty_cache()
    print("\nAll A2EC226 outputs complete.")


if __name__ == "__main__":
    main()
