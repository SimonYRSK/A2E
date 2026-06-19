"""
Plot denormalized ERA5/CMA/HRES/GFS fields for selected channels.

The source Zarrs are expected to already be normalized with the same ERA5
mean/std statistics. This script denormalizes each source to physical units,
checks latitude/longitude consistency, prints value ranges, and saves side-by-side
figures for visual comparison.

Example:
    python A2E/plot_denorm_sources.py --time 2025-01-01T00:00:00 --output_dir ./denorm_plots
"""

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")

DEFAULT_GFS_PATH = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/data/gfs_2020_2025_c226_0p25_norm.zarr"
DEFAULT_HRES_PATH = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/data/hres_2024_2025_c226_0p25_norm.zarr"
DEFAULT_CMA_PATH = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/data/cma_gfs_2020_2026.c226.norm.zarr"
DEFAULT_ERA5_PATH = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/fanjiang/dataset/era5.2010_2025.c226.zarr"

CHANNELS = ["z500", "t500", "u500", "v500", "r500", "t2m", "msl", 'u10m', 'v10m']
SOURCES = ["era5", "gfs"]


def open_zarr_safe(path: str) -> xr.Dataset:
    try:
        return xr.open_zarr(path, consolidated=True, decode_times=True)
    except Exception:
        return xr.open_zarr(path, consolidated=False, decode_times=True)


def decode_values(values) -> list[str]:
    out = []
    for v in values:
        if isinstance(v, bytes):
            out.append(v.decode().strip())
        else:
            out.append(str(v).strip())
    return out


def get_channel_dim(ds: xr.Dataset) -> str:
    if "channel" in ds.dims or "channel" in ds.coords:
        return "channel"
    if "level" in ds.dims or "level" in ds.coords:
        return "level"
    raise KeyError(f"Dataset missing channel/level dimension. dims={list(ds.dims)}")


def get_lat_lon_names(ds: xr.Dataset) -> tuple[str, str]:
    lat_name = "lat" if "lat" in ds.coords else "latitude"
    lon_name = "lon" if "lon" in ds.coords else "longitude"
    if lat_name not in ds.coords or lon_name not in ds.coords:
        raise KeyError(f"Dataset missing lat/lon coords. coords={list(ds.coords)}")
    return lat_name, lon_name


def repair_1d_coord(values, kind: str, source_name: str) -> np.ndarray:
    """Return finite 1D lat/lon coordinates, repairing known 0.25° global grids."""
    arr = np.asarray(values, dtype=np.float64).copy()
    if kind == "lon":
        arr = np.where(arr < 0, arr + 360, arr)

    finite = np.isfinite(arr)
    if finite.all():
        return arr.astype(np.float32)

    n_bad = int((~finite).sum())
    print(f"⚠️  {source_name}: {kind} contains {n_bad} NaN/Inf values, repairing coordinates for plotting/checking")

    # These datasets are expected to be 0.25° global 721x1440 grids.
    if kind == "lat" and arr.size == 721:
        return np.linspace(90.0, -90.0, arr.size, dtype=np.float32)
    if kind == "lon" and arr.size == 1440:
        return (np.arange(arr.size, dtype=np.float32) * 0.25).astype(np.float32)

    # Generic fallback: interpolate bad entries from finite neighbors.
    if finite.sum() >= 2:
        idx = np.arange(arr.size)
        arr[~finite] = np.interp(idx[~finite], idx[finite], arr[finite])
        return arr.astype(np.float32)

    raise ValueError(f"{source_name}: cannot repair {kind} coordinate with shape={arr.shape}")


def coord_endpoints(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(finite[0]), float(finite[-1])


def load_stats(era5_path: str, channels: list[str]) -> tuple[xr.DataArray, xr.DataArray]:
    mean_path = os.path.join(era5_path, "mean.nc")
    std_path = os.path.join(era5_path, "std.nc")
    mean_ds = xr.open_dataset(mean_path)
    std_ds = xr.open_dataset(std_path)

    if "mean" in mean_ds:
        mean = mean_ds["mean"]
    else:
        mean = mean_ds[list(mean_ds.data_vars)[0]]

    if "std" in std_ds:
        std = std_ds["std"]
    else:
        std = std_ds[list(std_ds.data_vars)[0]]

    if "channel" in mean.dims:
        mean = mean.sel(channel=channels)
        std = std.sel(channel=channels)

    return mean.astype("float32"), std.astype("float32")


def select_channels(ds: xr.Dataset, channels: list[str]) -> xr.DataArray:
    chan_dim = get_channel_dim(ds)
    src_channels = decode_values(ds[chan_dim].values)
    missing = [ch for ch in channels if ch not in src_channels]
    if missing:
        raise KeyError(f"Missing channels {missing}; available examples={src_channels[:20]}")

    indices = [src_channels.index(ch) for ch in channels]
    return ds["data"].isel({chan_dim: indices}).assign_coords({chan_dim: channels}).rename({chan_dim: "channel"})


def read_denorm_source(
    name: str,
    path: str,
    ts: pd.Timestamp,
    channels: list[str],
    mean: xr.DataArray,
    std: xr.DataArray,
) -> tuple[xr.DataArray, dict]:
    ds = open_zarr_safe(path)
    lat_name, lon_name = get_lat_lon_names(ds)

    if "time" not in ds.coords and "time" not in ds.dims:
        raise KeyError(f"{name}: missing time coordinate")

    times = pd.DatetimeIndex(ds["time"].values)
    if ts not in times:
        nearest_idx = int(np.argmin(np.abs(times - ts)))
        nearest_time = pd.Timestamp(times[nearest_idx])
        print(f"⚠️  {name}: requested {ts} not found, using nearest time {nearest_time}")
        time_sel = nearest_time
    else:
        time_sel = ts

    data = select_channels(ds, channels).sel(time=time_sel).astype("float32")

    if name == "era5":
        # ERA5 c226 zarr is normalized already; denormalize with the same mean/std.
        denorm = data * std + mean
    else:
        denorm = data * std + mean

    lat = repair_1d_coord(ds[lat_name].values, "lat", name)
    lon = repair_1d_coord(ds[lon_name].values, "lon", name)
    denorm = denorm.assign_coords(lat=lat, lon=lon)
    denorm = denorm.transpose("channel", "lat", "lon")

    info = {
        "path": path,
        "time_used": str(pd.Timestamp(time_sel)),
        "shape": tuple(denorm.shape),
        "lat": denorm["lat"].values,
        "lon": denorm["lon"].values,
    }
    ds.close()
    return denorm.load(), info


def summarize_grids(infos: dict):
    ref_name = next(iter(infos))
    ref_lat = infos[ref_name]["lat"]
    ref_lon = infos[ref_name]["lon"]

    print("\n=== Grid check ===")
    for name, info in infos.items():
        lat = info["lat"]
        lon = info["lon"]
        lat_same = lat.shape == ref_lat.shape and np.allclose(lat, ref_lat, rtol=0, atol=1e-6)
        lon_same = lon.shape == ref_lon.shape and np.allclose(lon, ref_lon, rtol=0, atol=1e-6)
        lat0, lat1 = coord_endpoints(lat)
        lon0, lon1 = coord_endpoints(lon)
        print(
            f"{name:>5s}: shape={info['shape']}, time={info['time_used']}, "
            f"lat[{lat.size}]={lat0:.4f}->{lat1:.4f}, "
            f"lon[{lon.size}]={lon0:.4f}->{lon1:.4f}, "
            f"same_as_{ref_name}: lat={lat_same}, lon={lon_same}"
        )


def summarize_values(fields: dict, channels: list[str], output_dir: str):
    print("\n=== Denormalized value ranges ===")
    csv_lines = ["source,channel,mean,std,min,p01,p99,max"]
    for ch in channels:
        print(f"\n[{ch}]")
        for name, da in fields.items():
            arr = da.sel(channel=ch).values.astype(np.float64)
            stats = {
                "mean": np.nanmean(arr),
                "std": np.nanstd(arr),
                "min": np.nanmin(arr),
                "p01": np.nanpercentile(arr, 1),
                "p99": np.nanpercentile(arr, 99),
                "max": np.nanmax(arr),
            }
            print(
                f"{name:>5s}: mean={stats['mean']:.6g}, std={stats['std']:.6g}, "
                f"min={stats['min']:.6g}, p01={stats['p01']:.6g}, "
                f"p99={stats['p99']:.6g}, max={stats['max']:.6g}"
            )
            csv_lines.append(
                f"{name},{ch},{stats['mean']},{stats['std']},{stats['min']},{stats['p01']},{stats['p99']},{stats['max']}"
            )

    csv_path = os.path.join(output_dir, "denorm_stats.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines) + "\n")
    print(f"\nStats CSV saved: {csv_path}")


def channel_groups(channels: list[str], size: int = 3) -> list[list[str]]:
    return [channels[i : i + size] for i in range(0, len(channels), size)]


def plot_channel(fields: dict, ch: str, output_dir: str, use_common_scale: bool):
    n = len(fields)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.2), constrained_layout=True)
    if n == 1:
        axes = [axes]

    arrays = {name: da.sel(channel=ch).values for name, da in fields.items()}
    if use_common_scale:
        merged = np.concatenate([arr[np.isfinite(arr)].ravel() for arr in arrays.values()])
        vmin = np.nanmin(merged)
        vmax = np.nanmax(merged)
    else:
        vmin = vmax = None

    last_im = None
    for ax, (name, da) in zip(axes, fields.items()):
        arr = arrays[name]
        lat = da["lat"].values
        lon = da["lon"].values
        this_vmin = vmin if use_common_scale else np.nanmin(arr)
        this_vmax = vmax if use_common_scale else np.nanmax(arr)
        last_im = ax.imshow(
            arr,
            origin="upper",
            extent=[float(lon[0]), float(lon[-1]), float(lat[-1]), float(lat[0])],
            aspect="auto",
            cmap="viridis",
            vmin=this_vmin,
            vmax=this_vmax,
        )
        ax.set_title(f"{name} {ch}\nmean={np.nanmean(arr):.4g}, min={np.nanmin(arr):.4g}, max={np.nanmax(arr):.4g}")
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        fig.colorbar(last_im, ax=ax, shrink=0.82)

    scale_label = "common" if use_common_scale else "per_source"
    fig.suptitle(f"Denormalized {ch} ({scale_label} color scale)", fontsize=14)
    out_path = os.path.join(output_dir, f"denorm_{ch}_{scale_label}.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Figure saved: {out_path}")


def plot_channel_group_common(fields: dict, channels: list[str], output_dir: str):
    n_rows = len(channels)
    n_cols = len(fields)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 3.7 * n_rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(n_rows, n_cols)

    for row, ch in enumerate(channels):
        arrays = {name: da.sel(channel=ch).values for name, da in fields.items()}
        merged = np.concatenate([arr[np.isfinite(arr)].ravel() for arr in arrays.values()])
        vmin = np.nanmin(merged)
        vmax = np.nanmax(merged)
        row_images = []

        for col, (name, da) in enumerate(fields.items()):
            ax = axes[row, col]
            arr = arrays[name]
            lat = da["lat"].values
            lon = da["lon"].values
            im = ax.imshow(
                arr,
                origin="upper",
                extent=[float(lon[0]), float(lon[-1]), float(lat[-1]), float(lat[0])],
                aspect="auto",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            row_images.append(im)
            ax.set_title(f"{name} {ch}\nmean={np.nanmean(arr):.4g}, min={np.nanmin(arr):.4g}, max={np.nanmax(arr):.4g}")
            ax.set_xlabel("lon")
            ax.set_ylabel("lat")

        fig.colorbar(row_images[-1], ax=axes[row, :], shrink=0.82, label=ch)

    group_label = "_".join(channels)
    fig.suptitle(f"Denormalized {' / '.join(channels)} (common color scale per channel)", fontsize=14)
    out_path = os.path.join(output_dir, f"denorm_{group_label}_common.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Grouped figure saved: {out_path}")


def plot_diff_to_era5(fields: dict, ch: str, output_dir: str):
    if "era5" not in fields:
        return

    era5 = fields["era5"].sel(channel=ch).values
    sources = [(name, da) for name, da in fields.items() if name != "era5"]
    if not sources:
        return

    fig, axes = plt.subplots(1, len(sources), figsize=(5.2 * len(sources), 4.2), constrained_layout=True)
    if len(sources) == 1:
        axes = [axes]

    diffs = {name: da.sel(channel=ch).values - era5 for name, da in sources}
    merged = np.concatenate([arr[np.isfinite(arr)].ravel() for arr in diffs.values()])
    vmax = np.nanmax(np.abs(merged))
    vmin = -vmax

    for ax, (name, da) in zip(axes, sources):
        diff = diffs[name]
        lat = da["lat"].values
        lon = da["lon"].values
        im = ax.imshow(
            diff,
            origin="upper",
            extent=[float(lon[0]), float(lon[-1]), float(lat[-1]), float(lat[0])],
            aspect="auto",
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"{name} - ERA5 {ch}\nmean={np.nanmean(diff):.4g}, rmse={np.sqrt(np.nanmean(diff ** 2)):.4g}")
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        fig.colorbar(im, ax=ax, shrink=0.82)

    out_path = os.path.join(output_dir, f"diff_to_era5_{ch}.png")
    fig.suptitle(f"Denormalized difference to ERA5: {ch}", fontsize=14)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Diff figure saved: {out_path}")


def plot_diff_group_to_era5(fields: dict, channels: list[str], output_dir: str):
    if "era5" not in fields:
        return

    sources = [(name, da) for name, da in fields.items() if name != "era5"]
    if not sources:
        return

    n_rows = len(channels)
    n_cols = len(sources)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 3.7 * n_rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(n_rows, n_cols)

    for row, ch in enumerate(channels):
        era5 = fields["era5"].sel(channel=ch).values
        diffs = {name: da.sel(channel=ch).values - era5 for name, da in sources}
        merged = np.concatenate([arr[np.isfinite(arr)].ravel() for arr in diffs.values()])
        vmax = np.nanmax(np.abs(merged))
        vmin = -vmax
        row_images = []

        for col, (name, da) in enumerate(sources):
            ax = axes[row, col]
            diff = diffs[name]
            lat = da["lat"].values
            lon = da["lon"].values
            im = ax.imshow(
                diff,
                origin="upper",
                extent=[float(lon[0]), float(lon[-1]), float(lat[-1]), float(lat[0])],
                aspect="auto",
                cmap="RdBu_r",
                vmin=vmin,
                vmax=vmax,
            )
            row_images.append(im)
            ax.set_title(f"{name} - ERA5 {ch}\nmean={np.nanmean(diff):.4g}, rmse={np.sqrt(np.nanmean(diff ** 2)):.4g}")
            ax.set_xlabel("lon")
            ax.set_ylabel("lat")

        fig.colorbar(row_images[-1], ax=axes[row, :], shrink=0.82, label=ch)

    group_label = "_".join(channels)
    out_path = os.path.join(output_dir, f"diff_to_era5_{group_label}.png")
    fig.suptitle(f"Denormalized difference to ERA5: {' / '.join(channels)}", fontsize=14)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Grouped diff figure saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot denormalized source fields and grid checks")
    parser.add_argument("--gfs_path", default=DEFAULT_GFS_PATH)
    parser.add_argument("--hres_path", default=DEFAULT_HRES_PATH)
    parser.add_argument("--cma_path", default=DEFAULT_CMA_PATH)
    parser.add_argument("--era5_path", default=DEFAULT_ERA5_PATH)
    parser.add_argument("--time", default="2025-01-01T00:00:00", help="Timestamp to plot")
    parser.add_argument("--channels", nargs="+", default=CHANNELS)
    parser.add_argument("--sources", nargs="+", default=SOURCES, choices=SOURCES)
    parser.add_argument("--output_dir", default="denorm_source_plots")
    parser.add_argument("--no_common_scale", action="store_true", help="Disable common 1-99 percentile color scale")
    parser.add_argument("--plot_diff", action="store_true", help="Also plot source - ERA5 difference maps")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ts = pd.Timestamp(args.time)

    paths = {
        "era5": args.era5_path,
        "gfs": args.gfs_path,
        "hres": args.hres_path,
        "cma": args.cma_path,
    }

    print(f"Requested time: {ts}")
    print(f"Channels: {args.channels}")
    print(f"Output dir: {args.output_dir}")

    mean, std = load_stats(args.era5_path, args.channels)

    fields = {}
    infos = {}
    for name in args.sources:
        print(f"\nLoading {name}: {paths[name]}")
        fields[name], infos[name] = read_denorm_source(name, paths[name], ts, args.channels, mean, std)

    summarize_grids(infos)
    summarize_values(fields, args.channels, args.output_dir)

    if args.no_common_scale:
        for ch in args.channels:
            plot_channel(fields, ch, args.output_dir, use_common_scale=False)
            if args.plot_diff:
                plot_diff_to_era5(fields, ch, args.output_dir)
    else:
        for channels in channel_groups(args.channels):
            plot_channel_group_common(fields, channels, args.output_dir)
            if args.plot_diff:
                plot_diff_group_to_era5(fields, channels, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
