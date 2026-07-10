import argparse
import csv
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xarray as xr
from tqdm import tqdm

from data.pairset import SOURCE_REGISTRY, TARGET_CHANNELS
from models.swinUNET_res import A2E

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True


DEFAULT_GFS_PATH = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/data/gfs_2020_2025_c226_0p25_norm.zarr"
DEFAULT_HRES_PATH = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/data/hres_2024_2025_c226_0p25_norm.zarr"
DEFAULT_CMA_PATH = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/data/cma_gfs_2020_2026.c226.norm.zarr"
DEFAULT_ERA5_PATH = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/data/era5.2020_2025_norm.zarr"


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_int_list(v):
    return [int(x.strip()) for x in str(v).split(",") if x.strip()]


def parse_sources(v):
    return [x.strip().lower() for x in str(v).replace(";", ",").split(",") if x.strip()]


def parse_variables(v):
    return [x.strip().lower() for x in str(v).replace(";", ",").split(",") if x.strip()]


def open_dataarray_robust(path: str) -> xr.DataArray:
    try:
        return xr.open_dataarray(path)
    except Exception:
        ds = xr.open_dataset(path)
        if len(ds.data_vars) == 0:
            raise ValueError(f"No data_vars found in {path}")
        return ds[list(ds.data_vars)[0]]


def get_time_dim(ds):
    for name in ["time", "datetime", "valid_time"]:
        if name in ds:
            return name
    raise ValueError(f"Cannot find time coordinate in {list(ds.variables.keys())}")


def get_channel_dim(ds):
    if "channel" in ds.dims or "channel" in ds.coords:
        return "channel"
    if "level" in ds.dims or "level" in ds.coords:
        return "level"
    raise ValueError("Cannot find channel/level dimension")


def decode_values(values):
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


def load_stats(era5_path: str, channels: list[str], device: torch.device):
    mean_da = open_dataarray_robust(os.path.join(era5_path, "mean.nc"))
    std_da = open_dataarray_robust(os.path.join(era5_path, "std.nc"))
    if "channel" in mean_da.dims:
        mean_da = mean_da.sel(channel=channels)
        std_da = std_da.sel(channel=channels)
    mean = mean_da.values.astype(np.float32)
    std = std_da.values.astype(np.float32)
    if mean.ndim == 1:
        mean = mean[:, None, None]
        std = std[:, None, None]
    elif mean.ndim == 3 and mean.shape[-1] == len(channels):
        mean = np.transpose(mean, (2, 0, 1))
        std = np.transpose(std, (2, 0, 1))
    return torch.from_numpy(mean).to(device), torch.from_numpy(std).to(device)


def read_norm_field(ds, ts: pd.Timestamp, channels: list[str], device: torch.device):
    time_dim = get_time_dim(ds)
    chan_dim = get_channel_dim(ds)
    chan_vals = decode_values(ds[chan_dim].values)
    idx = [chan_vals.index(ch) for ch in channels]
    arr = ds["data"].sel({time_dim: ts}).isel({chan_dim: idx}).values.astype(np.float32)
    return torch.from_numpy(arr).to(device)


def lat_weights_like(t: torch.Tensor, lat_values=None):
    h = t.shape[-2]
    if lat_values is None:
        lat = torch.linspace(-90 + 180 / (2 * h), 90 - 180 / (2 * h), h, device=t.device)
    else:
        lat = torch.as_tensor(lat_values, dtype=torch.float32, device=t.device)
        if lat.numel() != h:
            lat = torch.linspace(-90 + 180 / (2 * h), 90 - 180 / (2 * h), h, device=t.device)
    w = torch.cos(torch.deg2rad(torch.abs(lat))).view(-1, 1)
    return w


def compute_rmse(pred: torch.Tensor, truth: torch.Tensor, w: torch.Tensor):
    pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    truth = torch.nan_to_num(truth, nan=0.0, posinf=0.0, neginf=0.0)
    if pred.shape != truth.shape:
        truth = F.interpolate(truth[None, None], size=pred.shape[-2:], mode="bilinear", align_corners=False).squeeze()
    return torch.sqrt((((pred - truth) ** 2) * w).sum() / (w.sum() + 1e-12) + 1e-12)


def compute_acc(pred: torch.Tensor, truth: torch.Tensor, clim: torch.Tensor, w: torch.Tensor):
    pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    truth = torch.nan_to_num(truth, nan=0.0, posinf=0.0, neginf=0.0)
    clim = torch.nan_to_num(clim, nan=0.0, posinf=0.0, neginf=0.0)
    if clim.shape != pred.shape:
        clim = F.interpolate(clim[None, None], size=pred.shape[-2:], mode="bilinear", align_corners=False).squeeze()
    pred_anom = pred - clim
    truth_anom = truth - clim
    a = (w * pred_anom * truth_anom).sum()
    b = torch.sqrt((w * pred_anom ** 2).sum() * (w * truth_anom ** 2).sum() + 1e-12)
    return a / b


def build_model_from_config(config: dict, device: torch.device):
    model_cfg = config.get("model", {})
    model = A2E(
        img_size=tuple(model_cfg.get("img_size", [721, 1440])),
        patch_size=tuple(model_cfg.get("patch_size", [4, 4])),
        in_chans=int(model_cfg.get("in_chans", 70)),
        out_chans=int(model_cfg.get("out_chans", 70)),
        embed_dim=int(model_cfg.get("embed_dim", 384)),
        num_groups=int(model_cfg.get("num_groups", 32)),
        num_heads=int(model_cfg.get("num_heads", 8)),
        num_stages=int(model_cfg.get("num_stages", 3)),
        window_size=int(model_cfg.get("window_size", 9)),
        depth=list(model_cfg.get("depth", [0, 0, 1])),
        using_checkpoints=False,
        using_time_embedding=bool(model_cfg.get("using_time_embedding", True)),
        using_source_embedding=bool(model_cfg.get("using_source_embedding", True)),
        num_sources=len(SOURCE_REGISTRY),
        res_per_stage=list(model_cfg.get("res_per_stage", [1, 1, 1])),
        channels=list(model_cfg.get("channels", [384, 768, 1536])),
        using_kl=bool(model_cfg.get("using_kl", False)),
        dropout_rate=float(model_cfg.get("dropout_rate", 0.1)),
        use_skip_connections=bool(model_cfg.get("use_skip_connections", True)),
        use_residual_blocks=bool(model_cfg.get("use_residual_blocks", True)),
        using_dann=False,
    )
    return model.to(device).eval()


def load_checkpoint(model, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt
    clean = {}
    for k, v in state.items():
        k = k.replace("_orig_mod.", "").replace("module.", "").replace("_fsdp_wrapped_module.", "")
        clean[k] = v
    missing, unexpected = model.load_state_dict(clean, strict=False)
    print(f"Loaded {ckpt_path}; missing={len(missing)}, unexpected={len(unexpected)}")


def get_clim(clim_ds, var: str, ts: pd.Timestamp, device: torch.device):
    if clim_ds is None or var not in clim_ds:
        return None
    da = clim_ds[var]
    kwargs = {}
    if "doy" in da.dims or "doy" in da.coords:
        kwargs["doy"] = ts.dayofyear
    if "hour" in da.dims or "hour" in da.coords:
        kwargs["hour"] = ts.hour
    arr = da.sel(**kwargs).values.astype(np.float32)
    return torch.from_numpy(arr).to(device)


def metric_for_variable(var, pred_phys, truth_phys, channel_to_idx, w, clim_ds, ts, device):
    if var == "ws10m":
        uidx = channel_to_idx["u10m"]
        vidx = channel_to_idx["v10m"]
        pred = torch.sqrt(pred_phys[uidx] ** 2 + pred_phys[vidx] ** 2)
        truth = torch.sqrt(truth_phys[uidx] ** 2 + truth_phys[vidx] ** 2)
        cu = get_clim(clim_ds, "u10m", ts, device)
        cv = get_clim(clim_ds, "v10m", ts, device)
        clim = torch.sqrt(cu ** 2 + cv ** 2) if cu is not None and cv is not None else None
    else:
        if var not in channel_to_idx:
            raise KeyError(f"Variable {var!r} not in channel list")
        idx = channel_to_idx[var]
        pred = pred_phys[idx]
        truth = truth_phys[idx]
        clim = get_clim(clim_ds, var, ts, device)

    rmse = float(compute_rmse(pred, truth, w).detach().cpu())
    acc = float(compute_acc(pred, truth, clim, w).detach().cpu()) if clim is not None else float("nan")
    return rmse, acc


def main():
    parser = argparse.ArgumentParser(description="Evaluate A2E field RMSE/ACC for selected variables")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sources", type=str, default="cma")
    parser.add_argument("--dates", type=str, default="20250101")
    parser.add_argument("--variables", type=str, default="z500,t2m,tp,ws10m,msl,r700")
    parser.add_argument("--era5_path", type=str, default=DEFAULT_ERA5_PATH)
    parser.add_argument("--gfs_path", type=str, default=DEFAULT_GFS_PATH)
    parser.add_argument("--hres_path", type=str, default=DEFAULT_HRES_PATH)
    parser.add_argument("--cma_path", type=str, default=DEFAULT_CMA_PATH)
    parser.add_argument("--clim_path", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    sources = parse_sources(args.sources)
    variables = parse_variables(args.variables)
    dates = [pd.Timestamp(d) for d in parse_variables(args.dates)]

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    model = build_model_from_config(config, device)
    load_checkpoint(model, args.ckpt, device)

    channels = list(TARGET_CHANNELS)
    channel_to_idx = {ch: i for i, ch in enumerate(channels)}
    mean, std = load_stats(args.era5_path, channels, device)

    era5_ds = xr.open_zarr(args.era5_path, consolidated=False)
    source_paths = {"gfs": args.gfs_path, "hres": args.hres_path, "cma": args.cma_path}
    source_ds = {s: xr.open_zarr(source_paths[s], consolidated=False) for s in sources}
    clim_ds = xr.open_zarr(args.clim_path) if args.clim_path else None

    lat_vals = era5_ds["lat"].values if "lat" in era5_ds else None
    w = lat_weights_like(torch.empty(1, 721, 1440, device=device), lat_vals)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for source in sources:
        source_idx = SOURCE_REGISTRY[source]
        ds_x = source_ds[source]
        for ts in tqdm(dates, desc=f"eval:{source}"):
            try:
                x_norm = read_norm_field(ds_x, ts, channels, device)
                y_norm = read_norm_field(era5_ds, ts, channels, device)
            except Exception as e:
                print(f"skip {source} {ts}: {e}")
                continue

            with torch.no_grad():
                domains = torch.tensor([source_idx], dtype=torch.long, device=device)
                pred_norm = model(x_norm.unsqueeze(0), times=np.array([ts]), domains=domains).squeeze(0)

            pred_phys = torch.nan_to_num(pred_norm * std + mean, nan=0.0, posinf=0.0, neginf=0.0)
            truth_phys = torch.nan_to_num(y_norm * std + mean, nan=0.0, posinf=0.0, neginf=0.0)

            for var in variables:
                try:
                    rmse, acc = metric_for_variable(var, pred_phys, truth_phys, channel_to_idx, w, clim_ds, ts, device)
                except Exception as e:
                    print(f"metric skip {source} {ts} {var}: {e}")
                    rmse, acc = float("nan"), float("nan")
                rows.append(
                    {
                        "experiment": args.exp_name,
                        "source": source,
                        "time": str(ts),
                        "variable": var,
                        "rmse": rmse,
                        "acc": acc,
                    }
                )

    detail_path = output_dir / "field_metrics_detail.csv"
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["experiment", "source", "time", "variable", "rmse", "acc"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for row in rows:
        key = (row["source"], row["variable"])
        summary.setdefault(key, {"rmse": [], "acc": []})
        if not np.isnan(row["rmse"]):
            summary[key]["rmse"].append(row["rmse"])
        if not np.isnan(row["acc"]):
            summary[key]["acc"].append(row["acc"])

    summary_rows = []
    for (source, var), vals in sorted(summary.items()):
        summary_rows.append(
            {
                "experiment": args.exp_name,
                "source": source,
                "variable": var,
                "rmse_mean": float(np.mean(vals["rmse"])) if vals["rmse"] else float("nan"),
                "acc_mean": float(np.mean(vals["acc"])) if vals["acc"] else float("nan"),
                "n_rmse": len(vals["rmse"]),
                "n_acc": len(vals["acc"]),
            }
        )

    summary_path = output_dir / "field_metrics_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["experiment", "source", "variable", "rmse_mean", "acc_mean", "n_rmse", "n_acc"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved detail metrics: {detail_path}")
    print(f"Saved summary metrics: {summary_path}")


if __name__ == "__main__":
    main()
