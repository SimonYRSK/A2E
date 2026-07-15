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
from fuxi.fuxi_grad import UTransformer, FuXi, time_encoding
from models.swinUNET_res import A2E

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True

DEFAULT_GFS_PATH = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/data/gfs.2022_2025_0p25.norm.zarr"
DEFAULT_HRES_PATH = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/data/hres_0p25_2022_2025_c70.zarr"
DEFAULT_CMA_PATH = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/data/cma_gfs_2020_2026.c226.norm.zarr"
DEFAULT_ERA5_PATH = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/data/era5.2020_2025_norm.zarr"
DEFAULT_FUXI_DIR = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/fuxi_inference/main/fuxi"
DEFAULT_CLIM_PATH = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/fanjiang/eval/era5/clim.daily"

HOURS_PER_STEP = 6
DEFAULT_VARIABLES = "z500,t2m,t850,ws10,ws850,msl"


def parse_items(v: str) -> list[str]:
    return [x.strip().lower() for x in str(v).replace(";", ",").split(",") if x.strip()]


def parse_dates(v: str) -> list[pd.Timestamp]:
    text = str(v).strip()
    if ":" in text and "," not in text and ";" not in text:
        parts = [x.strip() for x in text.split(":") if x.strip()]
        if len(parts) != 2:
            raise ValueError(f"Date range must be START:END, got {v!r}")
        return pd.date_range(pd.Timestamp(parts[0]), pd.Timestamp(parts[1]), freq="D").to_list()
    return [pd.Timestamp(x) for x in parse_items(v)]


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


def open_dataarray_robust(path: str) -> xr.DataArray:
    try:
        return xr.open_dataarray(path)
    except Exception:
        ds = xr.open_dataset(path)
        if len(ds.data_vars) == 0:
            raise ValueError(f"No data_vars found in {path}")
        return ds[list(ds.data_vars)[0]]


def to_channel_first_stats(arr: np.ndarray, expected_c: int) -> np.ndarray:
    if arr.ndim == 1:
        return arr[:, None, None]
    if arr.ndim == 3:
        if arr.shape[0] == expected_c:
            return arr
        if arr.shape[-1] == expected_c:
            return np.transpose(arr, (2, 0, 1))
    raise ValueError(f"Unsupported stats shape {arr.shape}, expected channel dim={expected_c}")


def load_stats(era5_path: str, channels: list[str], device: torch.device):
    mean_da = open_dataarray_robust(os.path.join(era5_path, "mean.nc"))
    std_da = open_dataarray_robust(os.path.join(era5_path, "std.nc"))
    if "channel" in mean_da.dims:
        mean_da = mean_da.sel(channel=channels)
        std_da = std_da.sel(channel=channels)

    mean = to_channel_first_stats(mean_da.values.astype(np.float32), len(channels))
    std = to_channel_first_stats(std_da.values.astype(np.float32), len(channels))
    return torch.from_numpy(mean).to(device), torch.from_numpy(std).to(device)


def read_norm_field(ds, ts: pd.Timestamp, channels: list[str], device: torch.device):
    time_dim = get_time_dim(ds)
    chan_dim = get_channel_dim(ds)
    chan_vals = decode_values(ds[chan_dim].values)
    idx = [chan_vals.index(ch) for ch in channels]
    arr = ds["data"].sel({time_dim: ts}).isel({chan_dim: idx}).values.astype(np.float32)
    return torch.from_numpy(arr).to(device)


def denormalize(data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    if data.ndim == 4 and mean.ndim == 3:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    out = data * std + mean
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def zero_tp_channel(data: torch.Tensor, channels: list[str]) -> torch.Tensor:
    if "tp" not in channels:
        return data
    data = data.clone()
    data[..., channels.index("tp"), :, :] = 0.0
    return data


def lat_weights_like(shape, device: torch.device, lat_values=None, normalize: bool = False):
    h = shape[-2]
    if lat_values is None:
        lat = torch.linspace(-90 + 180 / (2 * h), 90 - 180 / (2 * h), h, device=device)
    else:
        lat = torch.as_tensor(lat_values, dtype=torch.float32, device=device)
        if lat.numel() != h:
            lat = torch.linspace(-90 + 180 / (2 * h), 90 - 180 / (2 * h), h, device=device)
    w = torch.cos(torch.deg2rad(torch.abs(lat))).view(-1, 1)
    if normalize:
        w = w / (w.mean() + 1e-12)
    return w


def match_lat_weights(w: torch.Tensor, pred: torch.Tensor):
    if w.shape[-2:] == pred.shape[-2:]:
        return w
    return F.interpolate(w[None, None], size=pred.shape[-2:], mode="bilinear", align_corners=False).squeeze(0).squeeze(0)


def compute_rmse(pred: torch.Tensor, truth: torch.Tensor, w: torch.Tensor):
    pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    truth = torch.nan_to_num(truth, nan=0.0, posinf=0.0, neginf=0.0)
    if pred.shape != truth.shape:
        truth = F.interpolate(truth[None, None], size=pred.shape[-2:], mode="bilinear", align_corners=False).squeeze()
    w = match_lat_weights(w, pred)
    return torch.sqrt((((pred - truth) ** 2) * w).sum() / (w.sum() + 1e-12) + 1e-12)


def compute_acc(pred: torch.Tensor, truth: torch.Tensor, clim: torch.Tensor, w: torch.Tensor):
    pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    truth = torch.nan_to_num(truth, nan=0.0, posinf=0.0, neginf=0.0)
    clim = torch.nan_to_num(clim, nan=0.0, posinf=0.0, neginf=0.0)
    if truth.shape != pred.shape:
        truth = F.interpolate(truth[None, None], size=pred.shape[-2:], mode="bilinear", align_corners=False).squeeze()
    if clim.shape != pred.shape:
        clim = F.interpolate(clim[None, None], size=pred.shape[-2:], mode="bilinear", align_corners=False).squeeze()
    w = match_lat_weights(w, pred)
    pred_anom = pred - clim
    truth_anom = truth - clim
    a = (w * pred_anom * truth_anom).sum()
    b = torch.sqrt((w * pred_anom ** 2).sum() * (w * truth_anom ** 2).sum() + 1e-12)
    return a / b


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


def wind_components_for(var: str) -> tuple[str, str] | None:
    if var == "ws10":
        return "u10m", "v10m"
    if var.startswith("ws") and var[2:].isdigit():
        level = var[2:]
        return f"u{level}", f"v{level}"
    return None


def variable_field(var: str, field: torch.Tensor, channel_to_idx: dict[str, int]):
    wind_components = wind_components_for(var)
    if wind_components is not None:
        u_name, v_name = wind_components
        u = field[channel_to_idx[u_name]]
        v = field[channel_to_idx[v_name]]
        return torch.sqrt(u ** 2 + v ** 2)
    if var not in channel_to_idx:
        raise KeyError(f"Variable {var!r} not in channel list")
    return field[channel_to_idx[var]]


def variable_clim(var: str, clim_ds, ts: pd.Timestamp, device: torch.device):
    if clim_ds is None:
        return None
    wind_components = wind_components_for(var)
    if wind_components is not None:
        u_name, v_name = wind_components
        cu = get_clim(clim_ds, u_name, ts, device)
        cv = get_clim(clim_ds, v_name, ts, device)
        return torch.sqrt(cu ** 2 + cv ** 2) if cu is not None and cv is not None else None
    return get_clim(clim_ds, var, ts, device)


def build_a2e_model_from_config(config: dict, device: torch.device):
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
    print(f"Loaded A2E checkpoint {ckpt_path}; missing={len(missing)}, unexpected={len(unexpected)}")


def build_fuxi_model(device: torch.device, fuxi_dir: str, forecast_steps: int):
    conds = np.load(os.path.join(fuxi_dir, "conds.npy"))
    std = np.load(os.path.join(fuxi_dir, "std.npy"))
    mean = np.load(os.path.join(fuxi_dir, "mean.npy"))

    const = torch.from_numpy(conds).to(device=device, dtype=torch.float32)
    std_t = torch.from_numpy(std).to(device=device, dtype=torch.float32)
    mean_t = torch.from_numpy(mean).to(device=device, dtype=torch.float32)

    decoder = UTransformer(
        in_chans=75,
        out_chans=70,
        in_frames=2,
        image_size=(720, 1440),
        window_size=9,
        patch_size=4,
        down_times=1,
        embed_dim=1536,
        num_heads=24,
        depths=[12, 12, 12, 12],
    )

    model = FuXi(
        in_frames=2,
        out_frames=1,
        step_range=[forecast_steps],
        decoder=[decoder, decoder, decoder],
        const=const,
        std=std_t,
        mean=mean_t,
        device=str(device),
        dtype=torch.float32,
    ).to(device=device, dtype=torch.float32)

    model.load(fuxi_dir, fmt="pth")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def compute_initial_losses(pred_norm: torch.Tensor, truth_norm: torch.Tensor, lat_weights_norm: torch.Tensor):
    pred_norm = torch.nan_to_num(pred_norm, nan=0.0, posinf=0.0, neginf=0.0)
    truth_norm = torch.nan_to_num(truth_norm, nan=0.0, posinf=0.0, neginf=0.0)

    w = match_lat_weights(lat_weights_norm, pred_norm).view(1, pred_norm.shape[-2], pred_norm.shape[-1])
    l1_loss = torch.mean(torch.abs(pred_norm - truth_norm) * w)

    pred_dx = pred_norm[:, :, 1:] - pred_norm[:, :, :-1]
    pred_dy = pred_norm[:, 1:, :] - pred_norm[:, :-1, :]
    tgt_dx = truth_norm[:, :, 1:] - truth_norm[:, :, :-1]
    tgt_dy = truth_norm[:, 1:, :] - truth_norm[:, :-1, :]
    grad_loss = 0.5 * (torch.mean(torch.abs(pred_dx - tgt_dx)) + torch.mean(torch.abs(pred_dy - tgt_dy)))
    return float(l1_loss.detach().cpu()), float(grad_loss.detach().cpu())


def robust_data_range(truth: torch.Tensor) -> torch.Tensor:
    truth = torch.nan_to_num(truth.float(), nan=0.0, posinf=0.0, neginf=0.0)
    flat = truth.reshape(-1)
    if flat.numel() == 0:
        return torch.tensor(float("nan"), device=truth.device)
    q = torch.quantile(flat, torch.tensor([0.01, 0.99], device=truth.device))
    data_range = q[1] - q[0]
    if not torch.isfinite(data_range) or data_range <= 0:
        data_range = flat.max() - flat.min()
    return data_range


def compute_psnr(pred: torch.Tensor, truth: torch.Tensor, data_range: torch.Tensor) -> float:
    pred = torch.nan_to_num(pred.float(), nan=0.0, posinf=0.0, neginf=0.0)
    truth = torch.nan_to_num(truth.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if pred.shape != truth.shape:
        truth = F.interpolate(truth[None, None], size=pred.shape[-2:], mode="bilinear", align_corners=False).squeeze()
    if not torch.isfinite(data_range) or data_range <= 0:
        return float("nan")
    mse = torch.mean((pred - truth) ** 2)
    if mse <= 0:
        return float("inf")
    psnr = 20.0 * torch.log10(data_range / torch.sqrt(mse))
    return float(psnr.detach().cpu())


def compute_global_ssim(pred: torch.Tensor, truth: torch.Tensor, data_range: torch.Tensor) -> float:
    pred = torch.nan_to_num(pred.float(), nan=0.0, posinf=0.0, neginf=0.0)
    truth = torch.nan_to_num(truth.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if pred.shape != truth.shape:
        truth = F.interpolate(truth[None, None], size=pred.shape[-2:], mode="bilinear", align_corners=False).squeeze()
    if not torch.isfinite(data_range) or data_range <= 0:
        return float("nan")

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_pred = pred.mean()
    mu_truth = truth.mean()
    pred_centered = pred - mu_pred
    truth_centered = truth - mu_truth
    var_pred = (pred_centered ** 2).mean()
    var_truth = (truth_centered ** 2).mean()
    cov = (pred_centered * truth_centered).mean()
    ssim = ((2 * mu_pred * mu_truth + c1) * (2 * cov + c2)) / (
        (mu_pred ** 2 + mu_truth ** 2 + c1) * (var_pred + var_truth + c2) + 1e-12
    )
    return float(ssim.detach().cpu())


def summarize(rows: list[dict], group_fields: list[str], value_fields: list[str]):
    groups = {}
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        groups.setdefault(key, {field: [] for field in value_fields})
        for field in value_fields:
            val = row.get(field)
            if val is None:
                continue
            try:
                if not np.isnan(float(val)):
                    groups[key][field].append(float(val))
            except Exception:
                pass

    out = []
    for key, vals in sorted(groups.items()):
        row = {field: key[i] for i, field in enumerate(group_fields)}
        for field in value_fields:
            arr = vals[field]
            row[f"{field}_mean"] = float(np.mean(arr)) if arr else float("nan")
            row[f"n_{field}"] = len(arr)
        out.append(row)
    return out


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate FuXi rollout initialized by A2E-corrected fields")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sources", type=str, default="gfs,cma,hres")
    parser.add_argument("--dates", type=str, default="20250101")
    parser.add_argument("--variables", type=str, default=DEFAULT_VARIABLES)
    parser.add_argument("--forecast_steps", type=int, default=40)
    parser.add_argument("--era5_path", type=str, default=DEFAULT_ERA5_PATH)
    parser.add_argument("--gfs_path", type=str, default=DEFAULT_GFS_PATH)
    parser.add_argument("--hres_path", type=str, default=DEFAULT_HRES_PATH)
    parser.add_argument("--cma_path", type=str, default=DEFAULT_CMA_PATH)
    parser.add_argument("--fuxi_dir", type=str, default=DEFAULT_FUXI_DIR)
    parser.add_argument("--clim_path", type=str, default=DEFAULT_CLIM_PATH)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    sources = parse_items(args.sources)
    dates = parse_dates(args.dates)
    variables = parse_items(args.variables)
    channels = list(TARGET_CHANNELS)
    channel_to_idx = {ch: i for i, ch in enumerate(channels)}

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    a2e_model = build_a2e_model_from_config(config, device)
    load_checkpoint(a2e_model, args.ckpt, device)

    print(f"Building FuXi model for {args.forecast_steps} steps from {args.fuxi_dir}")
    fuxi_model = build_fuxi_model(device, args.fuxi_dir, args.forecast_steps)

    mean, std = load_stats(args.era5_path, channels, device)
    era5_ds = xr.open_zarr(args.era5_path, consolidated=False)
    source_paths = {"gfs": args.gfs_path, "hres": args.hres_path, "cma": args.cma_path}
    source_ds = {s: xr.open_zarr(source_paths[s], consolidated=False) for s in sources}
    clim_ds = xr.open_zarr(args.clim_path) if args.clim_path else None

    lat_vals = era5_ds["lat"].values if "lat" in era5_ds else None
    lat_weights = lat_weights_like((1, len(channels), 721, 1440), device, lat_vals, normalize=False)
    lat_weights_norm = lat_weights_like((1, len(channels), 721, 1440), device, lat_vals, normalize=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_rows = []
    initial_rows = []

    for source in sources:
        if source not in SOURCE_REGISTRY:
            print(f"skip unknown source {source!r}")
            continue
        source_idx = SOURCE_REGISTRY[source]
        ds_x = source_ds[source]

        for init_time in tqdm(dates, desc=f"rollout:{source}"):
            try:
                x_norm = read_norm_field(ds_x, init_time, channels, device)
                y0_norm = read_norm_field(era5_ds, init_time, channels, device)
                prev_time = init_time - pd.Timedelta(hours=HOURS_PER_STEP)
                era5_prev_norm = read_norm_field(era5_ds, prev_time, channels, device)
            except Exception as e:
                print(f"skip {source} {init_time}: {e}")
                continue

            with torch.no_grad():
                domains = torch.tensor([source_idx], dtype=torch.long, device=device)
                output = a2e_model(x_norm.unsqueeze(0), times=np.array([init_time]), domains=domains)
                pred_norm = output[0] if isinstance(output, tuple) else output
                pred_norm = pred_norm.squeeze(0)

            l1_loss, grad_loss = compute_initial_losses(pred_norm, y0_norm, lat_weights_norm)
            a2e_current_phys = denormalize(pred_norm, mean, std)
            y0_phys = denormalize(y0_norm, mean, std)

            for var in variables:
                try:
                    a2e_var = variable_field(var, a2e_current_phys, channel_to_idx)
                    truth0_var = variable_field(var, y0_phys, channel_to_idx)
                    data_range = robust_data_range(truth0_var)
                    psnr = compute_psnr(a2e_var, truth0_var, data_range)
                    ssim = compute_global_ssim(a2e_var, truth0_var, data_range)
                    data_range_value = float(data_range.detach().cpu()) if torch.isfinite(data_range) else float("nan")
                except Exception as e:
                    print(f"image metric skip {source} {init_time} {var}: {e}")
                    psnr = float("nan")
                    ssim = float("nan")
                    data_range_value = float("nan")

                initial_rows.append(
                    {
                        "experiment": args.exp_name,
                        "source": source,
                        "init_time": str(init_time),
                        "variable": var,
                        "a2e_l1_loss": l1_loss,
                        "a2e_grad_loss": grad_loss,
                        "a2e_psnr": psnr,
                        "a2e_ssim": ssim,
                        "a2e_data_range": data_range_value,
                    }
                )

            era5_prev_phys = denormalize(era5_prev_norm, mean, std)
            fuxi_input = torch.stack([era5_prev_phys, a2e_current_phys], dim=0)
            fuxi_input = zero_tp_channel(fuxi_input, channels)
            tembs = time_encoding(init_time, args.forecast_steps, freq=HOURS_PER_STEP).to(device=device, dtype=torch.float32)

            with torch.no_grad():
                outputs = fuxi_model.forward((fuxi_input, tembs)).squeeze(0)

            for step in range(args.forecast_steps):
                lead_hours = (step + 1) * HOURS_PER_STEP
                target_time = init_time + pd.Timedelta(hours=lead_hours)
                try:
                    truth_norm = read_norm_field(era5_ds, target_time, channels, device)
                except Exception as e:
                    print(f"truth missing {source} {init_time} lead={lead_hours}: {e}")
                    continue
                truth_phys = denormalize(truth_norm, mean, std)
                pred_phys = outputs[step]

                for var in variables:
                    try:
                        pred_var = variable_field(var, pred_phys, channel_to_idx)
                        truth_var = variable_field(var, truth_phys, channel_to_idx)
                        clim_var = variable_clim(var, clim_ds, target_time, device)
                        rmse = float(compute_rmse(pred_var, truth_var, lat_weights).detach().cpu())
                        acc = float(compute_acc(pred_var, truth_var, clim_var, lat_weights_norm).detach().cpu()) if clim_var is not None else float("nan")
                    except Exception as e:
                        print(f"metric skip {source} {init_time} lead={lead_hours} {var}: {e}")
                        rmse, acc = float("nan"), float("nan")

                    detail_rows.append(
                        {
                            "experiment": args.exp_name,
                            "source": source,
                            "init_time": str(init_time),
                            "target_time": str(target_time),
                            "lead_step": step + 1,
                            "lead_hours": lead_hours,
                            "variable": var,
                            "rmse": rmse,
                            "acc": acc,
                        }
                    )

            del outputs
            torch.cuda.empty_cache()

    detail_path = output_dir / "fuxi_rollout_metrics_detail.csv"
    write_csv(
        detail_path,
        detail_rows,
        ["experiment", "source", "init_time", "target_time", "lead_step", "lead_hours", "variable", "rmse", "acc"],
    )

    summary_rows = summarize(detail_rows, ["experiment", "source", "variable", "lead_step", "lead_hours"], ["rmse", "acc"])
    summary_path = output_dir / "fuxi_rollout_metrics_summary.csv"
    write_csv(
        summary_path,
        summary_rows,
        [
            "experiment",
            "source",
            "variable",
            "lead_step",
            "lead_hours",
            "rmse_mean",
            "n_rmse",
            "acc_mean",
            "n_acc",
        ],
    )

    initial_path = output_dir / "a2e_initial_metrics.csv"
    write_csv(
        initial_path,
        initial_rows,
        [
            "experiment",
            "source",
            "init_time",
            "variable",
            "a2e_l1_loss",
            "a2e_grad_loss",
            "a2e_psnr",
            "a2e_ssim",
            "a2e_data_range",
        ],
    )

    initial_summary_rows = summarize(
        initial_rows,
        ["experiment", "source", "variable"],
        ["a2e_l1_loss", "a2e_grad_loss", "a2e_psnr", "a2e_ssim", "a2e_data_range"],
    )
    initial_summary_path = output_dir / "a2e_initial_metrics_summary.csv"
    write_csv(
        initial_summary_path,
        initial_summary_rows,
        [
            "experiment",
            "source",
            "variable",
            "a2e_l1_loss_mean",
            "n_a2e_l1_loss",
            "a2e_grad_loss_mean",
            "n_a2e_grad_loss",
            "a2e_psnr_mean",
            "n_a2e_psnr",
            "a2e_ssim_mean",
            "n_a2e_ssim",
            "a2e_data_range_mean",
            "n_a2e_data_range",
        ],
    )

    print(f"Saved FuXi rollout detail metrics: {detail_path}")
    print(f"Saved FuXi rollout summary metrics: {summary_path}")
    print(f"Saved A2E initial diagnostics and image metrics: {initial_path}")
    print(f"Saved A2E initial diagnostics and image metric summary: {initial_summary_path}")


if __name__ == "__main__":
    main()
