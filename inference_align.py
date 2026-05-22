"""
A2E Multi-Source + FuXi Align Inference (40-step forecast).

Supports GFS, HRES, CMA → A2E → FuXi pipeline.
Results are saved per source domain under DEFAULT_OUTPUT_DIR/<source_name>/.

Usage:
    python A2E/inference_align.py                          # all sources, DEFAULT_DATES
    python A2E/inference_align.py --sources gfs hres       # selected sources
    python A2E/inference_align.py --dates 20250101 20250102
    python A2E/inference_align.py --output_dir /path/to/out
"""

import os
import sys
import argparse
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import xarray as xr
from tqdm import tqdm
from fuxi.fuxi_grad import UTransformer, FuXi, time_encoding

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True

# ============================================================
# Constants
# ============================================================
DEFAULT_DATES = []

# 70-channel list (used by mainA2E_align0520.py)
CHANNELS_70 = [
    'z50', 'z100', 'z150', 'z200', 'z250', 'z300', 'z400', 'z500',
    'z600', 'z700', 'z850', 'z925', 'z1000', 't50', 't100', 't150',
    't200', 't250', 't300', 't400', 't500', 't600', 't700', 't850',
    't925', 't1000', 'u50', 'u100', 'u150', 'u200', 'u250', 'u300',
    'u400', 'u500', 'u600', 'u700', 'u850', 'u925', 'u1000', 'v50',
    'v100', 'v150', 'v200', 'v250', 'v300', 'v400', 'v500', 'v600',
    'v700', 'v850', 'v925', 'v1000', 'r50', 'r100', 'r150', 'r200',
    'r250', 'r300', 'r400', 'r500', 'r600', 'r700', 'r850', 'r925',
    'r1000', 't2m', 'u10m', 'v10m', 'msl', 'tp',
]

Z500_IDX = CHANNELS_70.index("z500")  # 7
TP_IDX = CHANNELS_70.index("tp")
FORECAST_STEPS = 40
HOURS_PER_STEP = 6

# Source registry: must match training SOURCE_REGISTRY
SOURCE_REGISTRY = {
    "gfs": 0,
    "hres": 1,
    "cma": 2,
}

# Default paths
DEFAULT_GFS_PATH = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/data/gfs_2020_2025_c226_0p25_norm.zarr"
DEFAULT_HRES_PATH = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/data/hres_2024_2025_c226_0p25_norm.zarr"
DEFAULT_ERA5_PATH = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/fanjiang/dataset/era5.2010_2025.c226.zarr"
DEFAULT_FUXI_DIR = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/fuxi_inference/main/fuxi"
DEFAULT_A2E_CKPT = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/A2E/checkpoints/A2E_0520/checkpoint_epoch_150.pth"
DEFAULT_CLIM_PATH = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/fanjiang/eval/era5/clim.daily"
DEFAULT_OUTPUT_DIR = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/A2E/inference_results/A2E_0520"


# ============================================================
# Model builder (mirrors mainA2E_align0520.py)
# ============================================================
def build_a2e_model(device: torch.device, checkpoint_path: str, in_chans: int = 70):
    from models.swinUNET import A2E

    model = A2E(
        img_size=(721, 1440),
        patch_size=(4, 4),
        in_chans=in_chans,
        out_chans=in_chans,
        embed_dim=384,
        num_groups=32,
        num_heads=8,
        num_stages=3,
        window_size=9,
        depth=[0, 0, 1],
        using_checkpoints=True,
        using_time_embedding=True,
        using_source_embedding=True,
        num_sources=len(SOURCE_REGISTRY),
        res_per_stage=[1, 1, 1],
        channels=[384, 768, 1536],
        using_kl=False,
        dropout_rate=0.1,
        use_skip_connections=True,
        use_residual_blocks=True,
        using_dann=False,  # inference 不需要 DANN
    )

    ckpt = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    # Strip FSDP / compiled wrappers
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("_orig_mod.", "").replace("module.", "")
        new_state_dict[k] = v

    # strict=False: 训练时如有 DANN 分类器 key，inference 模型忽略
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    model.to(device)
    return model


def build_fuxi_model(device: torch.device, fuxi_dir: str):
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
        step_range=[FORECAST_STEPS],
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


# ============================================================
# Data I/O helpers
# ============================================================
def _decode_channel_values(values) -> list:
    out = []
    for v in values:
        if isinstance(v, bytes):
            out.append(v.decode())
        else:
            out.append(str(v))
    return out


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


def load_era5_stats(era5_zarr_path: str, channels: list):
    mean_da = _open_dataarray_robust(os.path.join(era5_zarr_path, "mean.nc"))
    std_da = _open_dataarray_robust(os.path.join(era5_zarr_path, "std.nc"))

    if "channel" in mean_da.dims:
        mean_da = mean_da.sel(channel=channels)
        std_da = std_da.sel(channel=channels)

    mean_np = mean_da.values.astype(np.float32)
    std_np = std_da.values.astype(np.float32)

    mean_np = _to_channel_first_stats(mean_np, expected_c=len(channels))
    std_np = _to_channel_first_stats(std_np, expected_c=len(channels))

    return torch.from_numpy(mean_np), torch.from_numpy(std_np)


def read_source_data(zarr_path: str, ts: pd.Timestamp, channels: list) -> torch.Tensor:
    """Read normalized source data (GFS/HRES/CMA) at given timestamp."""
    ds = xr.open_zarr(zarr_path, consolidated=False)
    chan_dim = _get_channel_dim(ds)
    src_channels = _decode_channel_values(ds[chan_dim].values)
    chan_indices = [src_channels.index(ch) for ch in channels]
    data = ds["data"].sel(time=ts).isel({chan_dim: chan_indices}).values.astype(np.float32)
    ds.close()
    return torch.from_numpy(data)


def read_era5_normalized(era5_zarr_path: str, ts: pd.Timestamp, channels: list) -> torch.Tensor:
    ds = xr.open_zarr(era5_zarr_path, consolidated=False)
    chan_dim = _get_channel_dim(ds)
    era5_channels = _decode_channel_values(ds[chan_dim].values)
    chan_indices = [era5_channels.index(ch) for ch in channels]
    data = ds["data"].sel(time=ts).isel({chan_dim: chan_indices}).values.astype(np.float32)
    ds.close()
    return torch.from_numpy(data)


def denormalize(data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    if data.ndim == 4 and mean.ndim == 3:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    out = data * std + mean
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# ============================================================
# Metrics
# ============================================================
def load_climatology(clim_path: str):
    return xr.open_zarr(clim_path)


def get_clim_z500(clim_ds, ts: pd.Timestamp) -> torch.Tensor:
    doy = ts.dayofyear
    hour = ts.hour
    z500_clim = clim_ds["z500"].sel(doy=doy, hour=hour).values
    return torch.from_numpy(z500_clim.astype(np.float32))


def compute_rmse(pred: torch.Tensor, truth: torch.Tensor, lat_weights: torch.Tensor) -> torch.Tensor:
    pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    truth = torch.nan_to_num(truth, nan=0.0, posinf=0.0, neginf=0.0)
    err2 = (pred - truth) ** 2
    w = lat_weights.expand_as(err2)
    weighted_mse = (err2 * w).sum() / (w.sum() + 1e-12)
    return torch.sqrt(weighted_mse + 1e-12)


def compute_acc(
    pred: torch.Tensor,
    truth: torch.Tensor,
    clim_mean: torch.Tensor,
    lat_weights_norm: torch.Tensor,
) -> torch.Tensor:
    pred_anom = pred - clim_mean
    truth_anom = truth - clim_mean
    w = lat_weights_norm
    A = (w * pred_anom * truth_anom).sum()
    B = (w * pred_anom ** 2).sum()
    C = (w * truth_anom ** 2).sum()
    return A / torch.sqrt(B * C + 1e-12)


# ============================================================
# Per-source inference
# ============================================================
def run_inference_for_source(
    init_time: pd.Timestamp,
    source_name: str,
    source_idx: int,
    source_path: str,
    a2e_model: torch.nn.Module,
    fuxi_model,
    era5_mean: torch.Tensor,
    era5_std: torch.Tensor,
    lat_weights: torch.Tensor,
    lat_weights_norm: torch.Tensor,
    clim_ds,
    era5_dir: str,
    device: torch.device,
    channels: list,
    z500_idx: int,
) -> tuple:
    """Run A2E+FuXi 40-step inference for a single source at a single init time.

    Returns (rmse_list, acc_list) each of length FORECAST_STEPS.
    """
    # 1. Read source data at init time (normalized)
    src_norm = read_source_data(source_path, init_time, channels).to(device)  # [C, H, W]

    # 2. Read ERA5 at t-6h (for FuXi cold-start input)
    t_prev = init_time - pd.Timedelta(hours=6)
    era5_prev_norm = read_era5_normalized(era5_dir, t_prev, channels).to(device)

    # 3. A2E forward: source → ERA5-like (normalized)
    times_arr = np.array([str(init_time)])
    domains_tensor = torch.tensor([source_idx], dtype=torch.long).to(device)

    with torch.no_grad():
        output = a2e_model(
            src_norm.unsqueeze(0),
            times=times_arr,
            domains=domains_tensor,
        )
    # Handle varied return types (DANN / KL may add extra elements)
    if isinstance(output, tuple):
        a2e_output_norm = output[0].squeeze(0)
    else:
        a2e_output_norm = output.squeeze(0)

    # 4. Denormalize to physical units
    era5_prev_phys = denormalize(era5_prev_norm, era5_mean, era5_std)
    a2e_output_phys = denormalize(a2e_output_norm, era5_mean, era5_std)

    # 5. Stack FuXi input: [ERA5(t-6h), A2E(t0)]
    fuxi_input = torch.stack([era5_prev_phys, a2e_output_phys], dim=0)

    # 6. Time encoding for 40 steps
    tembs = time_encoding(init_time, FORECAST_STEPS, freq=HOURS_PER_STEP)
    tembs = tembs.to(device=device, dtype=torch.float32)

    # 7. FuXi 40-step forecast
    with torch.no_grad():
        outputs = fuxi_model.forward((fuxi_input, tembs))
    outputs = outputs.squeeze(0)  # [40, 70, 721, 1440]

    # 8. Per-step Z500 metrics
    rmse_list, acc_list = [], []

    pbar_steps = tqdm(range(FORECAST_STEPS),
                      desc=f"  Steps {source_name} {init_time.strftime('%Y%m%d')}",
                      leave=False)
    for step in pbar_steps:
        lead_hours = (step + 1) * HOURS_PER_STEP
        target_time = init_time + pd.Timedelta(hours=lead_hours)

        try:
            era5_truth_norm = read_era5_normalized(era5_dir, target_time, channels).to(device)
        except Exception:
            rmse_list.append(np.nan)
            acc_list.append(np.nan)
            pbar_steps.set_postfix({"status": "truth_missing"})
            continue

        era5_truth_phys = denormalize(era5_truth_norm, era5_mean, era5_std)

        z500_pred = outputs[step, z500_idx]
        z500_truth = era5_truth_phys[z500_idx]

        rmse = compute_rmse(z500_pred, z500_truth, lat_weights)
        rmse_list.append(float(rmse.cpu()))

        clim_mean = get_clim_z500(clim_ds, target_time).to(device)
        acc = compute_acc(z500_pred, z500_truth, clim_mean, lat_weights_norm)
        acc_list.append(float(acc.cpu()))

        pbar_steps.set_postfix({"step": step + 1, "rmse": f"{rmse_list[-1]:.4f}", "acc": f"{acc_list[-1]:.4f}"})

    return rmse_list, acc_list


# ============================================================
# Output writers
# ============================================================
def write_date_results(output_dir: str, date_str: str, rmse_list: list, acc_list: list):
    txt_path = os.path.join(output_dir, f"{date_str}_z500.txt")
    with open(txt_path, "w") as f:
        f.write(f"# Init: {date_str} 00Z  |  Steps: {FORECAST_STEPS}x{HOURS_PER_STEP}h\n")
        f.write(f"# {'Step':>6s}  {'Lead(h)':>8s}  {'RMSE':>10s}  {'ACC':>10s}\n")
        for step in range(FORECAST_STEPS):
            lead_h = (step + 1) * HOURS_PER_STEP
            rmse_str = f"{rmse_list[step]:.4f}" if not np.isnan(rmse_list[step]) else "N/A"
            acc_str = f"{acc_list[step]:.4f}" if not np.isnan(acc_list[step]) else "N/A"
            f.write(f"  {step+1:>4d}  {lead_h:>8d}  {rmse_str:>10s}  {acc_str:>10s}\n")


def write_summary(output_dir: str, all_results: dict):
    """Write per-source summary with key steps across all dates."""
    summary_path = os.path.join(output_dir, "summary_all_dates.txt")
    key_steps = [0, 9, 19, 29, 39]  # step 1, 10, 20, 30, 40
    with open(summary_path, "w") as f:
        header = (
            f"# {'Date':>10s}"
            + "".join(f"  {'S{}_{}h_RMSE'.format(s+1, (s+1)*6):>14s}  {'S{}_{}h_ACC'.format(s+1, (s+1)*6):>12s}" for s in key_steps)
            + "\n"
        )
        f.write(header)
        for date_str in sorted(all_results.keys()):
            rmse_list, acc_list = all_results[date_str]
            line = f"  {date_str:>10s}"
            for s in key_steps:
                if s < len(rmse_list):
                    line += f"  {rmse_list[s]:14.4f}  {acc_list[s]:12.4f}"
                else:
                    line += f"  {'N/A':>14s}  {'N/A':>12s}"
            line += "\n"
            f.write(line)


# ============================================================
# Entry point
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="A2E Multi-Source + FuXi Inference (40-step)")
    parser.add_argument("--gfs_path", type=str, default=DEFAULT_GFS_PATH)
    parser.add_argument("--hres_path", type=str, default=DEFAULT_HRES_PATH)
    parser.add_argument("--cma_path", type=str, default=None)
    parser.add_argument("--era5_dir", type=str, default=DEFAULT_ERA5_PATH)
    parser.add_argument("--fuxi_dir", type=str, default=DEFAULT_FUXI_DIR)
    parser.add_argument("--a2e_ckpt", type=str, default=DEFAULT_A2E_CKPT)
    parser.add_argument("--clim_path", type=str, default=DEFAULT_CLIM_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sources", type=str, nargs="+", default=["gfs", "hres"],
                        help="Source domains to evaluate (default: gfs hres)")
    parser.add_argument("--dates", type=str, nargs="+", default=DEFAULT_DATES)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--in_chans", type=int, default=70,
                        help="Input channels (70 for align, 165 for c226)")
    args = parser.parse_args()

    # Resolve channels
    if args.in_chans == 70:
        channels = CHANNELS_70
        z500_idx = Z500_IDX
    else:
        # 165-channel variant: load from pairsetc226
        from data.pairsetc226 import TARGET_CHANNELS as C165
        channels = list(C165)
        z500_idx = channels.index("z500")

    # Resolve source paths
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
        active_sources.append((src_name, SOURCE_REGISTRY[src_name], src_path))

    if not active_sources:
        raise ValueError("No valid sources configured. Use --sources gfs hres")

    if not args.dates:
        print("DEFAULT_DATES 为空，启动全量测试 (2025年全年每天)...")
        args.dates = pd.date_range("2025-01-01", "2025-12-31", freq="D").strftime("%Y%m%d").tolist()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---- Build models ----
    print("Building A2E model...")
    a2e_model = build_a2e_model(device, args.a2e_ckpt, in_chans=args.in_chans)
    print(f"A2E loaded from {args.a2e_ckpt}")

    print("Building FuXi model...")
    fuxi_model = build_fuxi_model(device, args.fuxi_dir)
    print("FuXi loaded.")

    # ---- Load ERA5 stats ----
    era5_mean, era5_std = load_era5_stats(args.era5_dir, channels)
    era5_mean = era5_mean.to(device)
    era5_std = era5_std.to(device)

    # ---- Latitude weights ----
    lat = np.linspace(90, -90, 721)
    lat_w = np.cos(np.deg2rad(np.abs(lat))).astype(np.float32)
    lat_weights = torch.from_numpy(lat_w).to(device).view(-1, 1)
    lat_weights_norm = lat_weights / lat_weights.mean()

    # ---- Climatology ----
    print("Loading climatology...")
    clim_ds = load_climatology(args.clim_path)

    # ---- Run inference per source ----
    for source_name, source_idx, source_path in active_sources:
        print(f"\n{'='*60}")
        print(f"Source: {source_name} (idx={source_idx})")
        print(f"Data path: {source_path}")
        print(f"Dates: {len(args.dates)}")
        print(f"{'='*60}")

        src_output_dir = os.path.join(args.output_dir, source_name)
        os.makedirs(src_output_dir, exist_ok=True)

        all_results = {}

        for date_str in tqdm(args.dates, desc=f"Dates [{source_name}]"):
            init_time = pd.Timestamp(f"{date_str} 00:00:00")

            try:
                rmse_list, acc_list = run_inference_for_source(
                    init_time=init_time,
                    source_name=source_name,
                    source_idx=source_idx,
                    source_path=source_path,
                    a2e_model=a2e_model,
                    fuxi_model=fuxi_model,
                    era5_mean=era5_mean,
                    era5_std=era5_std,
                    lat_weights=lat_weights,
                    lat_weights_norm=lat_weights_norm,
                    clim_ds=clim_ds,
                    era5_dir=args.era5_dir,
                    device=device,
                    channels=channels,
                    z500_idx=z500_idx,
                )

                all_results[date_str] = (rmse_list, acc_list)
                write_date_results(src_output_dir, date_str, rmse_list, acc_list)

                s1_rmse = rmse_list[0] if not np.isnan(rmse_list[0]) else float("nan")
                s1_acc = acc_list[0] if not np.isnan(acc_list[0]) else float("nan")
                s40_rmse = rmse_list[-1] if not np.isnan(rmse_list[-1]) else float("nan")
                s40_acc = acc_list[-1] if not np.isnan(acc_list[-1]) else float("nan")
                print(f"  [{source_name}] {date_str}: "
                      f"Step1 RMSE={s1_rmse:.4f} ACC={s1_acc:.4f}  "
                      f"Step40 RMSE={s40_rmse:.4f} ACC={s40_acc:.4f}")

            except Exception as e:
                print(f"  [{source_name}] {date_str}: FAILED - {e}")
                import traceback
                traceback.print_exc()

        if all_results:
            write_summary(src_output_dir, all_results)
            print(f"\n[{source_name}] Results saved to {src_output_dir}")
            print(f"[{source_name}] Summary: {os.path.join(src_output_dir, 'summary_all_dates.txt')}")

    torch.cuda.empty_cache()
    print("\nAll sources complete.")


if __name__ == "__main__":
    main()
