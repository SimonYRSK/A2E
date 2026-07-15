#!/usr/bin/env python3
"""Scan CMA normalized samples for physically suspicious values.

This is a standalone diagnostic script. It does not modify training code or data.
It reuses Any2ERA5Dataset/DataLoader, denormalizes CMA inputs with ERA5 stats,
and logs timestamps/channels whose physical min/max exceed lenient CSV bounds.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import DataLoader

A2E_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = A2E_ROOT.parent
if str(A2E_ROOT) not in sys.path:
    sys.path.insert(0, str(A2E_ROOT))

from data.pairset import Any2ERA5Dataset, SOURCE_REGISTRY, TARGET_CHANNELS  # noqa: E402


DEFAULT_ERA5_PATH = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/data/era5.2020_2025_norm.zarr"
DEFAULT_CMA_PATH = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/data/cma_gfs_2020_2026.c226.norm.zarr"
DEFAULT_STATS_CSV = "/home/ximutian/denorm_source_plots/denorm_stats.csv"
DEFAULT_LOG_DIR = A2E_ROOT / "data_quality_logs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check CMA samples for out-of-range physical values")
    parser.add_argument("--cma_path", type=str, default=DEFAULT_CMA_PATH)
    parser.add_argument("--era5_path", type=str, default=DEFAULT_ERA5_PATH)
    parser.add_argument("--stats_csv", type=str, default=str(DEFAULT_STATS_CSV))
    parser.add_argument("--stats_source", type=str, default="cma", choices=["cma", "era5"])
    parser.add_argument("--start", type=str, default="2022-01-01 00:00:00")
    parser.add_argument("--end", type=str, default="2024-12-31 18:00:00")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=1)
    parser.add_argument("--margin_ratio", type=float, default=0.10,
                        help="Lenient bound margin as ratio of CSV max-min range")
    parser.add_argument("--margin_abs", type=float, default=0.0,
                        help="Extra absolute margin added on both sides")
    parser.add_argument("--channels", type=str, default="",
                        help="Comma-separated channel allowlist. Empty means TARGET_CHANNELS.")
    parser.add_argument("--ignore_channels", type=str, default="tp",
                        help="Comma-separated channels to skip. Default skips CMA-missing tp.")
    parser.add_argument("--include_zero_range_channels", action="store_true",
                        help="Check channels whose CSV min==max. Disabled by default.")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Stop after N samples. 0 means scan all samples in date range.")
    parser.add_argument("--log_dir", type=str, default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--log_name", type=str, default="",
                        help="CSV log filename. Empty creates timestamped filename.")
    return parser.parse_args()


def custom_collate(batch):
    x, y, source_idx, times = zip(*batch)
    times = np.array([pd.Timestamp(str(t)) for t in times])
    domains = torch.as_tensor(source_idx, dtype=torch.long)
    return torch.stack(x), torch.stack(y), domains, times


def _open_dataarray_robust(path: Path) -> xr.DataArray:
    try:
        return xr.open_dataarray(path)
    except Exception:
        ds = xr.open_dataset(path)
        if len(ds.data_vars) == 0:
            raise ValueError(f"No data variables found in {path}")
        first = list(ds.data_vars)[0]
        return ds[first]


def _as_channel_vector(da: xr.DataArray, channels: list[str]) -> np.ndarray:
    if "channel" in da.dims:
        da = da.sel(channel=channels)
    arr = da.values.astype(np.float32)
    if arr.ndim == 1:
        return arr
    # Stats files may be [C, H, W] or [H, W, C]. For this scan we need a scalar
    # per channel. If spatial stats exist, average them for denormalizing bounds.
    if arr.shape[0] == len(channels):
        return arr.reshape(len(channels), -1).mean(axis=1).astype(np.float32)
    if arr.shape[-1] == len(channels):
        arr = np.moveaxis(arr, -1, 0)
        return arr.reshape(len(channels), -1).mean(axis=1).astype(np.float32)
    raise ValueError(f"Unsupported stats shape {arr.shape}; expected channel count {len(channels)}")


def load_era5_stats(era5_path: str, channels: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    era5_dir = Path(era5_path)
    mean_da = _open_dataarray_robust(era5_dir / "mean.nc")
    std_da = _open_dataarray_robust(era5_dir / "std.nc")
    mean = _as_channel_vector(mean_da, channels)
    std = _as_channel_vector(std_da, channels)
    std = np.maximum(std, 1e-8).astype(np.float32)
    return torch.from_numpy(mean)[:, None, None], torch.from_numpy(std)[:, None, None]


def load_bounds(
    csv_path: str,
    source: str,
    channels: list[str],
    margin_ratio: float,
    margin_abs: float,
    include_zero_range_channels: bool,
) -> dict[str, dict[str, float]]:
    df = pd.read_csv(csv_path)
    required = {"source", "channel", "min", "max", "mean", "std", "p01", "p99"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Stats CSV missing columns: {sorted(missing)}")

    df = df[df["source"].astype(str).str.lower() == source.lower()].copy()
    if df.empty:
        raise ValueError(f"No rows found in {csv_path} for source={source}")

    bounds: dict[str, dict[str, float]] = {}
    wanted = set(channels)
    for row in df.itertuples(index=False):
        ch = str(row.channel).strip()
        if ch not in wanted:
            continue
        vmin = float(row.min)
        vmax = float(row.max)
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            continue
        span = vmax - vmin
        if span <= 0.0 and not include_zero_range_channels:
            continue
        margin = margin_ratio * max(span, 0.0) + margin_abs
        bounds[ch] = {
            "csv_min": vmin,
            "csv_max": vmax,
            "lower": vmin - margin,
            "upper": vmax + margin,
            "csv_mean": float(row.mean),
            "csv_std": float(row.std),
            "p01": float(row.p01),
            "p99": float(row.p99),
        }
    return bounds


def split_csv(value: str) -> list[str]:
    return [v.strip() for v in str(value).split(",") if v.strip()]


def make_log_path(log_dir: str, log_name: str) -> Path:
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    if not log_name:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_name = f"cma_data_quality_{stamp}.csv"
    return path / log_name


def main() -> None:
    args = parse_args()

    channels = split_csv(args.channels) if args.channels else list(TARGET_CHANNELS)
    ignore_channels = set(split_csv(args.ignore_channels))
    channels = [ch for ch in channels if ch not in ignore_channels]

    bounds = load_bounds(
        csv_path=args.stats_csv,
        source=args.stats_source,
        channels=channels,
        margin_ratio=args.margin_ratio,
        margin_abs=args.margin_abs,
        include_zero_range_channels=args.include_zero_range_channels,
    )
    check_channels = [ch for ch in channels if ch in bounds]
    if not check_channels:
        raise RuntimeError("No channels left to check after applying CSV bounds and ignore list")

    channel_to_idx = {ch: i for i, ch in enumerate(TARGET_CHANNELS)}
    check_indices = [channel_to_idx[ch] for ch in check_channels]

    mean, std = load_era5_stats(args.era5_path, list(TARGET_CHANNELS))

    dataset = Any2ERA5Dataset(
        target_channels=list(TARGET_CHANNELS),
        start=args.start,
        end=args.end,
        x_path=args.cma_path,
        y_path=args.era5_path,
        source_name="cma",
        source_idx=SOURCE_REGISTRY.get("cma", 2),
        max_samples_per_year=None,
        sample_seed=43,
    )

    loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
        collate_fn=custom_collate,
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(dataset, **loader_kwargs)

    log_path = make_log_path(args.log_dir, args.log_name)
    print(f"Dataset samples: {len(dataset)}")
    print(f"Checking channels: {len(check_channels)} / {len(TARGET_CHANNELS)}")
    print(f"Ignored channels: {sorted(ignore_channels)}")
    print(f"Bounds: source={args.stats_source}, margin_ratio={args.margin_ratio}, margin_abs={args.margin_abs}")
    print(f"Log path: {log_path}")

    fieldnames = [
        "timestamp", "source", "channel", "batch_idx", "sample_in_batch", "global_sample_idx",
        "observed_min", "observed_max", "observed_mean", "allowed_lower", "allowed_upper",
        "csv_min", "csv_max", "p01", "p99", "finite_ratio", "nonfinite_count", "violation",
        "below_by", "above_by",
    ]

    bad_timestamps: set[str] = set()
    bad_counter: Counter[str] = Counter()
    violation_counter: Counter[str] = Counter()
    scanned = 0
    rows_written = 0
    global_idx = 0

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for batch_idx, (x_norm, _y, _domains, times) in enumerate(loader):
            x_phys = x_norm.float() * std.unsqueeze(0) + mean.unsqueeze(0)
            # [B, selected_C, H, W]
            selected = x_phys[:, check_indices]

            for b in range(selected.shape[0]):
                timestamp = str(pd.Timestamp(times[b]))
                for local_cidx, ch in enumerate(check_channels):
                    arr = selected[b, local_cidx]
                    finite = torch.isfinite(arr)
                    finite_count = int(finite.sum().item())
                    total_count = int(arr.numel())
                    finite_ratio = finite_count / max(total_count, 1)
                    nonfinite_count = total_count - finite_count

                    if finite_count == 0:
                        observed_min = observed_max = observed_mean = float("nan")
                    else:
                        valid = arr[finite]
                        observed_min = float(valid.min().item())
                        observed_max = float(valid.max().item())
                        observed_mean = float(valid.mean().item())

                    bnd = bounds[ch]
                    lower = bnd["lower"]
                    upper = bnd["upper"]
                    below = np.isfinite(observed_min) and observed_min < lower
                    above = np.isfinite(observed_max) and observed_max > upper
                    nonfinite = nonfinite_count > 0

                    if below or above or nonfinite:
                        labels = []
                        if below:
                            labels.append("below_min")
                        if above:
                            labels.append("above_max")
                        if nonfinite:
                            labels.append("nonfinite")
                        violation = ";".join(labels)
                        below_by = lower - observed_min if below else 0.0
                        above_by = observed_max - upper if above else 0.0

                        writer.writerow({
                            "timestamp": timestamp,
                            "source": "cma",
                            "channel": ch,
                            "batch_idx": batch_idx,
                            "sample_in_batch": b,
                            "global_sample_idx": global_idx + b,
                            "observed_min": observed_min,
                            "observed_max": observed_max,
                            "observed_mean": observed_mean,
                            "allowed_lower": lower,
                            "allowed_upper": upper,
                            "csv_min": bnd["csv_min"],
                            "csv_max": bnd["csv_max"],
                            "p01": bnd["p01"],
                            "p99": bnd["p99"],
                            "finite_ratio": finite_ratio,
                            "nonfinite_count": nonfinite_count,
                            "violation": violation,
                            "below_by": below_by,
                            "above_by": above_by,
                        })
                        rows_written += 1
                        bad_timestamps.add(timestamp)
                        bad_counter[ch] += 1
                        for label in labels:
                            violation_counter[label] += 1

                scanned += 1
                if args.max_samples > 0 and scanned >= args.max_samples:
                    break

            global_idx += x_norm.shape[0]
            if args.max_samples > 0 and scanned >= args.max_samples:
                break

    print("\n=== CMA data-quality scan summary ===")
    print(f"Scanned samples: {scanned}")
    print(f"Violation rows: {rows_written}")
    print(f"Bad timestamps: {len(bad_timestamps)}")
    print(f"Log CSV: {log_path}")
    if violation_counter:
        print(f"Violation types: {dict(violation_counter)}")
    if bad_counter:
        print("Top bad channels:")
        for ch, count in bad_counter.most_common(20):
            print(f"  {ch}: {count}")
        print("First bad timestamps:")
        for ts in sorted(bad_timestamps)[:20]:
            print(f"  {ts}")
    else:
        print("No out-of-range samples found with current bounds.")


if __name__ == "__main__":
    main()
