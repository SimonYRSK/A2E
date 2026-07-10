"""
Average and plot A2EC226 per-lead metrics from CSV files.

For each mode under::

    {metrics_root}/{mode}/YYYYMMDD.csv

this script averages RMSE/ACC by ``lead_hours`` across all available init dates.
It then plots each source pair on one figure:

    gfs  vs gfs_naive
    hres vs hres_naive
    cma  vs cma_naive

Each pair figure contains two panels: RMSE and ACC. Averaged per-mode CSV files
and a combined summary CSV are also written.

Usage::

    python /home/ximutian/A2E/plotc226.py

or::

    python /home/ximutian/A2E/plotc226.py \
      --metrics-root /cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/A2E/inference_results/A2Ec226/metrics \
      --output-dir /cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/A2E/inference_results/A2Ec226/plots_c226
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_METRICS_ROOT = Path(
    "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/"
    "A2E/inference_results/A2Ec226/metrics"
)
DEFAULT_OUTPUT_DIR = DEFAULT_METRICS_ROOT.parent / "plots_c226"

PAIRS = [
    ("gfs", "gfs_naive", "GFS"),
    ("hres", "hres_naive", "HRES"),
    ("cma", "cma_naive", "CMA"),
]

COLORS = {
    "gfs": "tab:blue",
    "gfs_naive": "tab:orange",
    "hres": "tab:blue",
    "hres_naive": "tab:orange",
    "cma": "tab:blue",
    "cma_naive": "tab:orange",
}
MARKERS = {
    "gfs": "o",
    "gfs_naive": "s",
    "hres": "o",
    "hres_naive": "s",
    "cma": "o",
    "cma_naive": "s",
}


def mode_label(mode: str) -> str:
    if mode.endswith("_naive"):
        return f"{mode[:-6].upper()} naive"
    return mode.upper()


def read_mode_csvs(metrics_root: Path, mode: str, start_date=None, end_date=None):
    mode_dir = metrics_root / mode
    if not mode_dir.is_dir():
        print(f"  {mode}: missing directory {mode_dir}")
        return pd.DataFrame()

    frames = []
    for csv_path in sorted(mode_dir.glob("*.csv")):
        date_str = csv_path.stem
        if start_date and date_str < start_date:
            continue
        if end_date and date_str > end_date:
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  {mode}: skip {csv_path.name}: {e}")
            continue
        if df.empty:
            continue
        df["source_file"] = csv_path.name
        df["date"] = date_str
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def average_mode_metrics(metrics_root: Path, mode: str, expected_steps: int, start_date=None, end_date=None):
    df = read_mode_csvs(metrics_root, mode, start_date=start_date, end_date=end_date)
    if df.empty:
        return None

    required = {"lead_hours", "rmse", "acc", "date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{mode} CSVs missing columns: {sorted(missing)}")

    # Keep numeric rows only. ACC may be blank/N/A in some runs.
    df["lead_hours"] = pd.to_numeric(df["lead_hours"], errors="coerce")
    df["rmse"] = pd.to_numeric(df["rmse"], errors="coerce")
    df["acc"] = pd.to_numeric(df["acc"], errors="coerce")
    df = df.dropna(subset=["lead_hours", "rmse"])
    df["lead_hours"] = df["lead_hours"].astype(int)

    grouped = (
        df.groupby("lead_hours", as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            acc_mean=("acc", "mean"),
            acc_std=("acc", "std"),
            n_dates=("date", "nunique"),
        )
        .sort_values("lead_hours")
    )
    grouped.insert(0, "mode", mode)
    grouped.insert(2, "step", grouped["lead_hours"] // 6)

    n_files = df["date"].nunique()
    n_leads = grouped["lead_hours"].nunique()
    print(f"  {mode}: {n_files} dates, {n_leads} leads")
    if expected_steps and n_leads != expected_steps:
        print(f"    WARNING: expected {expected_steps} leads, got {n_leads}")

    return grouped


def plot_pair(pair_name: str, mode_a: str, mode_b: str, averages: dict, output_dir: Path):
    data_a = averages.get(mode_a)
    data_b = averages.get(mode_b)
    if data_a is None and data_b is None:
        print(f"  {pair_name}: no data, skip plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    plotted_labels = []

    for mode, df in [(mode_a, data_a), (mode_b, data_b)]:
        if df is None:
            continue
        label = mode_label(mode)
        plotted_labels.append(f"{label}: {int(df['n_dates'].max())}d")
        x = df["lead_hours"].to_numpy()
        axes[0].plot(
            x,
            df["rmse_mean"].to_numpy(),
            label=label,
            color=COLORS.get(mode),
            marker=MARKERS.get(mode, "o"),
            linewidth=2,
            markersize=4,
        )
        axes[1].plot(
            x,
            df["acc_mean"].to_numpy(),
            label=label,
            color=COLORS.get(mode),
            marker=MARKERS.get(mode, "o"),
            linewidth=2,
            markersize=4,
        )

    axes[0].set_title("Z500 RMSE")
    axes[0].set_ylabel("RMSE")
    axes[1].set_title("Z500 ACC")
    axes[1].set_ylabel("ACC")
    for ax in axes:
        ax.set_xlabel("Lead hours")
        ax.grid(True, alpha=0.3)
        ax.legend()

    title_suffix = ", ".join(plotted_labels)
    fig.suptitle(f"{pair_name} C226 averaged metrics ({title_suffix})")
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = output_dir / f"{pair_name.lower()}_c226_metrics.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved {out_path}")


def write_outputs(output_dir: Path, averages: dict):
    avg_dir = output_dir / "averages"
    avg_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for mode, df in sorted(averages.items()):
        if df is None:
            continue
        out_csv = avg_dir / f"{mode}_mean.csv"
        df.to_csv(out_csv, index=False)
        frames.append(df)
        print(f"  Saved {out_csv}")

    if frames:
        summary = pd.concat(frames, ignore_index=True)
        summary_path = output_dir / "c226_mean_metrics_all_modes.csv"
        summary.to_csv(summary_path, index=False)
        print(f"  Saved {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Average and plot A2EC226 metrics CSVs.")
    parser.add_argument("--metrics-root", type=Path, default=DEFAULT_METRICS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["gfs", "gfs_naive", "hres", "hres_naive", "cma", "cma_naive"],
        help="Modes to average. Defaults to all three source/naive pairs.",
    )
    parser.add_argument("--expected-steps", type=int, default=40)
    parser.add_argument("--start-date", type=str, default=None, help="Optional YYYYMMDD lower bound.")
    parser.add_argument("--end-date", type=str, default=None, help="Optional YYYYMMDD upper bound.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Metrics root: {args.metrics_root}")
    print(f"Output dir:   {args.output_dir}")
    if args.start_date or args.end_date:
        print(f"Date filter:  {args.start_date or '-inf'} .. {args.end_date or '+inf'}")
    print()

    averages = {}
    print("Averaging modes:")
    for mode in args.modes:
        averages[mode] = average_mode_metrics(
            args.metrics_root,
            mode,
            expected_steps=args.expected_steps,
            start_date=args.start_date,
            end_date=args.end_date,
        )

    print()
    write_outputs(args.output_dir, averages)

    print("\nPlotting pairs:")
    for mode_a, mode_b, pair_name in PAIRS:
        if mode_a not in args.modes and mode_b not in args.modes:
            continue
        plot_pair(pair_name, mode_a, mode_b, averages, args.output_dir)

    print(f"\nDone. Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
