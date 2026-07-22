#!/usr/bin/env python3
"""
Compare A2E evaluation results across multiple experiments.

The script reads, for each experiment, the two CSVs produced by eval_one.sh / eval_fuxi_rollout.py:

  - fuxi_rollout_metrics_summary.csv
  - a2e_initial_metrics_summary.csv

It produces compact comparison plots and a lightweight Markdown report:

  - Forecast rollout line plots:
      one figure per (metric, variable), subplots by source, curves by experiment.
  - Initial alignment bar plots:
      one figure per initial metric, subplots by source, grouped bars by experiment.
  - Combined CSVs and a report.md for quick paper-table extraction.

Example:

  python A2E_backup/eval/plot_eval_report.py \
      --experiments A2Ec70_gfs_refnorm A2Ec70_ab_wo_fuxi A2Ec70_refnorm_w4em3 \
      --eval_dir eval_tmp \
      --output_dir /tmp/a2e_compare_gfs

If an experiment argument is a directory, it is used directly. Otherwise it is
resolved as {experiments_root}/{experiment}.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_EXPERIMENTS_ROOT = (
    "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/"
    "A2E/Formal/experiments"
)
DEFAULT_VARIABLES = "z500,t2m,t850,ws10,ws850,msl"
DEFAULT_KEY_LEADS = "6,24,72,120,240"
INITIAL_METRICS = [
    ("a2e_l1_loss", "A2E L1 Loss", "lower"),
    ("a2e_psnr", "A2E PSNR", "higher"),
    ("a2e_ssim", "A2E SSIM", "higher"),
]
FORECAST_METRICS = [
    ("rmse", "RMSE", "lower"),
    ("acc", "ACC", "higher"),
]


# Color cycle with enough separation for up to 6 experiments.
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
MARKERS = ["o", "s", "D", "^", "v", "P"]

# Nonlinear display transforms for dense forecast curves. Values are still read
# and reported in the original units; only the plotted y-coordinate is warped.
ACC_KNOTS = np.array([0.0, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0], dtype=float)
# Compress high-ACC values and expand low-ACC values: visually, the interval
# 0.4-0.6 is larger than 0.8-1.0. This makes late-lead degradation clearer
# without changing reported numeric values.
ACC_POS = np.array([0.0, 0.0, 2.0, 3.2, 3.75, 4.05, 4.25], dtype=float)
ACC_TICKS = np.array([0.4, 0.6, 0.8, 0.9, 0.95, 1.0], dtype=float)


def _fmt_tick(v: float) -> str:
    if abs(v) >= 10:
        return f"{v:.0f}"
    if abs(v) >= 1:
        return f"{v:.2g}"
    return f"{v:.2f}".rstrip("0").rstrip(".")


def _acc_forward(values):
    return np.interp(np.asarray(values, dtype=float), ACC_KNOTS, ACC_POS)


def transform_forecast_values(metric: str, values):
    arr = np.asarray(values, dtype=float)
    if metric == "acc":
        return _acc_forward(np.clip(arr, ACC_KNOTS[0], ACC_KNOTS[-1]))
    if metric == "rmse":
        return np.sqrt(np.maximum(arr, 0.0))
    return arr


def configure_forecast_axis(ax, metric: str):
    if metric == "acc":
        ax.set_ylim(_acc_forward([0.4])[0], _acc_forward([1.0])[0])
        ax.set_yticks(_acc_forward(ACC_TICKS))
        ax.set_yticklabels([_fmt_tick(v) for v in ACC_TICKS])
        return
    if metric == "rmse":
        lo, hi = ax.get_ylim()
        raw_lo = max(0.0, lo ** 2)
        raw_hi = max(raw_lo + 1e-12, hi ** 2)
        ticks = np.linspace(np.sqrt(raw_lo), np.sqrt(raw_hi), 6) ** 2
        ax.set_yticks(np.sqrt(np.maximum(ticks, 0.0)))
        ax.set_yticklabels([_fmt_tick(v) for v in ticks])


def configure_initial_axis(ax, metric: str):
    if metric == "a2e_ssim":
        ax.set_ylim(0.8, 1.0)
        ax.set_yticks([0.8, 0.85, 0.9, 0.95, 1.0])
    elif metric == "a2e_psnr":
        ax.set_ylim(20, 50)
        ax.set_yticks([20, 30, 40, 50])


def parse_csv_list(text: str | None) -> list[str]:
    if text is None:
        return []
    return [x.strip() for x in str(text).replace(";", ",").split(",") if x.strip()]


def parse_int_list(text: str | None) -> list[int]:
    out = []
    for item in parse_csv_list(text):
        out.append(int(item))
    return out


def normalize_initial_metric(name: str) -> str:
    key = str(name).strip().lower()
    aliases = {
        "l1": "a2e_l1_loss",
        "l1_loss": "a2e_l1_loss",
        "a2e_l1": "a2e_l1_loss",
        "psnr": "a2e_psnr",
        "ssim": "a2e_ssim",
    }
    key = aliases.get(key, key)
    valid = {m[0] for m in INITIAL_METRICS}
    if key not in valid:
        raise ValueError(f"Unknown initial metric {name!r}; choose from {sorted(valid)} or aliases l1/psnr/ssim")
    return key


def normalize_forecast_metric(name: str) -> str:
    key = str(name).strip().lower()
    valid = {m[0] for m in FORECAST_METRICS}
    if key not in valid:
        raise ValueError(f"Unknown forecast metric {name!r}; choose from {sorted(valid)}")
    return key


def parse_overview_pairs(text: str | None) -> list[tuple[str, list[str]]]:
    """Parse pairs like ``acc psnr; rmse ssim`` or ``acc:psnr,rmse:ssim``.

    Returns jobs in the same format used by build_overview_jobs:
    [(initial_metric, [forecast_metric]), ...].
    """
    if not text:
        return []
    jobs = []
    for part in str(text).split(";"):
        part = part.strip()
        if not part:
            continue
        tokens = part.replace(":", " ").replace(",", " ").split()
        if len(tokens) != 2:
            raise ValueError(
                "--overview_pairs entries must be '<forecast_metric> <initial_metric>', "
                f"got {part!r}. Example: 'acc psnr; rmse ssim'"
            )
        forecast_metric = normalize_forecast_metric(tokens[0])
        initial_metric = normalize_initial_metric(tokens[1])
        jobs.append((initial_metric, [forecast_metric]))
    return jobs


def build_overview_jobs(args) -> list[tuple[str, list[str]]]:
    paired = parse_overview_pairs(args.overview_pairs)
    if paired:
        return paired

    initial_text = args.overview_initial_metrics or args.overview_initial_metric
    initial_metrics = [normalize_initial_metric(x) for x in parse_csv_list(initial_text)]
    forecast_metrics = [normalize_forecast_metric(x) for x in parse_csv_list(args.overview_forecast_metrics)]
    return [(initial_metric, forecast_metrics) for initial_metric in initial_metrics]


def safe_name(text: str) -> str:
    keep = []
    for ch in str(text):
        if ch.isalnum() or ch in "._-":
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "unknown"


def resolve_experiment_dir(experiment: str, experiments_root: Path) -> Path:
    p = Path(experiment)
    if p.exists():
        return p
    return experiments_root / experiment


def subplot_grid(n: int) -> tuple[int, int]:
    if n <= 1:
        return 1, 1
    if n == 2:
        return 1, 2
    if n <= 4:
        return 2, 2
    return math.ceil(n / 3), 3


def get_sources(df: pd.DataFrame, requested: list[str]) -> list[str]:
    if requested:
        return requested
    if "source" not in df:
        return []
    return sorted(str(x).lower() for x in df["source"].dropna().unique())


def get_variables(df: pd.DataFrame, requested: list[str]) -> list[str]:
    if requested:
        return requested
    if "variable" not in df:
        return []
    return sorted(str(x).lower() for x in df["variable"].dropna().unique())


def load_experiment(
    experiment: str,
    label: str,
    experiments_root: Path,
    eval_dir: str,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict]:
    exp_dir = resolve_experiment_dir(experiment, experiments_root)
    metric_dir = exp_dir / eval_dir
    forecast_path = metric_dir / "fuxi_rollout_metrics_summary.csv"
    initial_path = metric_dir / "a2e_initial_metrics_summary.csv"

    info = {
        "experiment_arg": experiment,
        "label": label,
        "experiment_dir": str(exp_dir),
        "eval_dir": str(metric_dir),
        "forecast_path": str(forecast_path),
        "initial_path": str(initial_path),
        "forecast_found": forecast_path.exists(),
        "initial_found": initial_path.exists(),
    }

    forecast_df = None
    initial_df = None

    if forecast_path.exists():
        forecast_df = pd.read_csv(forecast_path)
        forecast_df["exp_label"] = label
        forecast_df["exp_arg"] = experiment
        for col in ["source", "variable"]:
            if col in forecast_df:
                forecast_df[col] = forecast_df[col].astype(str).str.lower()
    else:
        print(f"[WARN] Missing forecast summary: {forecast_path}")

    if initial_path.exists():
        initial_df = pd.read_csv(initial_path)
        initial_df["exp_label"] = label
        initial_df["exp_arg"] = experiment
        for col in ["source", "variable"]:
            if col in initial_df:
                initial_df[col] = initial_df[col].astype(str).str.lower()
    else:
        print(f"[WARN] Missing initial summary: {initial_path}")

    return forecast_df, initial_df, info


def truncate_leads(df: pd.DataFrame, max_lead_hours: int | None, lead_interval: int) -> pd.DataFrame:
    out = df.copy()
    if max_lead_hours is not None and "lead_hours" in out:
        out = out[out["lead_hours"] <= max_lead_hours]
    if lead_interval > 1 and "lead_step" in out:
        out = out[(out["lead_step"].astype(int) - 1) % lead_interval == 0]
    return out


def plot_forecast_curves(
    forecast_df: pd.DataFrame,
    output_dir: Path,
    sources: list[str],
    variables: list[str],
    max_lead_hours: int | None,
    lead_interval: int,
    image_format: str,
    dpi: int,
):
    plot_dir = output_dir / "forecast_curves"
    plot_dir.mkdir(parents=True, exist_ok=True)

    df = truncate_leads(forecast_df, max_lead_hours, lead_interval)
    labels = list(df["exp_label"].drop_duplicates())

    saved = []
    for metric, metric_label, direction in FORECAST_METRICS:
        mean_col = f"{metric}_mean"
        n_col = f"n_{metric}"
        if mean_col not in df:
            continue

        for var in variables:
            sub_var = df[df["variable"] == var]
            if sub_var.empty:
                continue

            active_sources = [s for s in sources if not sub_var[sub_var["source"] == s].empty]
            if not active_sources:
                continue

            nrows, ncols = subplot_grid(len(active_sources))
            fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), squeeze=False)
            fig.suptitle(f"{var.upper()} {metric_label} across experiments", fontsize=14)

            for idx, source in enumerate(active_sources):
                ax = axes[idx // ncols][idx % ncols]
                sub = sub_var[sub_var["source"] == source]

                for exp_idx, label in enumerate(labels):
                    cur = sub[sub["exp_label"] == label].sort_values("lead_hours")
                    if cur.empty:
                        continue
                    x = cur["lead_hours"].to_numpy()
                    y = transform_forecast_values(metric, cur[mean_col].to_numpy(dtype=float))
                    if n_col in cur:
                        n_med = cur[n_col].dropna().median()
                        curve_label = f"{label} (n≈{int(n_med)})" if pd.notna(n_med) else label
                    else:
                        curve_label = label
                    ax.plot(
                        x,
                        y,
                        label=curve_label,
                        color=COLORS[exp_idx % len(COLORS)],
                        linewidth=2.2,
                    )

                ax.set_title(source.upper())
                ax.set_xlabel("Lead time (h)")
                ax.set_ylabel(metric_label)
                configure_forecast_axis(ax, metric)
                ax.grid(True, linestyle="--", alpha=0.35)
                ax.legend(fontsize=8)

            # Hide unused axes.
            for j in range(len(active_sources), nrows * ncols):
                axes[j // ncols][j % ncols].axis("off")

            fig.tight_layout(rect=[0, 0, 1, 0.95])
            fname = f"forecast_{metric}_{safe_name(var)}.{image_format}"
            path = plot_dir / fname
            fig.savefig(path, dpi=dpi)
            plt.close(fig)
            saved.append(path)
            print(f"[plot] saved {path}")

    return saved


def plot_forecast_delta_curves(
    forecast_df: pd.DataFrame,
    output_dir: Path,
    sources: list[str],
    variables: list[str],
    max_lead_hours: int | None,
    lead_interval: int,
    image_format: str,
    dpi: int,
    baseline_label: str | None = None,
):
    """Plot improvement curves against a baseline experiment.

    For RMSE, improvement is baseline_rmse - experiment_rmse.
    For ACC, improvement is experiment_acc - baseline_acc.
    Therefore positive values are always better.
    """
    plot_dir = output_dir / "forecast_delta_curves"
    plot_dir.mkdir(parents=True, exist_ok=True)

    df = truncate_leads(forecast_df, max_lead_hours, lead_interval)
    labels = list(df["exp_label"].drop_duplicates())
    if not labels:
        return []
    baseline = baseline_label or labels[0]
    compare_labels = [x for x in labels if x != baseline]
    if not compare_labels:
        print("[WARN] Delta curves need at least two experiments; skip")
        return []

    saved = []
    for metric, metric_label, _ in FORECAST_METRICS:
        mean_col = f"{metric}_mean"
        if mean_col not in df:
            continue
        delta_label = f"Δ{metric_label} vs {baseline} (positive is better)"

        for var in variables:
            sub_var = df[df["variable"] == var]
            if sub_var.empty:
                continue

            active_sources = [s for s in sources if not sub_var[sub_var["source"] == s].empty]
            if not active_sources:
                continue

            nrows, ncols = subplot_grid(len(active_sources))
            fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), squeeze=False)
            fig.suptitle(f"{var.upper()} {delta_label}", fontsize=14)

            for idx, source in enumerate(active_sources):
                ax = axes[idx // ncols][idx % ncols]
                sub = sub_var[sub_var["source"] == source]
                base = sub[sub["exp_label"] == baseline][["lead_hours", mean_col]].rename(columns={mean_col: "baseline"})
                if base.empty:
                    ax.set_title(f"{source.upper()} (missing baseline)")
                    ax.axis("off")
                    continue

                for exp_idx, label in enumerate(compare_labels):
                    cur = sub[sub["exp_label"] == label][["lead_hours", mean_col]].rename(columns={mean_col: "value"})
                    if cur.empty:
                        continue
                    merged = pd.merge(base, cur, on="lead_hours", how="inner").sort_values("lead_hours")
                    if merged.empty:
                        continue
                    if metric == "rmse":
                        delta = merged["baseline"].to_numpy(dtype=float) - merged["value"].to_numpy(dtype=float)
                    else:
                        delta = merged["value"].to_numpy(dtype=float) - merged["baseline"].to_numpy(dtype=float)
                    ax.plot(
                        merged["lead_hours"].to_numpy(),
                        delta,
                        label=label,
                        color=COLORS[(exp_idx + 1) % len(COLORS)],
                        linewidth=2.2,
                    )

                ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.65)
                ax.set_title(source.upper())
                ax.set_xlabel("Lead time (h)")
                ax.set_ylabel(delta_label)
                ax.grid(True, linestyle="--", alpha=0.35)
                ax.legend(fontsize=8)

            for j in range(len(active_sources), nrows * ncols):
                axes[j // ncols][j % ncols].axis("off")

            fig.tight_layout(rect=[0, 0, 1, 0.95])
            fname = f"forecast_delta_{metric}_{safe_name(var)}_vs_{safe_name(baseline)}.{image_format}"
            path = plot_dir / fname
            fig.savefig(path, dpi=dpi)
            plt.close(fig)
            saved.append(path)
            print(f"[plot] saved {path}")

    return saved


def plot_initial_bars(
    initial_df: pd.DataFrame,
    output_dir: Path,
    sources: list[str],
    variables: list[str],
    image_format: str,
    dpi: int,
):
    plot_dir = output_dir / "initial_bars"
    plot_dir.mkdir(parents=True, exist_ok=True)

    labels = list(initial_df["exp_label"].drop_duplicates())
    saved = []

    for metric, metric_label, direction in INITIAL_METRICS:
        mean_col = f"{metric}_mean"
        n_col = f"n_{metric}"
        if mean_col not in initial_df:
            print(f"[WARN] Initial metric missing column {mean_col}; skip")
            continue

        active_sources = [s for s in sources if not initial_df[initial_df["source"] == s].empty]
        if not active_sources:
            continue

        nrows, ncols = subplot_grid(len(active_sources))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.8 * nrows), squeeze=False)
        fig.suptitle(f"Initial alignment: {metric_label} ({'lower is better' if direction == 'lower' else 'higher is better'})", fontsize=14)

        x = np.arange(len(variables))
        group_width = 0.82
        bar_width = group_width / max(1, len(labels))
        annotate_n = len(labels) <= 3 and len(variables) <= 8 and n_col in initial_df

        for idx, source in enumerate(active_sources):
            ax = axes[idx // ncols][idx % ncols]
            sub_src = initial_df[initial_df["source"] == source]

            for exp_idx, label in enumerate(labels):
                vals = []
                ns = []
                for var in variables:
                    cur = sub_src[(sub_src["variable"] == var) & (sub_src["exp_label"] == label)]
                    if cur.empty:
                        vals.append(np.nan)
                        ns.append(np.nan)
                    else:
                        vals.append(float(cur[mean_col].iloc[0]))
                        ns.append(float(cur[n_col].iloc[0]) if n_col in cur else np.nan)

                offset = (exp_idx - (len(labels) - 1) / 2) * bar_width
                bars = ax.bar(
                    x + offset,
                    vals,
                    width=bar_width * 0.92,
                    label=label,
                    color=COLORS[exp_idx % len(COLORS)],
                    alpha=0.9,
                )

                if annotate_n:
                    for bar, n_val in zip(bars, ns):
                        if np.isfinite(bar.get_height()) and np.isfinite(n_val):
                            ax.text(
                                bar.get_x() + bar.get_width() / 2,
                                bar.get_height(),
                                f"n={int(n_val)}",
                                ha="center",
                                va="bottom",
                                rotation=90,
                                fontsize=6,
                            )

            ax.set_title(source.upper())
            ax.set_xticks(x)
            ax.set_xticklabels([v.upper() for v in variables], rotation=35, ha="right")
            ax.set_ylabel(metric_label)
            configure_initial_axis(ax, metric)
            ax.grid(True, axis="y", linestyle="--", alpha=0.3)
            ax.legend(fontsize=8)

        for j in range(len(active_sources), nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fname = f"initial_{safe_name(metric)}.{image_format}"
        path = plot_dir / fname
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        saved.append(path)
        print(f"[plot] saved {path}")

    return saved


def plot_paper_overview_by_source(
    forecast_df: pd.DataFrame,
    initial_df: pd.DataFrame,
    output_dir: Path,
    sources: list[str],
    variables: list[str],
    initial_metric: str,
    forecast_metrics: list[str],
    max_lead_hours: int | None,
    lead_interval: int,
    image_format: str,
    dpi: int,
):
    """Create compact paper-style overview figures.

    For each source and forecast metric, one figure is produced:
      panel a: grouped bars over variables for the selected initial metric;
      panel b: small-multiple lead-time curves over variables.

    This follows the economical layout of multi-panel ML paper figures: a small
    number of dense figures instead of one figure for every source/variable pair.
    """
    if forecast_df.empty or initial_df.empty:
        return []

    plot_dir = output_dir / "paper_overview"
    plot_dir.mkdir(parents=True, exist_ok=True)

    labels = list(pd.concat([forecast_df["exp_label"], initial_df["exp_label"]]).drop_duplicates())
    init_mean_col = f"{initial_metric}_mean"
    init_n_col = f"n_{initial_metric}"
    if init_mean_col not in initial_df:
        print(f"[WARN] Overview initial metric missing column {init_mean_col}; skip overview")
        return []

    metric_lookup = {m: (label, direction) for m, label, direction in FORECAST_METRICS}
    forecast_df = truncate_leads(forecast_df, max_lead_hours, lead_interval)
    saved = []

    for source in sources:
        init_src = initial_df[initial_df["source"] == source]
        fc_src = forecast_df[forecast_df["source"] == source]
        if init_src.empty and fc_src.empty:
            continue

        active_vars = [v for v in variables if (not init_src[init_src["variable"] == v].empty) or (not fc_src[fc_src["variable"] == v].empty)]
        if not active_vars:
            continue

        for forecast_metric in forecast_metrics:
            if forecast_metric not in metric_lookup:
                print(f"[WARN] Unknown overview forecast metric {forecast_metric!r}; skip")
                continue
            fc_label, _ = metric_lookup[forecast_metric]
            fc_mean_col = f"{forecast_metric}_mean"
            if fc_mean_col not in fc_src:
                continue

            n_var = len(active_vars)
            fig = plt.figure(figsize=(max(11.0, 2.35 * n_var), 7.0), constrained_layout=False)
            gs = fig.add_gridspec(2, n_var, height_ratios=[1.0, 1.35], hspace=0.55, wspace=0.28)

            # Panel a: one wide grouped bar chart spanning all columns.
            ax_bar = fig.add_subplot(gs[0, :])
            x = np.arange(n_var)
            group_width = 0.82
            bar_width = group_width / max(1, len(labels))
            for exp_idx, label in enumerate(labels):
                vals = []
                ns = []
                for var in active_vars:
                    cur = init_src[(init_src["variable"] == var) & (init_src["exp_label"] == label)]
                    if cur.empty:
                        vals.append(np.nan)
                        ns.append(np.nan)
                    else:
                        vals.append(float(cur[init_mean_col].iloc[0]))
                        ns.append(float(cur[init_n_col].iloc[0]) if init_n_col in cur else np.nan)
                offset = (exp_idx - (len(labels) - 1) / 2) * bar_width
                ax_bar.bar(
                    x + offset,
                    vals,
                    width=bar_width * 0.92,
                    label=label,
                    color=COLORS[exp_idx % len(COLORS)],
                    alpha=0.9,
                )

            init_label = initial_metric.replace("a2e_", "A2E ").replace("_", " ").title()
            ax_bar.set_title(f"Initial alignment on {source.upper()}: {init_label}")
            ax_bar.set_xticks(x)
            ax_bar.set_xticklabels([v.upper() for v in active_vars])
            ax_bar.set_ylabel(init_label)
            configure_initial_axis(ax_bar, initial_metric)
            ax_bar.grid(True, axis="y", linestyle="--", alpha=0.35)
            ax_bar.legend(ncol=min(3, max(1, len(labels))), fontsize=9, loc="best")
            ax_bar.text(-0.06, 1.08, "a", transform=ax_bar.transAxes, fontsize=18, fontweight="bold", va="top")

            # Panel b: one small lead-time curve per variable.
            curve_axes = []
            for var_idx, var in enumerate(active_vars):
                ax = fig.add_subplot(gs[1, var_idx])
                curve_axes.append(ax)
                sub_var = fc_src[fc_src["variable"] == var]
                for exp_idx, label in enumerate(labels):
                    cur = sub_var[sub_var["exp_label"] == label].sort_values("lead_hours")
                    if cur.empty:
                        continue
                    x_lead = cur["lead_hours"].to_numpy()
                    y = transform_forecast_values(forecast_metric, cur[fc_mean_col].to_numpy(dtype=float))
                    ax.plot(
                        x_lead,
                        y,
                        label=label,
                        color=COLORS[exp_idx % len(COLORS)],
                        linewidth=1.9,
                    )
                ax.set_title(var.upper())
                ax.set_xlabel("Lead (h)")
                if var_idx == 0:
                    ax.set_ylabel(fc_label)
                    ax.text(-0.35, 1.12, "b", transform=ax.transAxes, fontsize=18, fontweight="bold", va="top")
                configure_forecast_axis(ax, forecast_metric)
                ax.grid(True, linestyle="--", alpha=0.35)

            # One shared legend for panel b when it does not duplicate panel a too much.
            handles, legend_labels = curve_axes[-1].get_legend_handles_labels() if curve_axes else ([], [])
            if handles:
                fig.legend(handles, legend_labels, loc="lower center", ncol=min(len(labels), 6), fontsize=9, frameon=False)
                bottom = 0.12
            else:
                bottom = 0.06

            fig.suptitle(f"A2E experiment comparison — {source.upper()} ({fc_label})", fontsize=15, y=0.98)
            fig.subplots_adjust(left=0.07, right=0.99, top=0.89, bottom=bottom)
            fname = f"overview_{safe_name(source)}_{safe_name(forecast_metric)}_{safe_name(initial_metric)}.{image_format}"
            path = plot_dir / fname
            fig.savefig(path, dpi=dpi)
            plt.close(fig)
            saved.append(path)
            print(f"[plot] saved {path}")

    return saved


def nearest_lead_rows(df: pd.DataFrame, key_leads: Iterable[int]) -> pd.DataFrame:
    rows = []
    if df.empty or "lead_hours" not in df:
        return pd.DataFrame()
    group_cols = ["exp_label", "source", "variable"]
    for key, group in df.groupby(group_cols, dropna=False):
        group = group.copy()
        leads = group["lead_hours"].astype(int).to_numpy()
        for target in key_leads:
            if len(leads) == 0:
                continue
            idx = int(np.argmin(np.abs(leads - target)))
            row = group.iloc[idx].to_dict()
            row["requested_lead_hours"] = target
            row["nearest_lead_hours"] = int(row.get("lead_hours"))
            rows.append(row)
    return pd.DataFrame(rows)


def build_report(
    output_dir: Path,
    infos: list[dict],
    forecast_df: pd.DataFrame | None,
    initial_df: pd.DataFrame | None,
    key_leads: list[int],
    forecast_plots: list[Path],
    initial_plots: list[Path],
):
    report_path = output_dir / "report.md"
    lines = []
    lines.append("# A2E Evaluation Comparison Report")
    lines.append("")
    lines.append("## Experiments")
    lines.append("")
    lines.append("| Label | Experiment argument | Eval dir | Forecast CSV | Initial CSV |")
    lines.append("|---|---|---|---|---|")
    for info in infos:
        lines.append(
            f"| {info['label']} | `{info['experiment_arg']}` | `{info['eval_dir']}` | "
            f"{'yes' if info['forecast_found'] else 'no'} | {'yes' if info['initial_found'] else 'no'} |"
        )

    if forecast_df is not None and not forecast_df.empty:
        lines.append("")
        lines.append("## Forecast rollout key-lead summary")
        lines.append("")
        key_df = nearest_lead_rows(forecast_df, key_leads)
        if not key_df.empty:
            keep_cols = [
                "exp_label",
                "source",
                "variable",
                "requested_lead_hours",
                "nearest_lead_hours",
                "rmse_mean",
                "n_rmse",
                "acc_mean",
                "n_acc",
            ]
            keep_cols = [c for c in keep_cols if c in key_df]
            key_out = key_df[keep_cols].sort_values(["source", "variable", "requested_lead_hours", "exp_label"])
            key_out.to_csv(output_dir / "forecast_key_leads.csv", index=False)
            lines.append("Saved full table: `forecast_key_leads.csv`.")
            lines.append("")
            # Keep Markdown compact: show only the first 40 rows.
            lines.append(key_out.head(40).to_markdown(index=False))
            if len(key_out) > 40:
                lines.append("")
                lines.append(f"... {len(key_out) - 40} more rows omitted from Markdown; see CSV.")

    if initial_df is not None and not initial_df.empty:
        lines.append("")
        lines.append("## Initial alignment metrics")
        lines.append("")
        init_cols = [
            "exp_label",
            "source",
            "variable",
            "a2e_l1_loss_mean",
            "n_a2e_l1_loss",
            "a2e_psnr_mean",
            "n_a2e_psnr",
            "a2e_ssim_mean",
            "n_a2e_ssim",
        ]
        init_cols = [c for c in init_cols if c in initial_df]
        init_out = initial_df[init_cols].sort_values(["source", "variable", "exp_label"])
        init_out.to_csv(output_dir / "initial_metrics_compact.csv", index=False)
        lines.append("Saved compact table: `initial_metrics_compact.csv`.")
        lines.append("")
        lines.append(init_out.head(40).to_markdown(index=False))
        if len(init_out) > 40:
            lines.append("")
            lines.append(f"... {len(init_out) - 40} more rows omitted from Markdown; see CSV.")

    lines.append("")
    lines.append("## Generated plots")
    lines.append("")
    lines.append("### Forecast curves")
    for path in forecast_plots:
        rel = path.relative_to(output_dir)
        lines.append(f"- `{rel}`")
    lines.append("")
    lines.append("### Initial metric bars")
    for path in initial_plots:
        rel = path.relative_to(output_dir)
        lines.append(f"- `{rel}`")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[report] saved {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Compare A2E eval CSVs and generate plots/report.")
    parser.add_argument("--experiments", nargs="+", required=True, help="Experiment names under experiments_root, or direct experiment directories. 1-6 recommended.")
    parser.add_argument("--labels", nargs="*", default=None, help="Optional display labels, same length as --experiments.")
    parser.add_argument("--experiments_root", type=str, default=os.environ.get("OUTPUT_ROOT", DEFAULT_EXPERIMENTS_ROOT))
    parser.add_argument("--eval_dir", type=str, default="eval", help="Subdirectory under each experiment, e.g. eval or eval_tmp.")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to save plots/report. Default: experiments_root/compare_{eval_dir}.")
    parser.add_argument("--sources", type=str, default="", help="Comma-separated sources. Default: all found in CSVs.")
    parser.add_argument("--variables", type=str, default=DEFAULT_VARIABLES, help="Comma-separated variables.")
    parser.add_argument("--max_lead_hours", type=int, default=None, help="Optional maximum lead time for forecast plots.")
    parser.add_argument("--lead_interval", type=int, default=1, help="Plot every N forecast steps.")
    parser.add_argument("--key_leads", type=str, default=DEFAULT_KEY_LEADS, help="Comma-separated lead hours for report tables.")
    parser.add_argument("--delta", action="store_true", help="Also plot delta/improvement curves relative to a baseline experiment. Positive is better.")
    parser.add_argument("--delta_baseline", type=str, default="", help="Baseline label for --delta. Default: first experiment label.")
    parser.add_argument("--overview", action="store_true", help="Also create compact paper-style overview figures: panel a initial bars + panel b forecast curves.")
    parser.add_argument("--overview_initial_metric", type=str, default="a2e_l1_loss", choices=[m[0] for m in INITIAL_METRICS], help="Backward-compatible single initial metric for overview panel a.")
    parser.add_argument("--overview_initial_metrics", type=str, default="", help="Comma-separated initial metrics for overview panel a, e.g. a2e_l1_loss,a2e_psnr,a2e_ssim. Aliases: l1, psnr, ssim.")
    parser.add_argument("--overview_forecast_metrics", type=str, default="rmse,acc", help="Comma-separated forecast metrics for overview panel b, e.g. rmse or rmse,acc.")
    parser.add_argument("--overview_pairs", type=str, default="", help="Paired overview jobs like 'acc psnr; rmse ssim'. Each pair is '<forecast_metric> <initial_metric>'.")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    if not (1 <= len(args.experiments) <= 6):
        raise ValueError("Please pass 1-6 experiments to keep figures readable.")

    if args.labels is not None and len(args.labels) > 0:
        if len(args.labels) != len(args.experiments):
            raise ValueError("--labels must have the same length as --experiments.")
        labels = args.labels
    else:
        labels = [Path(exp).name.rstrip("/") for exp in args.experiments]

    experiments_root = Path(args.experiments_root)
    output_dir = Path(args.output_dir) if args.output_dir else experiments_root / f"compare_{args.eval_dir}"
    output_dir.mkdir(parents=True, exist_ok=True)

    forecast_frames = []
    initial_frames = []
    infos = []
    for exp, label in zip(args.experiments, labels):
        forecast_df, initial_df, info = load_experiment(exp, label, experiments_root, args.eval_dir)
        infos.append(info)
        if forecast_df is not None:
            forecast_frames.append(forecast_df)
        if initial_df is not None:
            initial_frames.append(initial_df)

    forecast_all = pd.concat(forecast_frames, ignore_index=True) if forecast_frames else pd.DataFrame()
    initial_all = pd.concat(initial_frames, ignore_index=True) if initial_frames else pd.DataFrame()

    if not forecast_all.empty:
        forecast_all.to_csv(output_dir / "combined_fuxi_rollout_metrics_summary.csv", index=False)
    if not initial_all.empty:
        initial_all.to_csv(output_dir / "combined_a2e_initial_metrics_summary.csv", index=False)

    sources_requested = [s.lower() for s in parse_csv_list(args.sources)]
    variables_requested = [v.lower() for v in parse_csv_list(args.variables)]

    source_base_df = forecast_all if not forecast_all.empty else initial_all
    variable_base_df = forecast_all if not forecast_all.empty else initial_all
    sources = get_sources(source_base_df, sources_requested)
    variables = get_variables(variable_base_df, variables_requested)

    print(f"Experiments: {labels}")
    print(f"Eval dir: {args.eval_dir}")
    print(f"Sources: {sources}")
    print(f"Variables: {variables}")
    print(f"Output: {output_dir}")

    forecast_plots = []
    if not forecast_all.empty:
        forecast_plots = plot_forecast_curves(
            forecast_all,
            output_dir,
            sources,
            variables,
            args.max_lead_hours,
            max(1, int(args.lead_interval)),
            args.format,
            args.dpi,
        )
    else:
        print("[WARN] No forecast summary data loaded; skip forecast curve plots.")

    initial_plots = []
    if not initial_all.empty:
        initial_plots = plot_initial_bars(
            initial_all,
            output_dir,
            sources,
            variables,
            args.format,
            args.dpi,
        )
    else:
        print("[WARN] No initial summary data loaded; skip initial bar plots.")

    delta_plots = []
    if args.delta and not forecast_all.empty:
        delta_plots = plot_forecast_delta_curves(
            forecast_all,
            output_dir,
            sources,
            variables,
            args.max_lead_hours,
            max(1, int(args.lead_interval)),
            args.format,
            args.dpi,
            args.delta_baseline or None,
        )

    overview_plots = []
    if args.overview:
        if not forecast_all.empty and not initial_all.empty:
            for initial_metric, forecast_metrics in build_overview_jobs(args):
                overview_plots.extend(
                    plot_paper_overview_by_source(
                        forecast_all,
                        initial_all,
                        output_dir,
                        sources,
                        variables,
                        initial_metric,
                        forecast_metrics,
                        args.max_lead_hours,
                        max(1, int(args.lead_interval)),
                        args.format,
                        args.dpi,
                    )
                )
        else:
            print("[WARN] Overview needs both forecast and initial summaries; skip overview plots.")

    build_report(
        output_dir,
        infos,
        forecast_all if not forecast_all.empty else None,
        initial_all if not initial_all.empty else None,
        parse_int_list(args.key_leads),
        forecast_plots + delta_plots + overview_plots,
        initial_plots,
    )


if __name__ == "__main__":
    main()
# python /home/ximutian/A2E/eval/plot_eval_report.py \
#   --experiments A2Ec70_gfs_refnorm A2Ec70_gradw_0p1 \
#   --labels full gradw_0p1 \
#   --eval_dir eval_tmp \
#   --sources gfs \
#   --variables z500,t2m,t850,ws10,ws850,msl \
#   --delta \
#   --delta_baseline full \
#   --overview \
#   --overview_pairs "acc psnr; rmse ssim" \
#   --output_dir /cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/A2E/Formal/experiments/compare_eval_gfs_tmp
