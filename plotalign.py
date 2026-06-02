"""
Plot averaged ACC/RMSE curves, comparing A2E experiments and align results.

Supports two result formats:

1. Old model (G2E format) – per-date txt under::

       {metrics_root}/{tag}/{date}/metrics_{tag}_{date}.txt

   Each line::

       Step N: Naive ERA5 RMSE=X.XX, ACC=Y.YY | Naive GFS RMSE=X.XX, ACC=Y.YY | GFS2ERA5 RMSE=X.XX, ACC=Y.YY

2. New A2E align (inference_align.py) – per-source subdirs under::

       {align_root}/{source_name}/*_{suffix}.txt

   Each line (space-separated)::

       step  lead_hours  rmse  acc

Usage::

    # Compare two A2E experiments vs FuXi baseline
    python A2E/plotalign.py \\
        --metrics_root /path/to/metrics \\
        --tags A2E_0520 A2E_0523 \\
        --names "A2E 0520" "A2E 0523" \\
        --dates 20250101 20250115 20250131 \\
        --align_root /path/to/inference_results/A2E_0523 \\
        --output_dir /path/to/plots

    # Just plot align results (no old model comparison)
    python A2E/plotalign.py \\
        --align_root /path/to/inference_results/A2E_0523
"""

import argparse
import os
import re
import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Old-model line parser (G2E format)
# ---------------------------------------------------------------------------
LINE_RE = re.compile(
    r"Step\s+(?P<step>\d+):"
    r"Naive ERA5 RMSE=(?P<rmse_era5>[0-9.]+), ACC=(?P<acc_era5>[0-9.]+) \| "
    r"Naive GFS RMSE=(?P<rmse_naive>[0-9.]+), ACC=(?P<acc_naive>[0-9.]+) \| "
    r"GFS2ERA5 RMSE=(?P<rmse_trans>[0-9.]+), ACC=(?P<acc_trans>[0-9.]+)"
)


def parse_old_metrics_txt(txt_path):
    steps = []
    rmse_era5, rmse_naive, rmse_trans = [], [], []
    acc_era5, acc_naive, acc_trans = [], [], []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            steps.append(int(m.group("step")))
            rmse_era5.append(float(m.group("rmse_era5")))
            rmse_naive.append(float(m.group("rmse_naive")))
            rmse_trans.append(float(m.group("rmse_trans")))
            acc_era5.append(float(m.group("acc_era5")))
            acc_naive.append(float(m.group("acc_naive")))
            acc_trans.append(float(m.group("acc_trans")))
    return {
        "steps": steps,
        "rmse_era5": rmse_era5, "rmse_naive": rmse_naive, "rmse_trans": rmse_trans,
        "acc_era5": acc_era5, "acc_naive": acc_naive, "acc_trans": acc_trans,
    }


def load_old_series(metrics_root, tag, date):
    txt_path = Path(metrics_root) / tag / date / f"metrics_{tag}_{date}.txt"
    if not txt_path.exists():
        raise FileNotFoundError(f"Missing file: {txt_path}")
    return parse_old_metrics_txt(str(txt_path))


def average_old_metrics(metrics_root, tag, dates):
    all_metrics = []
    for date in dates:
        try:
            data = load_old_series(metrics_root, tag, date)
            all_metrics.append(data)
        except Exception as e:
            print(f"  Old format skip {tag} {date}: {e}")
    if not all_metrics:
        raise RuntimeError(f"No valid data for tag={tag}")
    steps = all_metrics[0]["steps"]
    n = len(steps)

    def stack_and_mean(key):
        arr = np.stack([m[key] for m in all_metrics if len(m[key]) == n])
        return arr.mean(axis=0)

    return {
        "steps": steps,
        "rmse_era5": stack_and_mean("rmse_era5"),
        "rmse_naive": stack_and_mean("rmse_naive"),
        "rmse_trans": stack_and_mean("rmse_trans"),
        "acc_era5": stack_and_mean("acc_era5"),
        "acc_naive": stack_and_mean("acc_naive"),
        "acc_trans": stack_and_mean("acc_trans"),
    }


# ---------------------------------------------------------------------------
# New A2E align parser (inference_align.py output)
# ---------------------------------------------------------------------------
def parse_a2e_txt(txt_path):
    """Parse date_z500.txt produced by inference_align.py.

    Format::

        # Init: 20250101 00Z  |  Steps: 40x6h
        #   Step   Lead(h)       RMSE        ACC
             1         6     12.3456     0.9876
    """
    steps, rmses, accs = [], [], []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                steps.append(int(parts[0]))
                rmses.append(float(parts[2]))
                accs.append(float(parts[3]))
    return {"steps": steps, "rmse": rmses, "acc": accs}


def average_new_align_metrics(align_root, suffix="z500"):
    """Average across all per-date files under align_root matching *_<suffix>.txt."""
    pattern = os.path.join(align_root, f"*_{suffix}.txt")
    files = glob.glob(pattern)
    all_metrics = []
    for f in files:
        if "summary" in os.path.basename(f):
            continue
        try:
            data = parse_a2e_txt(f)
            if len(data["steps"]) > 0:
                all_metrics.append(data)
        except Exception as e:
            print(f"  Skip {f}: {e}")
    if not all_metrics:
        raise RuntimeError(f"No valid align data in {align_root}")

    steps = all_metrics[0]["steps"]
    n = len(steps)

    def stack_and_mean(key):
        arr = np.stack([m[key] for m in all_metrics if len(m[key]) == n])
        return arr.mean(axis=0)

    return {
        "steps": steps,
        "rmse_trans": stack_and_mean("rmse"),
        "acc_trans": stack_and_mean("acc"),
    }


def average_new_align_by_source(align_root, sources, suffix="z500"):
    """Average per-source: align_root/{source}/*_{suffix}.txt"""
    results = {}
    for src in sources:
        src_dir = os.path.join(align_root, src)
        if not os.path.isdir(src_dir):
            print(f"  Source dir not found: {src_dir}, skipping")
            continue
        try:
            results[src] = average_new_align_metrics(src_dir, suffix=suffix)
            print(f"  Loaded align {src}: {len(results[src]['steps'])} steps")
        except Exception as e:
            print(f"  Failed to load align {src}: {e}")
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _truncate_series(series, num_steps=None, interval=1):
    total = len(series["steps"])
    n = total if num_steps is None else min(int(num_steps), total)
    interval = max(1, int(interval))
    sel_idx = np.arange(0, n, interval)
    out = {}
    for k, v in series.items():
        if isinstance(v, (list, np.ndarray)):
            out[k] = np.array(v)[:n][sel_idx]
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
MODEL_COLORS = [
    "tab:blue", "tab:orange", "tab:green", "tab:purple",
    "tab:brown", "tab:pink", "tab:olive", "tab:cyan",
]


def plot_curve(
    old_avg_by_tag,
    new_align_by_source,
    output_dir,
    metric="acc",
    num_steps=None,
    interval=1,
    title_suffix="z500",
    plot_naive_gfs=True,
    plot_naive_era5=True,
    display_name_by_tag=None,
):
    if not old_avg_by_tag and not new_align_by_source:
        raise RuntimeError("No data to plot")

    # Reference steps from first available series
    if old_avg_by_tag:
        ref = _truncate_series(
            next(iter(old_avg_by_tag.values())),
            num_steps=num_steps, interval=interval,
        )
    else:
        ref = _truncate_series(
            next(iter(new_align_by_source.values())),
            num_steps=num_steps, interval=interval,
        )
    steps = ref["steps"]

    if display_name_by_tag is None:
        display_name_by_tag = {tag: tag for tag in old_avg_by_tag}

    plt.figure(figsize=(10, 5))

    # Naive baselines (from old format – only if old data present)
    if old_avg_by_tag and metric == "acc":
        if plot_naive_era5 and "acc_era5" in ref:
            plt.plot(steps, ref["acc_era5"], label="Naive ERA5", marker="o", color="red")
        if plot_naive_gfs and "acc_naive" in ref:
            plt.plot(steps, ref["acc_naive"], label="Naive GFS", marker="o", color="gray")
    elif old_avg_by_tag and metric == "rmse":
        if plot_naive_era5 and "rmse_era5" in ref:
            plt.plot(steps, ref["rmse_era5"], label="Naive ERA5", marker="o", color="red")
        if plot_naive_gfs and "rmse_naive" in ref:
            plt.plot(steps, ref["rmse_naive"], label="Naive GFS", marker="o", color="gray")

    y_key_old = "acc_trans" if metric == "acc" else "rmse_trans"
    y_key_new = "acc_trans" if metric == "acc" else "rmse_trans"

    # Old model curves
    for idx, (tag, data) in enumerate(old_avg_by_tag.items()):
        d = _truncate_series(data, num_steps=num_steps, interval=interval)
        c = MODEL_COLORS[idx % len(MODEL_COLORS)]
        label = display_name_by_tag.get(tag, tag)
        plt.plot(d["steps"], d[y_key_old], label=label, marker="x", color=c)

    # New A2E align curves (per source)
    for idx, (source_name, data) in enumerate(new_align_by_source.items()):
        d = _truncate_series(data, num_steps=num_steps, interval=interval)
        c = MODEL_COLORS[(len(old_avg_by_tag) + idx) % len(MODEL_COLORS)]
        label = f"A2E ({source_name})"
        plt.plot(d["steps"], d[y_key_new], label=label, marker="d", color=c, linewidth=2)

    plt.xlabel("Forecast Step")
    if metric == "acc":
        plt.ylabel("ACC")
        plt.title(f"Z500 ACC ({title_suffix})")
        plt.ylim(0.34, 1.00)
    else:
        plt.ylabel("RMSE")
        plt.title(f"Z500 RMSE ({title_suffix})")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    fname = f"{metric}_curve.png"
    plt.savefig(os.path.join(output_dir, fname))
    plt.close()
    print(f"  Saved {os.path.join(output_dir, fname)}")


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------
def write_summary_txt(output_dir, old_avg_by_tag, new_align_by_source, display_name_by_tag=None):
    summary_path = os.path.join(output_dir, "summary_average.txt")
    key_steps = [0, 9, 19, 29, 39]
    if display_name_by_tag is None:
        display_name_by_tag = {}

    with open(summary_path, "w") as f:
        f.write("# Averaged Z500 metrics across all dates\n")
        f.write(f"# {'Experiment':>20s}")
        for s in key_steps:
            f.write(f"  S{s+1:02d}_RMSE  S{s+1:02d}_ACC")
        f.write("\n")

        for tag, data in old_avg_by_tag.items():
            label = display_name_by_tag.get(tag, tag)
            line = f"  {label:>20s}"
            for s in key_steps:
                if s < len(data["rmse_trans"]):
                    line += f"  {data['rmse_trans'][s]:10.4f}  {data['acc_trans'][s]:9.4f}"
                else:
                    line += f"  {'N/A':>10s}  {'N/A':>9s}"
            line += "\n"
            f.write(line)

        for src, data in new_align_by_source.items():
            line = f"  {'A2E_' + src:>20s}"
            for s in key_steps:
                if s < len(data["rmse_trans"]):
                    line += f"  {data['rmse_trans'][s]:10.4f}  {data['acc_trans'][s]:9.4f}"
                else:
                    line += f"  {'N/A':>10s}  {'N/A':>9s}"
            line += "\n"
            f.write(line)

    print(f"  Summary saved to {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Plot averaged ACC/RMSE curves, comparing old models and new align results."
    )
    # -- Old model comparison --
    parser.add_argument(
        "--metrics_root", type=str,
        default="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/infertest/metrics",
        help="旧模型结果根目录: {metrics_root}/{tag}/{date}/metrics_{tag}_{date}.txt",
    )
    parser.add_argument(
        "--tags", type=str, nargs="*",
        default=["3yr_L1+Gradloss_SS"],
        help="旧模型结果要对比的 tag 列表",
    )
    parser.add_argument(
        "--names", type=str, nargs="*", default=None,
        help="旧模型的显示名称列表，必须与 --tags 数量一致",
    )
    parser.add_argument(
        "--dates", type=str, nargs="+",
        default=["20250101", "20250115", "20250131", "20250214", "20250301",
                 "20250315", "20250331", "20250501", "20250515", "20250601"],
    )

    # -- New A2E align --
    parser.add_argument(
        "--align_root", type=str,
        default="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/A2E/inference_results/A2E_0520",
        help="新 A2E align 结果存放目录: {align_root}/{source}/*_{suffix}.txt",
    )
    parser.add_argument(
        "--sources", type=str, nargs="+", default=["gfs", "hres"],
    )
    parser.add_argument(
        "--suffix", type=str, default="z500",
    )

    # -- Plot controls --
    parser.add_argument("--plot_naive_gfs", dest="plot_naive_gfs",
                        action="store_true", default=True)
    parser.add_argument("--no_plot_naive_gfs", dest="plot_naive_gfs",
                        action="store_false")
    parser.add_argument("--plot_naive_era5", dest="plot_naive_era5",
                        action="store_true", default=True)
    parser.add_argument("--no_plot_naive_era5", dest="plot_naive_era5",
                        action="store_false")
    parser.add_argument("--num_steps", type=int, default=40)
    parser.add_argument("--interval", type=int, default=2)
    parser.add_argument("--title_suffix", type=str, default="z500")

    # -- Output --
    parser.add_argument(
        "--output_dir", type=str,
        default="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/A2E/inference_results/plots",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.align_root, "plots")
    os.makedirs(output_dir, exist_ok=True)

    # ---- Load old model metrics ----
    old_avg_by_tag = {}
    if args.tags:
        for tag in args.tags:
            try:
                old_avg_by_tag[tag] = average_old_metrics(
                    args.metrics_root, tag, args.dates,
                )
            except Exception as e:
                print(f"载入旧模型 {tag} 失败: {e}")

    if args.names is not None:
        if len(args.names) != len(args.tags):
            raise ValueError(
                f"--names 数量({len(args.names)})必须与 --tags 数量({len(args.tags)})一致"
            )
        display_name_by_tag = dict(zip(args.tags, args.names))
    else:
        display_name_by_tag = {tag: tag for tag in (args.tags or [])}

    # ---- Load new A2E align results ----
    new_align_by_source = {}
    if args.align_root and os.path.exists(args.align_root):
        new_align_by_source = average_new_align_by_source(
            args.align_root, args.sources, suffix=args.suffix,
        )
        total_steps = (
            next(iter(new_align_by_source.values()))["steps"]
            if new_align_by_source else []
        )
        if total_steps:
            print(f"成功载入新的 A2E align 结果（包含 {len(total_steps)} 个步长的数据）。")
    else:
        print(f"警告：未找到 align_root: {args.align_root}")

    if not old_avg_by_tag and not new_align_by_source:
        print("无任何绘制数据。")
        return

    # ---- Plot ----
    plot_curve(
        old_avg_by_tag, new_align_by_source, output_dir,
        metric="acc", num_steps=args.num_steps, interval=args.interval,
        title_suffix=args.title_suffix,
        plot_naive_gfs=args.plot_naive_gfs,
        plot_naive_era5=args.plot_naive_era5,
        display_name_by_tag=display_name_by_tag,
    )
    plot_curve(
        old_avg_by_tag, new_align_by_source, output_dir,
        metric="rmse", num_steps=args.num_steps, interval=args.interval,
        title_suffix=args.title_suffix,
        plot_naive_gfs=args.plot_naive_gfs,
        plot_naive_era5=args.plot_naive_era5,
        display_name_by_tag=display_name_by_tag,
    )

    # ---- Summary ----
    write_summary_txt(output_dir, old_avg_by_tag, new_align_by_source, display_name_by_tag)

    # Quick console summary (matching G2E style)
    if "3yr_L1+Gradloss_SS" in old_avg_by_tag:
        try:
            val = old_avg_by_tag["3yr_L1+Gradloss_SS"]["rmse_trans"][0]
            print(f"3yr_L1+Gradloss_SS 第1步 RMSE (Z500): {val:.4f}")
        except Exception:
            pass

    if new_align_by_source:
        for src, data in new_align_by_source.items():
            try:
                val = data["rmse_trans"][0]
                print(f"A2E ({src}) 第1步 RMSE (Z500): {val:.4f}")
            except Exception:
                pass

    print(f"绘图已保存到 {output_dir}/(acc/rmse)_curve.png")


if __name__ == "__main__":
    main()
