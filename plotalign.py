"""
Plot per-source ACC/RMSE curves: for each source (GFS, HRES, etc.), one figure
containing whatever curves are available:

  1. ERA5 (ref)      — Naive ERA5 (ERA5 → FuXi baseline)
  2. Naive {SOURCE}   — source directly → FuXi (no A2E)
  3. A2E ({SOURCE})   — source → A2E → FuXi

Each curve is independently averaged across its own available dates.
Missing curves are simply omitted from the figure.
A source with no data at all is skipped.

Data layout expected::

    {align_root}/
        {source}/            ← A2E results:  {date}_{suffix}.txt

    {naive_root}/
        {source}_naive/      ← Naive results: {date}_{suffix}.txt
        era5_naive/          ← Naive ERA5 results: {date}_{suffix}.txt

Usage::

    python A2E/plotalign.py \
        --align_root /path/to/inference_results/A2E_0523 \
        --naive_root /path/to/inference_results \
        --sources gfs hres \
        --output_dir /path/to/plots
"""

import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# A2E / Naive txt parser (inference_align.py output)
# ---------------------------------------------------------------------------
def parse_a2e_txt(txt_path):
    """Parse ``{date}_{suffix}.txt`` produced by inference_align.py.

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


def _extract_date_from_filename(fname, suffix):
    """Filename pattern: ``{date}_{suffix}.txt`` → date (8-digit string)."""
    basename = os.path.basename(fname)
    expected_suffix = f"_{suffix}.txt"
    if not basename.endswith(expected_suffix):
        return None
    date_str = basename[:-len(expected_suffix)]
    if len(date_str) == 8 and date_str.isdigit():
        return date_str
    return None


def load_per_date_from_dir(dir_path, suffix="z500"):
    """Load all per-date ``*_{suffix}.txt`` files from a directory.

    Returns ``{date_str: {steps, rmse, acc}}``.
    """
    if not os.path.isdir(dir_path):
        return {}
    pattern = os.path.join(dir_path, f"*_{suffix}.txt")
    result = {}
    for f in glob.glob(pattern):
        date_str = _extract_date_from_filename(f, suffix)
        if date_str is None:
            continue
        try:
            data = parse_a2e_txt(f)
            if len(data["steps"]) > 0:
                result[date_str] = data
        except Exception as e:
            print(f"  Skip {f}: {e}")
    return result


# ---------------------------------------------------------------------------
# Independent averaging (each curve averaged across its own dates)
# ---------------------------------------------------------------------------
def average_across_dates(per_date):
    """Average metrics independently across all available dates.

    Returns ``{steps, rmse, acc, n_dates}`` or **None** if empty.
    """
    if not per_date:
        return None
    dates = sorted(per_date.keys())
    ref_date = dates[0]
    n_steps = len(per_date[ref_date]["steps"])

    # Keep only dates with matching step count
    valid = []
    for d in dates:
        if len(per_date[d]["steps"]) == n_steps:
            valid.append(d)
    if not valid:
        return None

    rmse_arr = np.stack([np.array(per_date[d]["rmse"], dtype=np.float64)
                          for d in valid])
    acc_arr = np.stack([np.array(per_date[d]["acc"], dtype=np.float64)
                         for d in valid])

    return {
        "steps": np.array(per_date[ref_date]["steps"]),
        "rmse": rmse_arr.mean(axis=0),
        "acc": acc_arr.mean(axis=0),
        "n_dates": len(valid),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _truncate(steps, arr, num_steps=None, interval=1):
    """Slice array to length / interval. Returns (steps, arr)."""
    total = len(steps)
    n = total if num_steps is None else min(int(num_steps), total)
    interval = max(1, int(interval))
    sel = np.arange(0, n, interval)
    return np.array(steps)[:n][sel], np.array(arr)[:n][sel]


# ---------------------------------------------------------------------------
# Plotting — one figure per source
# ---------------------------------------------------------------------------
# (dict_key, label_template, color, marker)
CURVE_DEFS = [
    ("era5",  "ERA5 (ref)",        "red",  "o"),
    ("naive", "Naive {src}",       "gray", "s"),
    ("a2e",   "A2E ({src})",       "blue", "d"),
]


def plot_source_figure(source_name, curves, output_dir, metric,
                       num_steps=None, interval=1):
    """Plot available curves for one source on one figure.

    *curves* is a dict like ``{"era5": avg, "naive": avg, "a2e": avg}``
    where each value may be None (curve skipped).

    Saves ``{metric}_curve_{source_name}.png``.
    """
    mk = "rmse" if metric == "rmse" else "acc"
    ylabel = "RMSE" if metric == "rmse" else "ACC"

    plt.figure(figsize=(10, 5))

    any_plotted = False
    date_parts = []

    for key, label_tpl, color, marker in CURVE_DEFS:
        avg = curves.get(key)
        if avg is None:
            continue
        sel_steps, sel_vals = _truncate(avg["steps"], avg[mk],
                                         num_steps=num_steps, interval=interval)
        label = label_tpl.format(src=source_name.upper())
        plt.plot(sel_steps, sel_vals, label=label,
                 marker=marker, color=color, linewidth=2)
        any_plotted = True
        date_parts.append(f"{label}: {avg['n_dates']}d")

    if not any_plotted:
        plt.close()
        return

    plt.xlabel("Forecast Step")
    plt.ylabel(ylabel)
    plt.title(f"Z500 {ylabel} – {source_name.upper()}  ({', '.join(date_parts)})")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    fname = f"{metric}_curve_{source_name}.png"
    plt.savefig(os.path.join(output_dir, fname))
    plt.close()
    print(f"  Saved {os.path.join(output_dir, fname)}")


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------
def write_summary_txt(output_dir, all_curves, sources):
    """Write a summary table across sources with key-step RMSE / ACC."""
    summary_path = os.path.join(output_dir, "summary_average.txt")
    key_steps = [0, 9, 19, 29, 39]

    with open(summary_path, "w") as f:
        f.write("# Averaged Z500 metrics (each curve averaged independently)\n")
        header = f"{'Curve':>24s}"
        for s in key_steps:
            header += f"  S{s+1:02d}_RMSE  S{s+1:02d}_ACC"
        f.write(header + "\n")

        for src in sources:
            curves = all_curves.get(src, {})
            for key, label_tpl, _, _ in CURVE_DEFS:
                avg = curves.get(key)
                if avg is None:
                    continue
                label = label_tpl.format(src=src.upper())
                n = len(avg["steps"])
                line = f"  {label:>24s}"
                for s in key_steps:
                    if s < n:
                        line += (f"  {avg['rmse'][s]:10.4f}"
                                 f"  {avg['acc'][s]:9.4f}")
                    else:
                        line += f"  {'N/A':>10s}  {'N/A':>9s}"
                line += f"  (n={avg['n_dates']})\n"
                f.write(line)

    print(f"  Summary saved to {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Plot per-source ACC/RMSE curves with ERA5 reference."
    )

    # -- Naive / A2E data roots --
    parser.add_argument(
        "--align_root", type=str,
        default="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/A2E/inference_results/A2E_0603",
        help="A2E results root: {align_root}/{source}/*_{suffix}.txt",
    )
    parser.add_argument(
        "--naive_root", type=str,
        default="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/"
                "A2E/inference_results",
        help="Naive results root: {naive_root}/{source}_naive/*_{suffix}.txt "
             "and {naive_root}/era5_naive/*_{suffix}.txt",
    )
    parser.add_argument(
        "--sources", type=str, nargs="+", default=["gfs", "hres", "cma"],
        help="Source names (subdirs under align_root).",
    )
    parser.add_argument(
        "--suffix", type=str, default="z500",
        help="Filename suffix for per-date txt files (default: z500).",
    )

    # -- Plot controls --
    parser.add_argument("--num_steps", type=int, default=40)
    parser.add_argument("--interval", type=int, default=2)

    # -- Output --
    parser.add_argument(
        "--output_dir", type=str,
        default="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/"
                "A2E/inference_results/A2E_0603/plots",
    )

    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.align_root, "plots")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Sources: {args.sources}")
    print(f"Align root: {args.align_root}")
    print(f"Naive root: {args.naive_root}")
    print()

    # ---- Load ERA5 ref once from {naive_root}/era5_naive/ (unified format) ----
    era5_naive_dir = os.path.join(args.naive_root, "era5_naive")
    era5_per_date = load_per_date_from_dir(era5_naive_dir, args.suffix)
    era5_avg = average_across_dates(era5_per_date)
    if era5_avg:
        print(f"ERA5 ref loaded: {era5_avg['n_dates']} dates "
              f"from {era5_naive_dir}")
    else:
        print(f"ERA5 ref: 0 dates — will be skipped in figures "
              f"(checked {era5_naive_dir})")
    print()

    all_curves = {}

    for src in args.sources:
        print(f"--- [{src.upper()}] ---")

        curves = {"era5": era5_avg}  # shared ERA5 reference

        # Naive {source} from naive_root/{source}_naive/
        naive_dir = os.path.join(args.naive_root, f"{src}_naive")
        naive_per_date = load_per_date_from_dir(naive_dir, args.suffix)
        curves["naive"] = average_across_dates(naive_per_date)
        if curves["naive"]:
            print(f"  Naive {src}: {curves['naive']['n_dates']} dates")
        else:
            print(f"  Naive {src}: 0 dates — skipped")

        # A2E {source} from align_root/{source}/
        a2e_dir = os.path.join(args.align_root, src)
        a2e_per_date = load_per_date_from_dir(a2e_dir, args.suffix)
        curves["a2e"] = average_across_dates(a2e_per_date)
        if curves["a2e"]:
            print(f"  A2E {src}:   {curves['a2e']['n_dates']} dates")
        else:
            print(f"  A2E {src}:   0 dates — skipped")

        # Skip source if nothing at all (besides possibly ERA5 ref)
        has_own_data = curves["naive"] is not None or curves["a2e"] is not None
        if not has_own_data:
            print(f"  [{src}] SKIP: no naive or A2E data for this source.")
            continue

        all_curves[src] = curves

        # Plot
        plot_source_figure(src, curves, output_dir, "acc",
                           num_steps=args.num_steps, interval=args.interval)
        plot_source_figure(src, curves, output_dir, "rmse",
                           num_steps=args.num_steps, interval=args.interval)

        # Quick console summary
        for key, label_tpl, _, _ in CURVE_DEFS:
            avg = curves.get(key)
            if avg is not None:
                try:
                    label = label_tpl.format(src=src.upper())
                    print(f"  [{src}] {label:>16s}  "
                          f"Step1 RMSE={avg['rmse'][0]:.4f}  ACC={avg['acc'][0]:.4f}")
                except Exception:
                    pass

    # ---- Summary ----
    if all_curves:
        write_summary_txt(output_dir, all_curves, args.sources)
        print(f"\nPlots saved to {output_dir}/")
        for src in all_curves:
            print(f"  acc_curve_{src}.png  rmse_curve_{src}.png")
    else:
        print("\nNo sources had enough data to plot — check paths and dates.")


if __name__ == "__main__":
    main()
