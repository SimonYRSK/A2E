import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data.pairset import SOURCE_REGISTRY
from models.swinUNET_res import A2E


def parse_int_list(v: str) -> list[int]:
    return [int(x.strip()) for x in str(v).split(",") if x.strip()]


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


def bool_arg(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def run_forward(model, x, times, domains):
    out = model(x, times=times, domains=domains)
    return out[0] if isinstance(out, tuple) else out


def measure_latency(model, x, times, domains, warmup: int, iters: int, device: torch.device):
    with torch.no_grad():
        for _ in range(max(warmup, 0)):
            _ = run_forward(model, x, times, domains)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        latencies_ms = []
        for _ in range(max(iters, 1)):
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = run_forward(model, x, times, domains)
                end.record()
                torch.cuda.synchronize(device)
                latencies_ms.append(float(start.elapsed_time(end)))
            else:
                t0 = time.perf_counter()
                _ = run_forward(model, x, times, domains)
                latencies_ms.append(float((time.perf_counter() - t0) * 1000.0))

    arr = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "latency_ms_mean": float(arr.mean()),
        "latency_ms_std": float(arr.std()),
        "latency_ms_p50": float(np.percentile(arr, 50)),
        "latency_ms_p90": float(np.percentile(arr, 90)),
        "latency_ms_min": float(arr.min()),
        "latency_ms_max": float(arr.max()),
        "latency_iters": int(max(iters, 1)),
        "latency_warmup": int(max(warmup, 0)),
    }


def estimate_flops(model, x, times, domains, device: torch.device):
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.no_grad():
        with torch.profiler.profile(activities=activities, with_flops=True, record_shapes=False) as prof:
            _ = run_forward(model, x, times, domains)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

    total_flops = 0
    supported_events = 0
    for evt in prof.key_averages():
        flops = getattr(evt, "flops", 0) or 0
        if flops > 0:
            total_flops += int(flops)
            supported_events += 1

    return {
        "flops": int(total_flops),
        "gflops": float(total_flops / 1e9),
        "macs": int(total_flops // 2),
        "gmacs": float(total_flops / 2e9),
        "flop_supported_event_count": int(supported_events),
        "flops_note": "Estimated by torch.profiler(with_flops=True); unsupported ops may be omitted.",
    }


def write_csv(path: Path, row: dict):
    fieldnames = [
        "experiment",
        "params",
        "params_m",
        "flops",
        "gflops",
        "macs",
        "gmacs",
        "latency_ms_mean",
        "latency_ms_std",
        "latency_ms_p50",
        "latency_ms_p90",
        "latency_ms_min",
        "latency_ms_max",
        "peak_cuda_memory_mb",
        "device",
        "input_shape",
        "latency_iters",
        "latency_warmup",
        "flops_note",
        "flops_error",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def main():
    parser = argparse.ArgumentParser(description="Profile A2E params, FLOPs/MACs, and forward latency")
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--compute_flops", type=bool_arg, default=True)
    parser.add_argument("--time", default="2025-01-01 00:00:00")
    parser.add_argument("--source", default="gfs")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    model = build_model_from_config(config, device)
    params = sum(p.numel() for p in model.parameters())

    model_cfg = config.get("model", {})
    in_chans = int(model_cfg.get("in_chans", 70))
    img_size = tuple(model_cfg.get("img_size", [721, 1440]))
    x = torch.zeros((args.batch_size, in_chans, int(img_size[0]), int(img_size[1])), device=device, dtype=torch.float32)
    times = np.array([pd.Timestamp(args.time)] * args.batch_size)
    source_idx = SOURCE_REGISTRY.get(str(args.source).lower(), 0)
    domains = torch.full((args.batch_size,), int(source_idx), dtype=torch.long, device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    latency = measure_latency(model, x, times, domains, args.warmup, args.iters, device)

    if device.type == "cuda":
        peak_mem_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
    else:
        peak_mem_mb = float("nan")

    flops = {
        "flops": None,
        "gflops": None,
        "macs": None,
        "gmacs": None,
        "flop_supported_event_count": 0,
        "flops_note": "FLOP profiling disabled.",
        "flops_error": "",
    }
    if args.compute_flops:
        try:
            flops.update(estimate_flops(model, x, times, domains, device))
            flops["flops_error"] = ""
        except Exception as e:
            flops["flops_note"] = "FLOP profiling failed; params and latency are still valid."
            flops["flops_error"] = repr(e)

    profile = {
        "experiment": args.exp_name,
        "params": int(params),
        "params_m": float(params / 1e6),
        "device": str(device),
        "input_shape": list(x.shape),
        "model": model_cfg,
        "profile_scope": "A2E forward only; FuXi rollout is not included.",
        "peak_cuda_memory_mb": peak_mem_mb,
        **latency,
        **flops,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "model_profile.json"
    csv_path = output_dir / "model_profile.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)
    write_csv(csv_path, profile)

    print(f"Saved A2E model profile: {json_path}")
    print(f"Saved A2E model profile CSV: {csv_path}")
    print(
        f"params={profile['params_m']:.2f}M, "
        f"gflops={profile.get('gflops')}, gmacs={profile.get('gmacs')}, "
        f"latency_ms_mean={profile['latency_ms_mean']:.3f}"
    )


if __name__ == "__main__":
    main()
