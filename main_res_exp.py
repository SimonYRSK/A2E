import argparse
import json
import os
import random
import warnings
import multiprocessing as mp
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from data.pairset import Any2ERA5Dataset, SOURCE_REGISTRY
from models.swinUNET_res import A2E
from fuxi.fuxi_grad import UTransformer, FuXi
from fuxi_rmse_interface_new import (
    FuXiRMSEInterface,
    DEFAULT_CHANNEL_WEIGHTS,
    FUXI_ERA5_REFERENCE_RMSE,
    TARGET_RMSE_CHANNELS,
)
from trainers.fsdptrain_align_metrics import FSDPUNetAlignMetricsTrainer


try:
    from zarr.errors import ZarrUserWarning
except Exception:
    ZarrUserWarning = UserWarning


try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass


torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


DEFAULT_GFS_PATH = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/data/gfs_2020_2025_c226_0p25_norm.zarr"
DEFAULT_HRES_PATH = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/data/hres_2024_2025_c226_0p25_norm.zarr"
DEFAULT_CMA_PATH = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/data/cma_gfs_2020_2026.c226.norm.zarr"
DEFAULT_ERA5_PATH = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/data/era5.2020_2025_norm.zarr"
DEFAULT_FUXI_DIR = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/fuxi_inference/main/fuxi"
DEFAULT_PROJECT_ROOT = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/A2E"


def configure_warning_filters():
    warnings.filterwarnings(
        "ignore",
        message=r"Both zarr\.json \(Zarr format 3\) and \.zgroup \(Zarr format 2\) metadata objects exist.*",
        category=ZarrUserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Object at .* is not recognized as a component of a Zarr hierarchy\.",
        category=ZarrUserWarning,
    )


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {v!r}")


def parse_int_list(v: str) -> list[int]:
    return [int(x.strip()) for x in str(v).split(",") if x.strip()]


def parse_float_pair(v: str) -> tuple[float, float]:
    xs = [float(x.strip()) for x in str(v).split(",") if x.strip()]
    if len(xs) != 2:
        raise argparse.ArgumentTypeError(f"Expected two comma-separated floats, got {v!r}")
    return xs[0], xs[1]


def parse_sources(v: str | Iterable[str]) -> list[str]:
    if isinstance(v, str):
        parts = v.replace(";", ",").split(",")
    else:
        parts = list(v)
    out = []
    for p in parts:
        p = str(p).strip().lower()
        if not p:
            continue
        if p not in SOURCE_REGISTRY:
            raise ValueError(f"Unknown source {p!r}; choices={sorted(SOURCE_REGISTRY)}")
        out.append(p)
    if not out:
        raise ValueError("At least one source is required")
    return out


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_distributed():
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() and os.name != "nt" else "gloo"
        dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    return device, rank, world_size


def custom_collate(batch):
    x, y, i, times = zip(*batch)
    times = np.array([pd.Timestamp(str(t)) for t in times])
    domains = torch.as_tensor(i, dtype=torch.long)
    return torch.stack(x), torch.stack(y), domains, times


def build_fuxi_model(device: torch.device, fuxi_dir: str) -> FuXi:
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
        step_range=[1],
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


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Parameterized A2E-c70 + FuXi training")

    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--sources", type=str, default="cma")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--era5_path", type=str, default=DEFAULT_ERA5_PATH)
    parser.add_argument("--gfs_path", type=str, default=DEFAULT_GFS_PATH)
    parser.add_argument("--hres_path", type=str, default=DEFAULT_HRES_PATH)
    parser.add_argument("--cma_path", type=str, default=DEFAULT_CMA_PATH)
    parser.add_argument("--fuxi_dir", type=str, default=DEFAULT_FUXI_DIR)
    parser.add_argument("--output_root", type=str, default=os.path.join(DEFAULT_PROJECT_ROOT, "experiments"))
    parser.add_argument("--checkpoint_root", type=str, default=os.path.join(DEFAULT_PROJECT_ROOT, "checkpoints"))
    parser.add_argument("--tensorboard_root", type=str, default="/home/ximutian/tensorboard_logs")
    parser.add_argument("--plot_root", type=str, default=os.path.join(DEFAULT_PROJECT_ROOT, "channelpics"))

    parser.add_argument("--train_start", type=str, default="2022-01-01 00:00:00")
    parser.add_argument("--train_end", type=str, default="2024-12-31 18:00:00")
    parser.add_argument("--val_start", type=str, default="2025-01-01 00:00:00")
    parser.add_argument("--val_end", type=str, default="2025-11-20 18:00:00")
    parser.add_argument("--gfs_train_start", type=str, default=None)
    parser.add_argument("--gfs_train_end", type=str, default=None)
    parser.add_argument("--gfs_val_start", type=str, default=None)
    parser.add_argument("--gfs_val_end", type=str, default=None)
    parser.add_argument("--cma_train_start", type=str, default=None)
    parser.add_argument("--cma_train_end", type=str, default=None)
    parser.add_argument("--cma_val_start", type=str, default=None)
    parser.add_argument("--cma_val_end", type=str, default=None)
    parser.add_argument("--hres_train_start", type=str, default="2024-01-01 00:00:00")
    parser.add_argument("--hres_train_end", type=str, default="2024-12-31 18:00:00")
    parser.add_argument("--hres_val_start", type=str, default=None)
    parser.add_argument("--hres_val_end", type=str, default=None)
    parser.add_argument("--val_sample_per_month", type=int, default=4)
    parser.add_argument("--val_sample_year", type=int, default=2025)
    parser.add_argument("--max_samples_per_year", type=str, default="none")
    parser.add_argument("--sample_seed", type=int, default=43)

    parser.add_argument("--img_size", type=str, default="721,1440")
    parser.add_argument("--patch_size", type=str, default="4,4")
    parser.add_argument("--in_chans", type=int, default=70)
    parser.add_argument("--out_chans", type=int, default=70)
    parser.add_argument("--embed_dim", type=int, default=384)
    parser.add_argument("--num_groups", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_stages", type=int, default=3)
    parser.add_argument("--window_size", type=int, default=9)
    parser.add_argument("--depth", type=str, default="0,0,1")
    parser.add_argument("--res_per_stage", type=str, default="1,1,1")
    parser.add_argument("--channels", type=str, default="384,768,1536")
    parser.add_argument("--using_time_embedding", type=str2bool, default=True)
    parser.add_argument("--using_source_embedding", type=str2bool, default=True)
    parser.add_argument("--using_kl", type=str2bool, default=False)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--use_skip_connections", type=str2bool, default=True)
    parser.add_argument("--use_residual_blocks", type=str2bool, default=True)
    parser.add_argument("--using_dann", type=str2bool, default=False)
    parser.add_argument("--domain_loss_weight", type=float, default=1e-3)
    parser.add_argument("--dann_gamma", type=float, default=10.0)

    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=1)
    parser.add_argument("--base_lr", type=float, default=2e-4)
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=2e-5)
    parser.add_argument("--betas", type=str, default="0.9,0.999")
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--use_amp", type=str2bool, default=False)

    parser.add_argument("--beta", type=float, default=1e-4)
    parser.add_argument("--kl_anneal", type=str2bool, default=False)
    parser.add_argument("--kl_anneal_epochs", type=int, default=7)
    parser.add_argument("--recon_loss_type", type=str, default="l1")
    parser.add_argument("--charbonnier_eps", type=float, default=1e-3)
    parser.add_argument("--use_grad_loss", type=str2bool, default=True)
    parser.add_argument("--grad_loss_weight", type=float, default=0.4)
    parser.add_argument("--l1_reg_weight", type=float, default=0.0)
    parser.add_argument("--l2_reg_weight", type=float, default=0.0)
    parser.add_argument("--fuxi_loss_mode", type=str, default="reference_norm", choices=["manual_weighted", "raw_mean", "reference_norm"])
    parser.add_argument("--channel_rmse_weight", type=float, default=4e-3)
    parser.add_argument("--rmse_every_n_steps", type=int, default=1)
    parser.add_argument("--rmse_samples_per_batch", type=int, default=1)
    parser.add_argument("--fuxi_lead_hours", type=int, default=6)

    return parser


def main():
    args = build_arg_parser().parse_args()
    configure_warning_filters()

    if "RANK" not in os.environ:
        raise RuntimeError("main_res_exp.py must be launched by torchrun")

    device, rank, world_size = setup_distributed()
    is_master = rank == 0

    if is_master:
        print(f"World size={world_size}, rank={rank}, device={device}")

    set_random_seed(args.seed)

    sources = parse_sources(args.sources)
    img_size = tuple(parse_int_list(args.img_size))
    patch_size = tuple(parse_int_list(args.patch_size))
    depth = parse_int_list(args.depth)
    res_per_stage = parse_int_list(args.res_per_stage)
    channels = parse_int_list(args.channels)
    betas = parse_float_pair(args.betas)

    source_paths = {
        "gfs": args.gfs_path,
        "hres": args.hres_path,
        "cma": args.cma_path,
    }
    date_ranges = {
        "gfs": {
            "train_start": args.gfs_train_start or args.train_start,
            "train_end": args.gfs_train_end or args.train_end,
            "val_start": args.gfs_val_start or args.val_start,
            "val_end": args.gfs_val_end or args.val_end,
        },
        "cma": {
            "train_start": args.cma_train_start or args.train_start,
            "train_end": args.cma_train_end or args.train_end,
            "val_start": args.cma_val_start or args.val_start,
            "val_end": args.cma_val_end or args.val_end,
        },
        "hres": {
            "train_start": args.hres_train_start,
            "train_end": args.hres_train_end,
            "val_start": args.hres_val_start or args.val_start,
            "val_end": args.hres_val_end or args.val_end,
        },
    }

    max_samples_per_year = None if str(args.max_samples_per_year).lower() in {"none", "null", "0", ""} else int(args.max_samples_per_year)

    train_sets = []
    val_sets = []
    source_configs = []
    for source_name in sources:
        source_idx = SOURCE_REGISTRY[source_name]
        source_path = source_paths[source_name]
        dates = date_ranges[source_name]
        source_configs.append((source_name, source_path, source_idx))

        train_sets.append(
            Any2ERA5Dataset(
                start=dates["train_start"],
                end=dates["train_end"],
                x_path=source_path,
                y_path=args.era5_path,
                source_name=source_name,
                source_idx=source_idx,
                max_samples_per_year=max_samples_per_year,
                sample_seed=args.sample_seed,
            )
        )
        val_sets.append(
            Any2ERA5Dataset(
                start=dates["val_start"],
                end=dates["val_end"],
                x_path=source_path,
                y_path=args.era5_path,
                source_name=source_name,
                source_idx=source_idx,
                val_sample_per_month=args.val_sample_per_month,
                val_sample_year=args.val_sample_year,
                sample_seed=args.sample_seed,
            )
        )

    train_set = ConcatDataset(train_sets)
    val_set = ConcatDataset(val_sets)

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate,
        prefetch_factor=args.prefetch_factor,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate,
        prefetch_factor=args.prefetch_factor,
    )

    base_model = A2E(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=args.in_chans,
        out_chans=args.out_chans,
        embed_dim=args.embed_dim,
        num_groups=args.num_groups,
        num_heads=args.num_heads,
        num_stages=args.num_stages,
        window_size=args.window_size,
        depth=depth,
        using_checkpoints=True,
        using_time_embedding=args.using_time_embedding,
        using_source_embedding=args.using_source_embedding,
        num_sources=len(SOURCE_REGISTRY),
        res_per_stage=res_per_stage,
        channels=channels,
        using_kl=args.using_kl,
        dropout_rate=args.dropout_rate,
        use_skip_connections=args.use_skip_connections,
        use_residual_blocks=args.use_residual_blocks,
        using_dann=args.using_dann,
    )

    param_count = sum(p.numel() for p in base_model.parameters())
    if is_master:
        print(f"A2E params: {param_count / 1e6:.2f} M")

    exp_dir = Path(args.output_root) / args.exp_name
    metrics_dir = exp_dir / "metrics"
    save_dir = Path(args.checkpoint_root) / args.exp_name
    tb_dir = Path(args.tensorboard_root) / args.exp_name
    plot_root = Path(args.plot_root) / args.exp_name

    exp_config = {
        "exp_name": args.exp_name,
        "sources": sources,
        "source_configs": source_configs,
        "date_ranges": {s: date_ranges[s] for s in sources},
        "paths": {
            "era5_path": args.era5_path,
            "gfs_path": args.gfs_path,
            "hres_path": args.hres_path,
            "cma_path": args.cma_path,
            "fuxi_dir": args.fuxi_dir,
            "save_dir": str(save_dir),
            "tb_dir": str(tb_dir),
            "plot_root": str(plot_root),
            "metrics_dir": str(metrics_dir),
        },
        "model": {
            "img_size": list(img_size),
            "patch_size": list(patch_size),
            "in_chans": args.in_chans,
            "out_chans": args.out_chans,
            "embed_dim": args.embed_dim,
            "channels": channels,
            "num_heads": args.num_heads,
            "num_stages": args.num_stages,
            "window_size": args.window_size,
            "depth": depth,
            "res_per_stage": res_per_stage,
            "using_time_embedding": args.using_time_embedding,
            "using_source_embedding": args.using_source_embedding,
            "use_skip_connections": args.use_skip_connections,
            "use_residual_blocks": args.use_residual_blocks,
            "dropout_rate": args.dropout_rate,
            "params": param_count,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "base_lr": args.base_lr,
            "min_lr": args.min_lr,
            "warmup_epochs": args.warmup_epochs,
            "weight_decay": args.weight_decay,
            "betas": list(betas),
            "seed": args.seed,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
            "use_amp": args.use_amp,
        },
        "loss": {
            "recon_loss_type": args.recon_loss_type,
            "use_grad_loss": args.use_grad_loss,
            "grad_loss_weight": args.grad_loss_weight,
            "fuxi_loss_mode": args.fuxi_loss_mode,
            "channel_rmse_weight": args.channel_rmse_weight,
            "rmse_every_n_steps": args.rmse_every_n_steps,
            "rmse_samples_per_batch": args.rmse_samples_per_batch,
            "target_rmse_channels": TARGET_RMSE_CHANNELS,
            "fuxi_era5_reference_rmse": FUXI_ERA5_REFERENCE_RMSE,
        },
    }

    if is_master:
        metrics_dir.mkdir(parents=True, exist_ok=True)
        with open(metrics_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(exp_config, f, indent=2, ensure_ascii=False)

    base_model.to(device)
    model = FSDP(base_model, device_id=device)

    fuxi_model = build_fuxi_model(device, fuxi_dir=args.fuxi_dir)
    channel_names = train_sets[0].target_channels if train_sets else None

    fuxi_rmse_interface = FuXiRMSEInterface(
        fuxi_model=fuxi_model,
        era5_zarr_path=args.era5_path,
        channel_names=channel_names,
        device=device,
        target_channels=TARGET_RMSE_CHANNELS,
        channel_weights=DEFAULT_CHANNEL_WEIGHTS,
        reference_rmse=FUXI_ERA5_REFERENCE_RMSE,
        loss_mode=args.fuxi_loss_mode,
        lead_hours=args.fuxi_lead_hours,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay,
        betas=betas,
    )

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs - args.warmup_epochs, 1),
        eta_min=args.min_lr,
    )
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epochs])

    trainer = FSDPUNetAlignMetricsTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        device=device,
        beta=args.beta,
        tb_dir=str(tb_dir),
        save_dir=str(save_dir),
        save_interval=args.save_interval,
        use_amp=args.use_amp,
        rank=rank,
        world_size=world_size,
        kl_anneal=args.kl_anneal,
        kl_anneal_epochs=args.kl_anneal_epochs,
        plot_root=str(plot_root),
        recon_loss_type=args.recon_loss_type,
        charbonnier_eps=args.charbonnier_eps,
        use_grad_loss=args.use_grad_loss,
        grad_loss_weight=args.grad_loss_weight,
        l1_reg_weight=args.l1_reg_weight,
        l2_reg_weight=args.l2_reg_weight,
        fuxi_model=fuxi_model,
        fuxi_rmse_interface=fuxi_rmse_interface,
        channel_rmse_weight=args.channel_rmse_weight,
        rmse_every_n_steps=args.rmse_every_n_steps,
        rmse_samples_per_batch=args.rmse_samples_per_batch,
        using_dann=args.using_dann,
        domain_loss_weight=args.domain_loss_weight,
        dann_gamma=args.dann_gamma,
        metrics_dir=str(metrics_dir),
        exp_config=exp_config,
    )

    try:
        trainer.train(resume_path=None, only_model=False)
    finally:
        fuxi_rmse_interface.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
