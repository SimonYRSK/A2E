import os

import numpy as np
import pandas as pd
import xarray as xr
import torch
import torch.nn.functional as F
import torch.distributed as dist
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim.lr_scheduler import ReduceLROnPlateau

from trainers.trainUNET import UNetTrainer
from models.dann import grl_lambda_schedule


class FSDPUNetTrainer(UNetTrainer):
    """UNetTrainer 的 FSDP 版本。

    - 支持单机多卡 FSDP 训练
    - 只在 rank==0 时写日志 / 保存模型 / 打印 epoch 级信息
    - 对 DistributedSampler 调用 set_epoch
    """

    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer,
        scheduler,
        epochs: int,
        device: torch.device,
        beta: float = 0.0,
        tb_dir: str = "./tensorboard_logs",
        save_dir: str = "./checkpoints",
        save_interval: int = 1,
        use_amp: bool = False,
        rank: int = 0,
        world_size: int = 1,
        is_master: bool | None = None,
        kl_anneal: bool = False,
        kl_anneal_epochs: int = 10,
        plot_root: str | None = None,
        recon_loss_type: str = "l2",
        use_grad_loss: bool = False,
        grad_loss_weight: float = 0.0,
        l1_reg_weight: float = 0.0,
        l2_reg_weight: float = 0.0,
        charbonnier_eps: float = 1e-3,
        using_dann: bool = False,
        domain_loss_weight: float = 0.1,
        dann_gamma: float = 10.0,
    ):
        self.rank = rank
        self.world_size = world_size
        self.is_master = (rank == 0) if is_master is None else is_master
        self.recon_loss_type = str(recon_loss_type).lower()
        if self.recon_loss_type not in {"l1", "l2", "charbonnier"}:
            raise ValueError(f"recon_loss_type must be one of ['l1','l2','charbonnier'], got: {recon_loss_type}")
        self.use_grad_loss = bool(use_grad_loss)
        self.grad_loss_weight = float(grad_loss_weight)
        self.l1_reg_weight = float(l1_reg_weight)
        self.l2_reg_weight = float(l2_reg_weight)
        self.charbonnier_eps = float(charbonnier_eps)
        self.using_dann = bool(using_dann)
        self.domain_loss_weight = float(domain_loss_weight)
        self.dann_gamma = float(dann_gamma)

        # 对 FSDP 包裹的模型，同步内外层的 using_kl 标志，
        # 确保 Trainer 在分布式场景下也能正确识别是否启用 KL
        inner_using_kl = False
        if hasattr(model, "using_kl"):
            inner_using_kl = bool(getattr(model, "using_kl", False))
        elif hasattr(model, "module") and hasattr(model.module, "using_kl"):
            inner_using_kl = bool(getattr(model.module, "using_kl", False))

        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=epochs,
            device=device,
            beta=beta,
            tb_dir=tb_dir,
            save_dir=save_dir,
            save_interval=save_interval,
            use_amp=use_amp,
            kl_anneal=kl_anneal,
            kl_anneal_epochs=kl_anneal_epochs,
        )

        # 覆盖/刷新 Trainer 自身的 using_kl 标志
        # （单卡时 UNetTrainer 已在 __init__ 中设置，这里在 FSDP 场景下做一次统一）
        self.using_kl = bool(getattr(self, "using_kl", False) or inner_using_kl)

        if self.is_master:
            print(f"[FSDPUNetTrainer] using_kl = {self.using_kl}")
            print(
                f"[FSDPUNetTrainer] recon_loss_type = {self.recon_loss_type}, "
                f"use_grad_loss = {self.use_grad_loss}, grad_loss_weight = {self.grad_loss_weight}"
            )
            if self.l1_reg_weight > 0 or self.l2_reg_weight > 0:
                print(
                    f"[FSDPUNetTrainer] l1_reg_weight = {self.l1_reg_weight}, "
                    f"l2_reg_weight = {self.l2_reg_weight}"
                )

        # 非主进程关闭 TensorBoard，避免多进程同时写
        if not self.is_master and hasattr(self, "writer") and self.writer is not None:
            self.writer.close()
            self.writer = None

        # 画图输出根目录（可由外部传入）
        self.plot_root = plot_root

        # 从验证集 Dataset 中记录通道名与经纬度，用于画图
        ds = getattr(val_loader, "dataset", None)
        self.plot_lat = None
        self.plot_lon = None
        self.channel_names = None
        self.channel_to_idx = None
        self.era5_mean = None
        self.era5_std = None
        if ds is not None:
            # ConcatDataset 不直接暴露底层属性，需要从第一个子数据集提取
            _ds_attr_source = ds
            from torch.utils.data import ConcatDataset as _ConcatDataset
            if isinstance(ds, _ConcatDataset) and len(ds.datasets) > 0:
                _ds_attr_source = ds.datasets[0]
                if self.is_master:
                    print(f"[Init] 检测到 ConcatDataset（{len(ds.datasets)} 个子集），"
                          f"从第一个子集提取通道名与反归一化参数")

            # GFS2ERA5Dataset / Any2ERA5Dataset 中有 target_channels 和 ds_y
            if hasattr(_ds_attr_source, "target_channels"):
                self.channel_names = list(_ds_attr_source.target_channels)
                self.channel_to_idx = {name: idx for idx, name in enumerate(self.channel_names)}

            # 加载经纬度坐标：依次尝试 ds_y（ERA5）和 ds_x（GFS），兼容不同命名习惯
            self._load_plot_coords(_ds_attr_source)

            # 尝试从 ERA5 路径加载反归一化所需的 mean/std
            try:
                self._load_denorm_stats(_ds_attr_source)
            except Exception as e:
                if self.is_master:
                    print(f"[Init] 加载 ERA5 归一化参数失败，将使用未反归一化的值绘图: {e}")

    def save_checkpoint(self, epoch, current_avg_loss):
        if not self.is_master:
            return
        super().save_checkpoint(epoch, current_avg_loss)

    def _load_denorm_stats(self, dataset):
        """从 ERA5 数据目录加载 mean/std，用于反归一化 GT 和预测。

        假定 ERA5 路径下存在 mean.nc 和 std.nc，格式与 FuXi 推理一致：
        使用 xr.open_dataarray 读取，并通过 channel 这个坐标对齐。
        只选择 target_channels 对应的通道（通常 70 个），保存为 numpy 数组 (C, H, W)。
        同时在首次加载时，可选地额外导出裁剪后的 meanc70.nc / stdc70.nc 方便复用。
        """
        # 数据集需要提供 y_path（ERA5 根目录）和 target_channels
        if not hasattr(dataset, "y_path") or self.channel_names is None:
            return

        era5_root = dataset.y_path
        mean_path = os.path.join(era5_root, "mean.nc")
        std_path = os.path.join(era5_root, "std.nc")

        if not (os.path.exists(mean_path) and os.path.exists(std_path)):
            if self.is_master:
                print(f"[Init] ERA5 mean/std 文件不存在: {mean_path}, {std_path}")
            return

        # 按 FuXi 推理方式读取：DataArray + 按 channel 对齐
        mean_da_full = xr.open_dataarray(mean_path)
        std_da_full = xr.open_dataarray(std_path)

        if "channel" not in mean_da_full.dims:
            if self.is_master:
                print("[Init] mean/std 中缺少 channel 维度，无法按通道名对齐")
            return

        # 裁剪并重排到 target_channels 顺序（通常 70 个通道）
        mean_da_c70 = mean_da_full.sel(channel=self.channel_names)
        std_da_c70 = std_da_full.sel(channel=self.channel_names)

        self.era5_mean = mean_da_c70.values.astype(np.float32)
        self.era5_std = std_da_c70.values.astype(np.float32)

        # 可选：在首次加载时额外导出 meanc70.nc / stdc70.nc，方便其他脚本直接使用
        if self.is_master:
            mean_c70_path = os.path.join(era5_root, "meanc70.nc")
            std_c70_path = os.path.join(era5_root, "stdc70.nc")
            try:
                if not os.path.exists(mean_c70_path):
                    mean_da_c70.to_netcdf(mean_c70_path)
                    print(f"[Init] 已导出裁剪后的均值文件: {mean_c70_path}")
                if not os.path.exists(std_c70_path):
                    std_da_c70.to_netcdf(std_c70_path)
                    print(f"[Init] 已导出裁剪后的方差文件: {std_c70_path}")
            except Exception as e:
                print(f"[Init] 导出 meanc70/stdc70 失败，可忽略（不影响训练绘图）: {e}")

            print(f"[Init] 已加载 ERA5 归一化参数，形状: mean={self.era5_mean.shape}, std={self.era5_std.shape}")

    def _load_plot_coords(self, dataset):
        """加载经纬度坐标，依次尝试多数据源和多种命名习惯。

        优先级：
        1. dataset.ds_y (ERA5) → "lat"/"lon"
        2. dataset.ds_y (ERA5) → "latitude"/"longitude"
        3. dataset.ds_x (GFS)  → "lat"/"lon"
        4. dataset.ds_x (GFS)  → "latitude"/"longitude"
        """
        coord_sources = []
        if hasattr(dataset, "ds_y"):
            coord_sources.append(("ds_y (ERA5)", dataset.ds_y))
        if hasattr(dataset, "ds_x"):
            coord_sources.append(("ds_x (GFS)", dataset.ds_x))

        name_pairs = [
            ("lat", "lon"),
            ("latitude", "longitude"),
        ]

        last_error = None
        for src_label, ds_obj in coord_sources:
            for lat_name, lon_name in name_pairs:
                try:
                    lat_vals = ds_obj[lat_name].values
                    lon_vals = ds_obj[lon_name].values
                    self.plot_lat = lat_vals
                    self.plot_lon = lon_vals
                    if self.is_master:
                        print(f"[Init] 已从 {src_label} 获取经纬度（{lat_name}/{lon_name}），"
                              f"lat=[{lat_vals.min():.2f}, {lat_vals.max():.2f}] "
                              f"lon=[{lon_vals.min():.2f}, {lon_vals.max():.2f}]")
                    return
                except Exception as e:
                    last_error = e
                    continue

        # 全部失败时打印最后一个错误
        if self.is_master:
            tried = ", ".join(
                f"{src_label}[{lat_name}]"
                for src_label, _ in coord_sources
                for lat_name, _ in name_pairs
            )
            print(f"[Init] 无法加载经纬度坐标（尝试了: {tried}），"
                  f"最后错误: {last_error}，将跳过画图")

    def _grl_lambda(self, current_step: int, total_steps: int) -> float:
        """计算当前步的 GRL lambda（仅在 using_dann 时有效）。"""
        if not self.using_dann:
            return 1.0
        return grl_lambda_schedule(current_step, total_steps, gamma=self.dann_gamma)

    def _all_reduce_loss(self, total_loss: float, total_recon: float, num_batches: int):
        """在所有进程间做 all_reduce，得到全局平均 loss。"""
        if not dist.is_available() or not dist.is_initialized():
            avg_loss = total_loss / max(num_batches, 1)
            avg_recon = total_recon / max(num_batches, 1)
            return avg_loss, avg_recon

        device = self.device if isinstance(self.device, torch.device) else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor = torch.tensor([total_loss, total_recon, float(num_batches)], device=device, dtype=torch.float64)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        total_loss_g, total_recon_g, num_batches_g = tensor.tolist()
        num_batches_g = max(num_batches_g, 1.0)
        avg_loss = float(total_loss_g / num_batches_g)
        avg_recon = float(total_recon_g / num_batches_g)
        return avg_loss, avg_recon

    def _compute_recon_loss_details(self, pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None):
        """返回 (recon_loss, l1_raw, l2_raw)。recon 可为 L1/L2/Charbonnier。"""
        l1_raw = None
        l2_raw = None

        if self.recon_loss_type == "l1":
            abs_err = torch.abs(pred - target)
            if weight is not None:
                l1_raw = torch.mean(abs_err * weight.float())
            else:
                l1_raw = torch.mean(abs_err)
            recon_loss = l1_raw
        elif self.recon_loss_type == "charbonnier":
            diff = pred - target
            eps2 = self.charbonnier_eps * self.charbonnier_eps
            charbonnier = torch.sqrt(diff * diff + eps2)
            if weight is not None:
                recon_loss = torch.mean(charbonnier * weight.float())
            else:
                recon_loss = torch.mean(charbonnier)
        else:
            l2_raw = self.cal_losses(pred, target, weight=weight)
            recon_loss = l2_raw

        return recon_loss, l1_raw, l2_raw

    def _compute_grad_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """一阶差分梯度损失（L1）。"""
        # 经向（lon）与纬向（lat）梯度
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        tgt_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        tgt_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

        loss_dx = torch.mean(torch.abs(pred_dx - tgt_dx))
        loss_dy = torch.mean(torch.abs(pred_dy - tgt_dy))
        return 0.5 * (loss_dx + loss_dy)

    def _compute_recon_loss_per_sample(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """按样本返回重建损失，形状 [B]。"""
        if self.recon_loss_type == "l1":
            diff = torch.abs(pred - target)
        elif self.recon_loss_type == "charbonnier":
            eps2 = self.charbonnier_eps * self.charbonnier_eps
            diff = torch.sqrt((pred - target) ** 2 + eps2)
        else:
            diff = (pred - target) ** 2

        if weight is not None:
            diff = diff * weight.float()

        return diff.mean(dim=(1, 2, 3))

    def _compute_grad_loss_per_sample(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """按样本返回一阶差分梯度损失（L1），形状 [B]。"""
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        tgt_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        tgt_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

        loss_dx = torch.mean(torch.abs(pred_dx - tgt_dx), dim=(1, 2, 3))
        loss_dy = torch.mean(torch.abs(pred_dy - tgt_dy), dim=(1, 2, 3))
        return 0.5 * (loss_dx + loss_dy)

    def _get_expected_domains(self) -> list[int] | None:
        model = self.model.module if hasattr(self.model, "module") else self.model
        num_sources = getattr(model, "num_sources", None)
        if num_sources is None:
            num_sources = getattr(model, "num_domains", None)
        if num_sources is None:
            return None
        return list(range(int(num_sources)))

    def _compute_reg_loss(self) -> tuple[torch.Tensor, torch.Tensor]:
        """计算 L1/L2 正则项（可选）。"""
        if (self.l1_reg_weight <= 0) and (self.l2_reg_weight <= 0):
            zero = torch.tensor(0.0, device=self.device)
            return zero, zero

        l1_reg = torch.tensor(0.0, device=self.device)
        l2_reg = torch.tensor(0.0, device=self.device)
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            if self.l1_reg_weight > 0:
                l1_reg = l1_reg + p.abs().sum()
            if self.l2_reg_weight > 0:
                l2_reg = l2_reg + p.pow(2).sum()

        return l1_reg, l2_reg

    def validate_one_epoch(self, epoch):
        self.model.eval()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_l1_loss = 0.0
        total_l2_loss = 0.0
        total_grad_loss = 0.0
        num_batches = 0

        # 按 domain 维度追踪画图状态：每个 domain 画一次即停
        plotted_domains: set[int] = set()
        expected_domains = set(self._get_expected_domains() or [])

        device_type = self.device.type if isinstance(self.device, torch.device) else str(self.device).split(":")[0]

        domain_stats = {}
        val_pbar = tqdm(self.vallo, desc=f"Epoch {epoch+1}/{self.epochs} Val", disable=not self.is_master)
        with torch.no_grad():
            for batch_idx, (x, y, i, times) in enumerate(val_pbar):
                x = x.to(self.device)
                y = y.to(self.device)
                domains = i.to(self.device)

                # 检查验证集 batch 是否存在 NaN/Inf，并打印对应时间
                has_nan_inf_x = torch.isnan(x).any() or torch.isinf(x).any()
                has_nan_inf_y = torch.isnan(y).any() or torch.isinf(y).any()
                if has_nan_inf_x or has_nan_inf_y:
                    if self.is_master:
                        times_str = ", ".join(str(t) for t in list(times))
                        print(f"[Val] batch {batch_idx} contains NaN/Inf, times: {times_str}")
                        print("[Val] 该 batch 已跳过，用于避免验证损失变为 NaN")
                    continue

                # domains: source domain index
                weights = self.lat_weight(y.shape)
                with torch.amp.autocast(device_type=device_type, enabled=self.use_amp):
                    if self.using_dann:
                        if getattr(self, "using_kl", False):
                            x_recon, mu, log_var, domain_logits = self.model(
                                x, times=times, domains=domains, grl_lambda=1.0,
                            )
                        else:
                            x_recon, domain_logits = self.model(
                                x, times=times, domains=domains, grl_lambda=1.0,
                            )
                            mu = log_var = None
                    else:
                        if getattr(self, "using_kl", False):
                            x_recon, mu, log_var = self.model(x, times=times, domains=domains)
                        else:
                            x_recon = self.model(x, times=times, domains=domains)
                            mu = log_var = None
                        domain_logits = None

                    recon_loss, l1_raw, l2_raw = self._compute_recon_loss_details(x_recon, y, weight=weights)
                    grad_loss = self._compute_grad_loss(x_recon, y) if self.use_grad_loss else torch.tensor(0.0, device=self.device)

                    if getattr(self, "using_kl", False) and mu is not None and log_var is not None:
                        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                        loss = recon_loss + self.grad_loss_weight * grad_loss + self.beta * kl_loss
                    else:
                        loss = recon_loss + self.grad_loss_weight * grad_loss

                total_loss += float(loss.detach())
                total_recon_loss += float(recon_loss.detach())
                if l1_raw is not None:
                    total_l1_loss += float(l1_raw.detach())
                if l2_raw is not None:
                    total_l2_loss += float(l2_raw.detach())
                total_grad_loss += float(grad_loss.detach())
                num_batches += 1

                if self.is_master:
                    postfix = {
                        "loss": f"{float(loss.detach()):.4f}",
                        "recon": f"{float(recon_loss.detach()):.4f}",
                    }
                    if self.use_grad_loss:
                        postfix["grad"] = f"{float(grad_loss.detach()):.4f}"
                    if self.using_dann and domain_logits is not None:
                        dacc = (domain_logits.argmax(dim=1) == domains).float().mean()
                        postfix["Dacc"] = f"{float(dacc):.2f}"
                    val_pbar.set_postfix(postfix)

                recon_per = self._compute_recon_loss_per_sample(x_recon, y, weight=weights)
                if self.use_grad_loss:
                    grad_per = self._compute_grad_loss_per_sample(x_recon, y)
                else:
                    grad_per = torch.zeros_like(recon_per)
                total_per = recon_per + self.grad_loss_weight * grad_per

                for domain_id in torch.unique(domains).tolist():
                    mask = domains == int(domain_id)
                    count = int(mask.sum().item())
                    if count == 0:
                        continue
                    stats = domain_stats.setdefault(
                        int(domain_id),
                        {"count": 0, "recon": 0.0, "grad": 0.0, "total": 0.0},
                    )
                    stats["count"] += count
                    stats["recon"] += float(recon_per[mask].sum().detach())
                    stats["grad"] += float(grad_per[mask].sum().detach())
                    stats["total"] += float(total_per[mask].sum().detach())

                # 对尚未画过的 domain，取第一个样本画图
                if self.is_master and expected_domains:
                    current_domains = set(torch.unique(domains).tolist())
                    new_domains = current_domains - plotted_domains
                    if new_domains:
                        try:
                            self._plot_validation_maps(epoch, x_recon, y, times, domains=domains, skip_domains=plotted_domains)
                        except Exception as e:
                            if self.is_master:
                                print(f"[Val] 绘图时出错: {e}")
                        plotted_domains |= new_domains

        # 若全部 batch 都被跳过，避免除以 0
        if num_batches == 0:
            if self.is_master:
                print("[Val] 所有 batch 均因包含 NaN/Inf 被跳过，返回损失 0.0 以保持训练继续进行")
            avg_loss = 0.0
            avg_recon = 0.0
        else:
            avg_loss, avg_recon = self._all_reduce_loss(total_loss, total_recon_loss, num_batches)

        # 日志项按 mode 输出（这里按本 rank 统计；主进程打印）
        avg_l1 = total_l1_loss / max(num_batches, 1)
        avg_l2 = total_l2_loss / max(num_batches, 1)
        avg_grad = total_grad_loss / max(num_batches, 1)

        if self.is_master:
            print(f"\nEpoch {epoch+1} 验证集平均:")
            if self.recon_loss_type == "l1":
                if self.use_grad_loss:
                    print(f"总损失={avg_loss:.5f}, L1loss={avg_l1:.5f}, Gradloss={avg_grad:.5f}")
                else:
                    print(f"总损失={avg_loss:.5f}, L1loss={avg_l1:.5f}")
            elif self.recon_loss_type == "charbonnier":
                if self.use_grad_loss:
                    print(f"总损失={avg_loss:.5f}, Charbonnierloss={avg_recon:.5f}, Gradloss={avg_grad:.5f}")
                else:
                    print(f"总损失={avg_loss:.5f}, Charbonnierloss={avg_recon:.5f}")
            elif self.recon_loss_type == "l2":
                if self.use_grad_loss:
                    print(f"总损失={avg_loss:.5f}, L2loss={avg_l2:.5f}, Gradloss={avg_grad:.5f}")
                else:
                    print(f"总损失={avg_loss:.5f}, L2loss={avg_l2:.5f}")

            global_step = epoch
            if hasattr(self, "writer") and self.writer:
                self.writer.add_scalar("Loss/val/total", avg_loss, global_step)
                if self.recon_loss_type == "l1":
                    self.writer.add_scalar("Loss/val/L1loss", avg_l1, global_step)
                elif self.recon_loss_type == "charbonnier":
                    self.writer.add_scalar("Loss/val/Charbonnierloss", avg_recon, global_step)
                else:
                    self.writer.add_scalar("Loss/val/L2loss", avg_l2, global_step)
                if self.use_grad_loss:
                    self.writer.add_scalar("Loss/val/Gradloss", avg_grad, global_step)

                for domain_id, stats in sorted(domain_stats.items()):
                    if stats["count"] == 0:
                        continue
                    denom = max(stats["count"], 1)
                    recon_avg = stats["recon"] / denom
                    grad_avg = stats["grad"] / denom
                    total_avg = stats["total"] / denom
                    self.writer.add_scalar(f"Loss/val/domain_{domain_id}/recon", recon_avg, global_step)
                    self.writer.add_scalar(f"Loss/val/domain_{domain_id}/total_no_kl", total_avg, global_step)
                    if self.use_grad_loss:
                        self.writer.add_scalar(f"Loss/val/domain_{domain_id}/grad", grad_avg, global_step)

            expected_domains = self._get_expected_domains()
            if expected_domains is not None:
                missing = sorted(set(expected_domains) - set(domain_stats.keys()))
                if missing:
                    print(f"[Val] 警告：本轮验证未覆盖源域 {missing}")

        return avg_loss

    def _plot_validation_maps(self, epoch, x_recon, y, times, domains=None, skip_domains=None):
        """在验证集首个正常 batch 上，为每个源域取第一个样本画 GT vs 预测 对比图。

        仅在 rank0 调用。每个 domain 每个 epoch 只画一次，已画过的 domain 通过 skip_domains 跳过。
        """
        if not self.is_master:
            return

        if self.plot_lat is None or self.plot_lon is None:
            print("[Val] 无法获取经纬度坐标，跳过画图")
            return

        if self.channel_to_idx is None or self.channel_names is None:
            print("[Val] 无法获取通道名称，跳过画图")
            return

        # 近地面变量与 500hPa 高空变量通道名
        near_surface_channels = ["t2m", "u10m", "v10m", "msl", "tp"]
        level500_channels = ["t500", "u500", "v500", "z500", "q500"]

        # 获取源数据反向注册表，用于获取源名称
        from data.pairsetc226 import SOURCE_REGISTRY
        INV_SOURCE_REGISTRY = {v: k for k, v in SOURCE_REGISTRY.items()}

        lat = self.plot_lat
        lon = self.plot_lon

        # 输出根目录：优先使用外部传入的 plot_root
        if self.plot_root is None:
            print("[Val] 未设置 plot_root，使用默认路径")
            out_root = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/A2E/channelpics/swinunet_2022_2024_3_21"
        else:
            out_root = self.plot_root

        # 时间字符串用于文件名（取首个样本的时间，同一 batch 内时间应相同）
        try:
            t0 = pd.Timestamp(str(times[0]))
            time_str = t0.strftime("%Y%m%d_%H%M")
        except Exception:
            time_str = "unknown_time"

        # 找出 batch 中每个源域的第一个样本索引
        domain_indices = {}
        if domains is not None:
            for di in range(len(domains)):
                d = int(domains[di].item())
                if d not in domain_indices:
                    domain_indices[d] = di

        if not domain_indices:
            # 无 domain 信息时退化为画第一个样本
            domain_indices = {0: 0}

        skip = set(skip_domains or [])
        for domain_idx, sample_idx in domain_indices.items():
            if domain_idx in skip:
                continue
            source_name = INV_SOURCE_REGISTRY.get(domain_idx, f"source{domain_idx}")

            pred_sample = x_recon[sample_idx].detach().cpu().numpy()  # (C, H, W)
            gt_sample = y[sample_idx].detach().cpu().numpy()          # (C, H, W)

            # 按源名称分子文件夹存放
            epoch_dir = os.path.join(out_root, f"epoch_{epoch+1:03d}", source_name)
            os.makedirs(epoch_dir, exist_ok=True)

            def plot_one_channel(ch_name: str, _pred=pred_sample, _gt=gt_sample):
                if ch_name not in self.channel_to_idx:
                    print(f"[Val] 通道 {ch_name} 不在当前 target_channels 中，跳过")
                    return

                idx = self.channel_to_idx[ch_name]
                gt_2d = _gt[idx]
                pred_2d = _pred[idx]

                # 若存在 ERA5 的 mean/std，则先反归一化到物理量
                if self.era5_mean is not None and self.era5_std is not None:
                    try:
                        mean_2d = self.era5_mean[idx]
                        std_2d = self.era5_std[idx]
                        gt_2d = gt_2d * std_2d + mean_2d
                        pred_2d = pred_2d * std_2d + mean_2d
                    except Exception as e:
                        if self.is_master:
                            print(f"[Val] 通道 {ch_name} 反归一化失败，将使用归一化值绘图: {e}")

                # 统一 GT 与 Forecast 的 colorbar 范围
                vmin = float(np.nanmin([gt_2d.min(), pred_2d.min()]))
                vmax = float(np.nanmax([gt_2d.max(), pred_2d.max()]))
                if vmin == vmax:
                    vmax = vmin + 1e-6

                diff_2d = pred_2d - gt_2d
                diff_max = float(np.nanmax(np.abs(diff_2d)))
                if diff_max == 0:
                    diff_max = 1e-6

                fig, axes = plt.subplots(1, 3, figsize=(15, 4))

                im0 = axes[0].pcolormesh(lon, lat, gt_2d, shading="auto", vmin=vmin, vmax=vmax)
                axes[0].set_title(f"GT - {ch_name}")
                plt.colorbar(im0, ax=axes[0])

                im1 = axes[1].pcolormesh(lon, lat, pred_2d, shading="auto", vmin=vmin, vmax=vmax)
                axes[1].set_title(f"Forecast - {ch_name}")
                plt.colorbar(im1, ax=axes[1])

                # 差值图使用蓝-白-红的发散色图：0 附近为白/灰，正值为红，负值为蓝
                im2 = axes[2].pcolormesh(
                    lon,
                    lat,
                    diff_2d,
                    shading="auto",
                    vmin=-diff_max,
                    vmax=diff_max,
                    cmap="bwr",
                )
                axes[2].set_title(f"Forecast - GT - {ch_name}")
                plt.colorbar(im2, ax=axes[2])

                for ax in axes:
                    ax.set_xlabel("lon")
                    ax.set_ylabel("lat")

                fig.suptitle(f"Epoch {epoch+1} Val Sample, {time_str}, {ch_name}")
                fig.tight_layout()

                fname = f"epoch{epoch+1:03d}_{time_str}_{ch_name}.png"
                save_path = os.path.join(epoch_dir, fname)
                plt.savefig(save_path, dpi=200, bbox_inches="tight")
                plt.close(fig)
                print(f"[Val] 已保存通道 {ch_name} 图像到: {save_path}")

            for ch in near_surface_channels + level500_channels:
                plot_one_channel(ch)

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_l1_loss = 0.0
        total_l2_loss = 0.0
        total_grad_loss = 0.0
        total_kl_loss = 0.0
        total_domain_loss = 0.0
        total_domain_acc = 0.0
        num_batches = 0

        # DistributedSampler 设 epoch，保证每轮 shuffle 不同
        sampler = getattr(self.trainlo, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

        from tqdm import tqdm

        device_type = self.device.type if isinstance(self.device, torch.device) else str(self.device).split(":")[0]

        pbar = tqdm(self.trainlo, desc=f"Epoch {epoch+1}/{self.epochs}", disable=not self.is_master)

        for batch_idx, (x, y, i, times) in enumerate(pbar):
            x = x.to(self.device)
            y = y.to(self.device)
            domains = i.to(self.device)
            # domains: source domain index
            has_nan_inf_x = torch.isnan(x).any() or torch.isinf(x).any()
            has_nan_inf_y = torch.isnan(y).any() or torch.isinf(y).any()
            if has_nan_inf_x or has_nan_inf_y:
                if self.is_master:
                    times_str = ", ".join(str(t) for t in list(times))
                    print(f"[Train] batch {batch_idx} contains NaN/Inf, times: {times_str}")
                    print("[Train] 该 batch 已跳过，用于避免训练权重被 NaN 污染")
                continue

            weights = self.lat_weight(y.shape)

            self.opt.zero_grad(set_to_none=True)

            # GRL lambda 渐进调度
            total_steps = self.epochs * len(self.trainlo)
            current_step = epoch * len(self.trainlo) + batch_idx
            grl_lambda = self._grl_lambda(current_step, total_steps)

            with torch.amp.autocast(device_type=device_type, enabled=self.use_amp):
                if self.using_dann:
                    if getattr(self, "using_kl", False):
                        x_recon, mu, log_var, domain_logits = self.model(
                            x, times=times, domains=domains, grl_lambda=grl_lambda,
                        )
                    else:
                        x_recon, domain_logits = self.model(
                            x, times=times, domains=domains, grl_lambda=grl_lambda,
                        )
                        mu = log_var = None
                else:
                    if getattr(self, "using_kl", False):
                        x_recon, mu, log_var = self.model(x, times=times, domains=domains)
                    else:
                        x_recon = self.model(x, times=times, domains=domains)
                        mu = log_var = None
                    domain_logits = None

                recon_loss, l1_raw, l2_raw = self._compute_recon_loss_details(x_recon, y, weight=weights)
                grad_loss = self._compute_grad_loss(x_recon, y) if self.use_grad_loss else torch.tensor(0.0, device=self.device)

                # 域分类损失
                domain_loss = torch.tensor(0.0, device=self.device)
                if domain_logits is not None:
                    domain_loss = F.cross_entropy(domain_logits, domains)

                if getattr(self, "using_kl", False) and mu is not None and log_var is not None:
                    kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                    loss = recon_loss + self.grad_loss_weight * grad_loss + self.beta * kl_loss + self.domain_loss_weight * domain_loss
                else:
                    kl_loss = torch.tensor(0.0, device=self.device)
                    loss = recon_loss + self.grad_loss_weight * grad_loss + self.domain_loss_weight * domain_loss

            # 如果 loss 本身出现 NaN/Inf，同样跳过该 batch，避免反向传播污染参数
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                if self.is_master:
                    times_str = ", ".join(str(t) for t in list(times))
                    print(f"[Train] batch {batch_idx} loss is NaN/Inf, times: {times_str}")
                    print("[Train] 该 batch 的梯度已跳过，请检查数据或数值稳定性")
                continue

            loss_item = float(loss.detach())
            recon_item = float(recon_loss.detach())
            kl_item = float(kl_loss.detach())
            domain_loss_item = float(domain_loss.detach())
            domain_acc_item = float(
                (domain_logits.argmax(dim=1) == domains).float().mean()
            ) if domain_logits is not None else 0.0

            total_loss += loss_item
            total_recon_loss += recon_item
            if l1_raw is not None:
                total_l1_loss += float(l1_raw.detach())
            if l2_raw is not None:
                total_l2_loss += float(l2_raw.detach())
            total_grad_loss += float(grad_loss.detach())
            total_kl_loss += kl_item
            total_domain_loss += domain_loss_item
            total_domain_acc += domain_acc_item
            num_batches += 1

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)

            # FSDP 建议使用专门的梯度裁剪：传入 FSDP 模型本身
            if isinstance(self.model, FSDP):
                FSDP.clip_grad_norm_(self.model, max_norm=5.0)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            self.scaler.step(self.opt)
            self.scaler.update()

            if self.is_master:
                recon_name = "L1loss" if self.recon_loss_type == "l1" else ("L2loss" if self.recon_loss_type == "l2" else "Charbonnierloss")
                if getattr(self, "using_kl", False):
                    postfix = {
                        'Loss': f'{loss_item:.4f}',
                        recon_name: f'{recon_item:.4f}',
                        'KL': f'{kl_item:.4f}',
                    }
                    if self.use_grad_loss:
                        postfix['Grad'] = f'{float(grad_loss.detach()):.4f}'
                    if self.using_dann:
                        postfix['Dacc'] = f'{domain_acc_item:.2f}'
                    pbar.set_postfix(postfix)
                else:
                    postfix = {
                        'Loss': f'{loss_item:.4f}',
                        recon_name: f'{recon_item:.4f}',
                    }
                    if self.use_grad_loss:
                        postfix['Grad'] = f'{float(grad_loss.detach()):.4f}'
                    if self.using_dann:
                        postfix['Dacc'] = f'{domain_acc_item:.2f}'
                    pbar.set_postfix(postfix)

                if batch_idx % 10 == 0 and hasattr(self, 'writer') and self.writer:
                    step = epoch * len(self.trainlo) + batch_idx
                    self.writer.add_scalar("Loss/batch/total", loss_item, step)
                    if self.recon_loss_type == "l1":
                        self.writer.add_scalar("Loss/batch/L1loss", float(l1_raw.detach()) if l1_raw is not None else recon_item, step)
                    elif self.recon_loss_type == "charbonnier":
                        self.writer.add_scalar("Loss/batch/Charbonnierloss", recon_item, step)
                    else:
                        self.writer.add_scalar("Loss/batch/L2loss", float(l2_raw.detach()) if l2_raw is not None else recon_item, step)
                    if self.use_grad_loss:
                        self.writer.add_scalar("Loss/batch/Gradloss", float(grad_loss.detach()), step)
                    if getattr(self, "using_kl", False):
                        self.writer.add_scalar("Loss/batch/kl", kl_item, step)
                    if self.using_dann:
                        self.writer.add_scalar("DANN/batch/domain_loss", domain_loss_item, step)
                        self.writer.add_scalar("DANN/batch/domain_acc", domain_acc_item, step)
                        self.writer.add_scalar("DANN/batch/grl_lambda", grl_lambda, step)

        # 得到全局平均 train loss（这里只 all_reduce 总损失和重建损失，KL 已体现在总损失中）
        if num_batches == 0:
            if self.is_master:
                print("[Train] 本 epoch 所有 batch 均因 NaN/Inf 被跳过，返回损失 0.0")
            avg_loss = 0.0
            avg_recon = 0.0
        else:
            avg_loss, avg_recon = self._all_reduce_loss(total_loss, total_recon_loss, num_batches)

        avg_l1 = total_l1_loss / max(num_batches, 1)
        avg_l2 = total_l2_loss / max(num_batches, 1)
        avg_grad = total_grad_loss / max(num_batches, 1)
        avg_domain_loss = total_domain_loss / max(num_batches, 1)
        avg_domain_acc = total_domain_acc / max(num_batches, 1)

        # 验证也返回全局损失
        val_loss = self.validate_one_epoch(epoch)

        # scheduler 所有 rank 都要 step，保证 lr 一致
        if isinstance(self.sch, ReduceLROnPlateau):
            self.sch.step(val_loss)
        else:
            self.sch.step()

        if self.is_master:
            print(f"\nEpoch {epoch+1} 训练集平均:")
            if self.recon_loss_type == "l1":
                if self.use_grad_loss:
                    print(f"总损失={avg_loss:.5f}, L1loss={avg_l1:.5f}, Gradloss={avg_grad:.5f}")
                else:
                    print(f"总损失={avg_loss:.5f}, L1loss={avg_l1:.5f}")
            elif self.recon_loss_type == "charbonnier":
                if self.use_grad_loss:
                    print(f"总损失={avg_loss:.5f}, Charbonnierloss={avg_recon:.5f}, Gradloss={avg_grad:.5f}")
                else:
                    print(f"总损失={avg_loss:.5f}, Charbonnierloss={avg_recon:.5f}")
            else:
                if self.use_grad_loss:
                    print(f"总损失={avg_loss:.5f}, L2loss={avg_l2:.5f}, Gradloss={avg_grad:.5f}")
                else:
                    print(f"总损失={avg_loss:.5f}, L2loss={avg_l2:.5f}")

            if self.using_dann:
                print(f"DomainLoss={avg_domain_loss:.5f}, DomainAcc={avg_domain_acc:.3f}")

            global_step = epoch
            if hasattr(self, 'writer') and self.writer:
                self.writer.add_scalar("Loss/train/total",    avg_loss,  global_step)
                if self.recon_loss_type == "l1":
                    self.writer.add_scalar("Loss/train/L1loss", avg_l1, global_step)
                elif self.recon_loss_type == "charbonnier":
                    self.writer.add_scalar("Loss/train/Charbonnierloss", avg_recon, global_step)
                else:
                    self.writer.add_scalar("Loss/train/L2loss", avg_l2, global_step)
                if self.use_grad_loss:
                    self.writer.add_scalar("Loss/train/Gradloss", avg_grad, global_step)
                if self.using_dann:
                    self.writer.add_scalar("DANN/epoch/domain_loss", avg_domain_loss, global_step)
                    self.writer.add_scalar("DANN/epoch/domain_acc", avg_domain_acc, global_step)
                # 当使用 KL 且启用 KL annealing 时，记录当前 beta
                if getattr(self, "using_kl", False) and getattr(self, "kl_anneal", False):
                    self.writer.add_scalar("hyper/beta",      self.beta, global_step)
                self.writer.add_scalar("hyper/lr",            self.opt.param_groups[0]['lr'], global_step)

        return avg_loss, val_loss
