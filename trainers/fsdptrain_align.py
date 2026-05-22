import torch
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from trainers.fsdptrain import FSDPUNetTrainer


class FSDPUNetAlignTrainer(FSDPUNetTrainer):
    """A2E + FuXi 联合训练 Trainer。

    目标：
    1) 训练 A2E（参数更新）
    2) 冻结 FuXi，仅作为一步推理 RMSE 评估器
    3) 总损失 = A2E 重建相关损失 + channel_rmse_weight * FuXi 通道RMSE损失
    """

    def __init__(
        self,
        *args,
        fuxi_model,
        fuxi_rmse_interface,
        channel_rmse_weight: float = 0.5,
        rmse_every_n_steps: int = 1,
        rmse_samples_per_batch: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.fuxi_model = fuxi_model
        self.fuxi_rmse_interface = fuxi_rmse_interface
        self.channel_rmse_weight = float(channel_rmse_weight)
        self.rmse_every_n_steps = max(1, int(rmse_every_n_steps))
        self.rmse_samples_per_batch = int(rmse_samples_per_batch)

        self.fuxi_model.eval()
        for p in self.fuxi_model.parameters():
            p.requires_grad = False

        if self.is_master:
            print(f"[FSDPUNetAlignTrainer] channel_rmse_weight = {self.channel_rmse_weight}")
            print(
                f"[FSDPUNetAlignTrainer] rmse_every_n_steps = {self.rmse_every_n_steps}, "
                f"rmse_samples_per_batch = {self.rmse_samples_per_batch}"
            )

        self._era5_mean_t = None
        self._era5_std_t = None
        if self.era5_mean is not None and self.era5_std is not None:
            expected_c = len(self.channel_names) if self.channel_names is not None else None
            self._era5_mean_t = self._prepare_stats_tensor(self.era5_mean, expected_c=expected_c).to(self.device)
            self._era5_std_t = self._prepare_stats_tensor(self.era5_std, expected_c=expected_c).to(self.device)

    def _prepare_stats_tensor(self, arr, expected_c=None) -> torch.Tensor:
        t = torch.from_numpy(arr) if not torch.is_tensor(arr) else arr
        t = t.float()

        # 目标形状：[1, C, H, W]，兼容 [C] / [C,H,W] / [H,W,C] / [1,C,H,W]
        if t.ndim == 1:
            t = t.view(1, t.shape[0], 1, 1)
        elif t.ndim == 3:
            if expected_c is not None and t.shape[0] == expected_c:
                t = t.unsqueeze(0)
            elif expected_c is not None and t.shape[-1] == expected_c:
                t = t.permute(2, 0, 1).unsqueeze(0)
            else:
                # 默认按 channel-first 处理
                t = t.unsqueeze(0)
        elif t.ndim == 4:
            # [B,C,H,W] 或 [B,H,W,C]
            if expected_c is not None and t.shape[1] == expected_c:
                pass
            elif expected_c is not None and t.shape[-1] == expected_c:
                t = t.permute(0, 3, 1, 2)
            if t.shape[0] != 1:
                t = t[:1]
        else:
            raise ValueError(f"Unsupported stats ndim: {t.ndim}")

        return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

    def _nan_clean(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def _denorm_pred(self, x_norm: torch.Tensor) -> torch.Tensor:
        if self._era5_mean_t is None or self._era5_std_t is None:
            return self._nan_clean(x_norm)
        mean_t = self._era5_mean_t
        std_t = self._era5_std_t

        # 通道兜底对齐
        if mean_t.shape[1] != x_norm.shape[1]:
            c = min(mean_t.shape[1], x_norm.shape[1])
            mean_t = mean_t[:, :c]
            std_t = std_t[:, :c]
            x_norm = x_norm[:, :c]

        # 空间分辨率兜底对齐
        if mean_t.shape[-2:] != x_norm.shape[-2:]:
            mean_t = torch.nn.functional.interpolate(mean_t, size=x_norm.shape[-2:], mode="bilinear", align_corners=False)
            std_t = torch.nn.functional.interpolate(std_t, size=x_norm.shape[-2:], mode="bilinear", align_corners=False)

        out = x_norm * std_t + mean_t
        return self._nan_clean(out)

    def validate_one_epoch(self, epoch):
        self.model.eval()
        self.fuxi_model.eval()

        total_loss = 0.0
        total_recon_loss = 0.0
        total_grad_loss = 0.0
        total_channel_loss = 0.0
        num_batches = 0

        channel_sum = {ch: 0.0 for ch in self.fuxi_rmse_interface.target_channels}
        channel_n = 0
        domain_channel_sum = {}
        domain_channel_n = {}

        device_type = self.device.type if isinstance(self.device, torch.device) else str(self.device).split(":")[0]

        domain_stats = {}
        domain_rmse_stats = {}
        with torch.no_grad():
            for batch_idx, (x, y, i, times) in enumerate(self.vallo):
                x = self._nan_clean(x.to(self.device))
                y = self._nan_clean(y.to(self.device))
                domains = i.to(self.device)

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

                    x_recon = self._nan_clean(x_recon)
                    recon_loss, l1_raw, l2_raw = self._compute_recon_loss_details(x_recon, y, weight=weights)
                    grad_loss = self._compute_grad_loss(x_recon, y) if self.use_grad_loss else torch.tensor(0.0, device=self.device)

                    if getattr(self, "using_kl", False) and mu is not None and log_var is not None:
                        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                    else:
                        kl_loss = torch.tensor(0.0, device=self.device)

                do_rmse = (batch_idx % self.rmse_every_n_steps == 0)
                if do_rmse:
                    x_recon_denorm = self._denorm_pred(x_recon)
                    if self.rmse_samples_per_batch > 0 and self.rmse_samples_per_batch < x_recon_denorm.shape[0]:
                        k = self.rmse_samples_per_batch
                        x_recon_denorm_rmse = x_recon_denorm[:k]
                        times_rmse = times[:k]
                    else:
                        x_recon_denorm_rmse = x_recon_denorm
                        times_rmse = times

                    rmse_loss_norm, rmse_dict, weighted_rmse_raw, valid_count = self.fuxi_rmse_interface.compute_batch_loss(
                        x_recon_denorm_rmse,
                        times_rmse,
                        requires_grad=False,
                    )
                else:
                    rmse_loss_norm = torch.tensor(0.0, device=self.device)
                    rmse_dict = {ch: 0.0 for ch in self.fuxi_rmse_interface.target_channels}
                    weighted_rmse_raw = 0.0
                    valid_count = 0

                loss = recon_loss + self.grad_loss_weight * grad_loss + self.beta * kl_loss + self.channel_rmse_weight * rmse_loss_norm

                total_loss += float(loss.detach())
                total_recon_loss += float(recon_loss.detach())
                total_grad_loss += float(grad_loss.detach())
                total_channel_loss += float(rmse_loss_norm.detach())
                num_batches += 1

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

                if do_rmse:
                    for domain_id in torch.unique(domains).tolist():
                        mask = domains == int(domain_id)
                        count = int(mask.sum().item())
                        if count == 0:
                            continue
                        x_rmse = x_recon_denorm[mask]
                        times_rmse_dom = times[mask.cpu().numpy()]
                        rmse_loss_dom, rmse_dict_dom, weighted_rmse_dom, valid_dom = self.fuxi_rmse_interface.compute_batch_loss(
                            x_rmse,
                            times_rmse_dom,
                            requires_grad=False,
                        )
                        if valid_dom <= 0:
                            continue
                        rmse_stats = domain_rmse_stats.setdefault(
                            int(domain_id),
                            {"count": 0, "rmse_norm": 0.0, "rmse_raw": 0.0},
                        )
                        rmse_stats["count"] += 1
                        rmse_stats["rmse_norm"] += float(rmse_loss_dom.detach())
                        rmse_stats["rmse_raw"] += float(weighted_rmse_dom)

                        channel_stats = domain_channel_sum.setdefault(
                            int(domain_id),
                            {ch: 0.0 for ch in self.fuxi_rmse_interface.target_channels},
                        )
                        for ch in channel_stats:
                            channel_stats[ch] += rmse_dict_dom.get(ch, 0.0)
                        domain_channel_n[int(domain_id)] = domain_channel_n.get(int(domain_id), 0) + 1

                if valid_count > 0:
                    for ch in channel_sum:
                        channel_sum[ch] += rmse_dict.get(ch, 0.0)
                    channel_n += 1

                if self.is_master and batch_idx % 20 == 0 and self.writer is not None:
                    step = epoch * len(self.vallo) + batch_idx
                    self.writer.add_scalar("Align/batch_val/weighted_rmse_raw", weighted_rmse_raw, step)
                    for ch, v in rmse_dict.items():
                        self.writer.add_scalar(f"Align/batch_val/channel_rmse/{ch}", v, step)

        if num_batches == 0:
            avg_loss = 0.0
            avg_recon = 0.0
        else:
            avg_loss, avg_recon = self._all_reduce_loss(total_loss, total_recon_loss, num_batches)

        avg_grad = total_grad_loss / max(num_batches, 1)
        avg_channel = total_channel_loss / max(num_batches, 1)
        avg_channel_dict = {ch: (channel_sum[ch] / max(channel_n, 1)) for ch in channel_sum}
        weighted_rmse_epoch = sum(
            self.fuxi_rmse_interface.channel_weights[ch] * avg_channel_dict[ch]
            for ch in self.fuxi_rmse_interface.target_channels
        )

        if self.is_master:
            print(
                f"\nEpoch {epoch+1} 验证集平均: 总损失={avg_loss:.5f}, "
                f"重建={avg_recon:.5f}, Grad={avg_grad:.5f}, AlignRMSE(norm)={avg_channel:.5f}, "
                f"AlignRMSE(raw)={weighted_rmse_epoch:.5f}"
            )
            if self.writer is not None:
                self.writer.add_scalar("Loss/val/total", avg_loss, epoch)
                self.writer.add_scalar("Loss/val/recon", avg_recon, epoch)
                if self.use_grad_loss:
                    self.writer.add_scalar("Loss/val/Gradloss", avg_grad, epoch)
                self.writer.add_scalar("Align/val/rmse_norm", avg_channel, epoch)
                self.writer.add_scalar("Align/val/weighted_rmse_raw", weighted_rmse_epoch, epoch)
                for ch, v in avg_channel_dict.items():
                    self.writer.add_scalar(f"Align/val/channel_rmse/{ch}", v, epoch)

                for domain_id, ch_stats in sorted(domain_channel_sum.items()):
                    n_dom = max(domain_channel_n.get(domain_id, 0), 1)
                    for ch, v in ch_stats.items():
                        self.writer.add_scalar(
                            f"Align/val/domain_{domain_id}/channel_rmse/{ch}",
                            v / n_dom,
                            epoch,
                        )

                for domain_id, stats in sorted(domain_stats.items()):
                    if stats["count"] == 0:
                        continue
                    denom = max(stats["count"], 1)
                    recon_avg = stats["recon"] / denom
                    grad_avg = stats["grad"] / denom
                    total_avg = stats["total"] / denom
                    self.writer.add_scalar(f"Loss/val/domain_{domain_id}/recon", recon_avg, epoch)
                    self.writer.add_scalar(f"Loss/val/domain_{domain_id}/total_no_kl", total_avg, epoch)
                    if self.use_grad_loss:
                        self.writer.add_scalar(f"Loss/val/domain_{domain_id}/grad", grad_avg, epoch)

                for domain_id, stats in sorted(domain_rmse_stats.items()):
                    if stats["count"] == 0:
                        continue
                    denom = max(stats["count"], 1)
                    rmse_norm_avg = stats["rmse_norm"] / denom
                    rmse_raw_avg = stats["rmse_raw"] / denom
                    self.writer.add_scalar(f"Align/val/domain_{domain_id}/rmse_norm", rmse_norm_avg, epoch)
                    self.writer.add_scalar(f"Align/val/domain_{domain_id}/weighted_rmse_raw", rmse_raw_avg, epoch)

            expected_domains = self._get_expected_domains()
            if expected_domains is not None:
                missing = sorted(set(expected_domains) - set(domain_stats.keys()))
                if missing:
                    print(f"[Val] 警告：本轮验证未覆盖源域 {missing}")

        return avg_loss

    def train_one_epoch(self, epoch):
        self.model.train()
        self.fuxi_model.eval()

        total_loss = 0.0
        total_recon_loss = 0.0
        total_grad_loss = 0.0
        total_channel_loss = 0.0
        num_batches = 0

        channel_sum = {ch: 0.0 for ch in self.fuxi_rmse_interface.target_channels}
        channel_n = 0

        sampler = getattr(self.trainlo, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

        device_type = self.device.type if isinstance(self.device, torch.device) else str(self.device).split(":")[0]

        pbar = tqdm(self.trainlo, desc=f"Epoch {epoch+1}/{self.epochs}", disable=not self.is_master)

        for batch_idx, (x, y, i, times) in enumerate(pbar):
            x = self._nan_clean(x.to(self.device))
            y = self._nan_clean(y.to(self.device))
            domains = i.to(self.device)

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

                x_recon = self._nan_clean(x_recon)

                recon_loss, l1_raw, l2_raw = self._compute_recon_loss_details(x_recon, y, weight=weights)
                grad_loss = self._compute_grad_loss(x_recon, y) if self.use_grad_loss else torch.tensor(0.0, device=self.device)

                if getattr(self, "using_kl", False) and mu is not None and log_var is not None:
                    kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                else:
                    kl_loss = torch.tensor(0.0, device=self.device)

                # 域分类损失 (DANN)
                domain_loss = torch.tensor(0.0, device=self.device)
                if domain_logits is not None:
                    domain_loss = F.cross_entropy(domain_logits, domains)

            do_rmse = (batch_idx % self.rmse_every_n_steps == 0)
            if do_rmse:
                x_recon_denorm = self._denorm_pred(x_recon)
                if self.rmse_samples_per_batch > 0 and self.rmse_samples_per_batch < x_recon_denorm.shape[0]:
                    k = self.rmse_samples_per_batch
                    x_recon_denorm_rmse = x_recon_denorm[:k]
                    times_rmse = times[:k]
                else:
                    x_recon_denorm_rmse = x_recon_denorm
                    times_rmse = times

                rmse_loss_norm, rmse_dict, weighted_rmse_raw, valid_count = self.fuxi_rmse_interface.compute_batch_loss(
                    x_recon_denorm_rmse,
                    times_rmse,
                    requires_grad=True,
                )
            else:
                rmse_loss_norm = torch.tensor(0.0, device=self.device)
                rmse_dict = {ch: 0.0 for ch in self.fuxi_rmse_interface.target_channels}
                weighted_rmse_raw = 0.0
                valid_count = 0

            loss = recon_loss + self.grad_loss_weight * grad_loss + self.beta * kl_loss + self.channel_rmse_weight * rmse_loss_norm + self.domain_loss_weight * domain_loss

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                if self.is_master:
                    print(f"[Train] batch {batch_idx} loss is NaN/Inf, skip")
                continue

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)

            if isinstance(self.model, FSDP):
                FSDP.clip_grad_norm_(self.model, max_norm=5.0)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            self.scaler.step(self.opt)
            self.scaler.update()

            total_loss += float(loss.detach())
            total_recon_loss += float(recon_loss.detach())
            total_grad_loss += float(grad_loss.detach())
            total_channel_loss += float(rmse_loss_norm.detach())
            num_batches += 1

            if valid_count > 0:
                for ch in channel_sum:
                    channel_sum[ch] += rmse_dict.get(ch, 0.0)
                channel_n += 1

            if self.is_master:
                z500_raw = float(rmse_dict.get("z500", 0.0))
                postfix = {
                    "loss": f"{float(loss.detach()):.4f}",
                    "recon": f"{float(recon_loss.detach()):.4f}",
                    "channelrmse": f"{weighted_rmse_raw:.4f}",
                    "z500": f"{z500_raw:.4f}",
                }
                if self.using_dann and domain_logits is not None:
                    dacc = float((domain_logits.argmax(dim=1) == domains).float().mean())
                    postfix["Dacc"] = f"{dacc:.2f}"
                pbar.set_postfix(postfix)
                if batch_idx % 10 == 0 and self.writer is not None:
                    step = epoch * len(self.trainlo) + batch_idx
                    self.writer.add_scalar("Loss/batch/total", float(loss.detach()), step)
                    self.writer.add_scalar("Loss/batch/recon", float(recon_loss.detach()), step)
                    if self.use_grad_loss:
                        self.writer.add_scalar("Loss/batch/Gradloss", float(grad_loss.detach()), step)
                    self.writer.add_scalar("Align/batch/rmse_norm", float(rmse_loss_norm.detach()), step)
                    self.writer.add_scalar("Align/batch/weighted_rmse_raw", weighted_rmse_raw, step)
                    for ch, v in rmse_dict.items():
                        self.writer.add_scalar(f"Align/batch/channel_rmse/{ch}", v, step)
                    if self.using_dann:
                        self.writer.add_scalar("DANN/batch/domain_loss", float(domain_loss.detach()), step)
                        self.writer.add_scalar("DANN/batch/domain_acc", dacc, step)
                        self.writer.add_scalar("DANN/batch/grl_lambda", grl_lambda, step)

        if num_batches == 0:
            avg_loss = 0.0
            avg_recon = 0.0
        else:
            avg_loss, avg_recon = self._all_reduce_loss(total_loss, total_recon_loss, num_batches)

        avg_grad = total_grad_loss / max(num_batches, 1)
        avg_channel = total_channel_loss / max(num_batches, 1)

        avg_channel_dict = {ch: (channel_sum[ch] / max(channel_n, 1)) for ch in channel_sum}
        weighted_rmse_epoch = sum(
            self.fuxi_rmse_interface.channel_weights[ch] * avg_channel_dict[ch]
            for ch in self.fuxi_rmse_interface.target_channels
        )

        val_loss = self.validate_one_epoch(epoch)

        if isinstance(self.sch, ReduceLROnPlateau):
            self.sch.step(val_loss)
        else:
            self.sch.step()

        if self.is_master:
            print(
                f"\nEpoch {epoch+1} 训练集平均: 总损失={avg_loss:.5f}, "
                f"重建={avg_recon:.5f}, Grad={avg_grad:.5f}, AlignRMSE(norm)={avg_channel:.5f}, "
                f"AlignRMSE(raw)={weighted_rmse_epoch:.5f}"
            )
            if self.writer is not None:
                self.writer.add_scalar("Loss/train/total", avg_loss, epoch)
                self.writer.add_scalar("Loss/train/recon", avg_recon, epoch)
                if self.use_grad_loss:
                    self.writer.add_scalar("Loss/train/Gradloss", avg_grad, epoch)
                self.writer.add_scalar("Align/train/rmse_norm", avg_channel, epoch)
                self.writer.add_scalar("Align/train/weighted_rmse_raw", weighted_rmse_epoch, epoch)
                for ch, v in avg_channel_dict.items():
                    self.writer.add_scalar(f"Align/train/channel_rmse/{ch}", v, epoch)
                self.writer.add_scalar("hyper/lr", self.opt.param_groups[0]["lr"], epoch)

        return avg_loss, val_loss
