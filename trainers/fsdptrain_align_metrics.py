import csv
import json
import os
import time
from typing import Any, Dict, Optional

from trainers.fsdptrain_align import FSDPUNetAlignTrainer


class FSDPUNetAlignMetricsTrainer(FSDPUNetAlignTrainer):
    """FSDPUNetAlignTrainer copy with lightweight CSV/JSON metric output.

    TensorBoard remains the detailed training monitor. This subclass additionally
    writes an epoch-level CSV so experiments launched from shell scripts are easy
    to aggregate for paper tables.
    """

    def __init__(self, *args, metrics_dir: Optional[str] = None, exp_config: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_dir = metrics_dir
        self.exp_config = exp_config or {}

        if self.metrics_dir and self.is_master:
            os.makedirs(self.metrics_dir, exist_ok=True)
            config_path = os.path.join(self.metrics_dir, "config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(self.exp_config, f, indent=2, ensure_ascii=False)
            print(f"[Metrics] config saved to: {config_path}")

    def _append_epoch_metrics(self, row: Dict[str, Any]):
        if not self.metrics_dir or not self.is_master:
            return

        path = os.path.join(self.metrics_dir, "epoch_metrics.csv")
        file_exists = os.path.exists(path)
        fieldnames = [
            "epoch",
            "train_loss",
            "val_loss",
            "lr",
            "seconds",
            "fuxi_loss_mode",
            "channel_rmse_weight",
            "grad_loss_weight",
            "use_grad_loss",
            "sources",
            "channels",
            "depth",
            "epochs",
        ]
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    def train_one_epoch(self, epoch):
        t0 = time.time()
        avg_loss, val_loss = super().train_one_epoch(epoch)
        elapsed = time.time() - t0

        loss_mode = getattr(self.fuxi_rmse_interface, "loss_mode", "manual_weighted")
        self._append_epoch_metrics(
            {
                "epoch": int(epoch) + 1,
                "train_loss": float(avg_loss),
                "val_loss": float(val_loss),
                "lr": float(self.opt.param_groups[0]["lr"]),
                "seconds": float(elapsed),
                "fuxi_loss_mode": loss_mode,
                "channel_rmse_weight": float(self.channel_rmse_weight),
                "grad_loss_weight": float(self.grad_loss_weight),
                "use_grad_loss": bool(self.use_grad_loss),
                "sources": ",".join(self.exp_config.get("sources", [])),
                "channels": ",".join(map(str, self.exp_config.get("model", {}).get("channels", []))),
                "depth": ",".join(map(str, self.exp_config.get("model", {}).get("depth", []))),
                "epochs": self.exp_config.get("training", {}).get("epochs", ""),
            }
        )
        return avg_loss, val_loss
