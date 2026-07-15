# A2E-c70 实验运行说明

本文档优先说明：**怎么运行各组实验、怎么单独运行某个实验、结果保存在哪里、如何评估和汇总指标**。

当前推荐入口是：

```bash
bash A2E/scripts/run_interface.sh
```

默认都是 dry-run，只打印命令，不真正执行。真正执行需要加：

```bash
RUN=1
```

---

## 1. 最常用命令

### 1.1 查看所有实验编号

```bash
bash A2E/scripts/run_interface.sh list
```

或者：

```bash
bash A2E/scripts/run_interface.sh
```

### 1.2 预览某组实验，不真正运行

```bash
bash A2E/scripts/run_interface.sh 1
```

例如预览主实验：

```bash
bash A2E/scripts/run_interface.sh main
```

### 1.3 真正运行某组实验

```bash
RUN=1 bash A2E/scripts/run_interface.sh 1
```

如果端口冲突，指定不同 `MASTER_PORT`：

```bash
MASTER_PORT=29517 RUN=1 bash A2E/scripts/run_interface.sh 1
```

### 1.4 一次运行多组实验

```bash
RUN=1 bash A2E/scripts/run_interface.sh 1 2 3
```

这会依次运行：

```text
main -> loss_ablation -> fuxi_loss
```

### 1.5 只跑 smoke test

```bash
MASTER_PORT=29517 RUN=1 bash A2E/scripts/run_interface.sh 0
```

### 1.6 只跑主实验

```bash
MASTER_PORT=29518 RUN=1 bash A2E/scripts/run_interface.sh 1
```

### 1.7 只跑 loss 消融

```bash
MASTER_PORT=29519 RUN=1 bash A2E/scripts/run_interface.sh 2
```

### 1.8 只跑模型规模 / 深度实验

```bash
MASTER_PORT=29520 RUN=1 bash A2E/scripts/run_interface.sh 5
```

`5 / scale` 会额外统计：

```text
params
FLOPs / MACs
A2E forward latency
```

---

## 2. 实验编号总表

| 编号 | 名称 | 内容 | 是否训练 | 是否评估 |
|---:|---|---|---|---|
| `0` | `smoke` | 快速流程测试 | 是 | 是 |
| `1` | `main` | 主实验：GFS/CMA/HRES 单源 + 三源联合 | 是 | 是 |
| `2` | `loss` / `loss_ablation` | L1 / Grad / FuXi loss 消融 | 是，复用 full baseline | 是 |
| `3` | `fuxi` / `fuxi_loss` | FuXi loss 形式消融 | 是，复用 reference baseline | 是 |
| `4` | `emb` / `embedding` | Time / Source embedding 消融 | 是，复用 multi-source full | 是 |
| `5` | `scale` / `scaling` / `depth` | 宽度规模 + 深度 scaling | 是，复用 A2E-Lite | 是，并 profile |
| `6` | `param` / `parameter` | 超参数敏感性 | 是 | 是 |
| `7` | `dual` / `dual_source` | 双源组合，可选 | 是 | 是 |
| `8` | `paper_min` | smoke + main + loss + fuxi + scale | 是 | 是 |
| `9` | `paper_full` | smoke + main + loss + fuxi + emb + scale + param | 是 | 是 |
| `a` | `all` | raw_note + 0 + 1 + 2 + 3 + 4 + 5 + 6 | 是 | 是 |
| `r` | `raw` / `raw_note` | Raw baseline 提醒 | 否 | 否 |

---

## 3. 推荐运行顺序

### Step 0：先 smoke

```bash
MASTER_PORT=29517 RUN=1 bash A2E/scripts/run_interface.sh 0
```

确认以下内容都能正常生成：

```text
train.log
best.pth
metrics/config.json
metrics/epoch_metrics.csv
eval/fuxi_rollout_metrics_summary.csv
eval/a2e_initial_metrics_summary.csv
```

### Step 1：主实验

```bash
MASTER_PORT=29518 RUN=1 bash A2E/scripts/run_interface.sh 1
```

训练并评估：

```text
A2Ec70_gfs_refnorm
A2Ec70_cma_refnorm
A2Ec70_hres_refnorm
A2Ec70_gfs_cma_hres_refnorm
```

### Step 2：核心 loss 消融

```bash
MASTER_PORT=29519 RUN=1 bash A2E/scripts/run_interface.sh 2
```

训练并评估：

```text
A2Ec70_gfs_refnorm          # full baseline，复用 main
A2Ec70_ab_wo_fuxi
A2Ec70_ab_wo_grad
A2Ec70_ab_l1_only
```

### Step 3：FuXi loss mode 消融

```bash
MASTER_PORT=29520 RUN=1 bash A2E/scripts/run_interface.sh 3
```

训练并评估：

```text
A2Ec70_fuxi_rawmean_w5e4
A2Ec70_gfs_refnorm          # reference_norm baseline，复用 main
```

### Step 4：模型规模 / 深度 scaling

```bash
MASTER_PORT=29521 RUN=1 bash A2E/scripts/run_interface.sh 5
```

训练、评估并 profile：

```text
A2Ec70_small_refnorm        # width small
A2Ec70_base_refnorm         # width base
A2Ec70_gfs_refnorm          # A2E-Lite，复用 main
A2Ec70_deep_refnorm         # A2E-Deep
```

### Step 5：embedding 消融

```bash
MASTER_PORT=29522 RUN=1 bash A2E/scripts/run_interface.sh 4
```

训练并评估：

```text
A2Ec70_gfs_cma_hres_refnorm # multi-source full，复用 main
A2Ec70_ms_wo_time_emb
A2Ec70_ms_wo_source_emb
```

### Step 6：参数敏感性

```bash
MASTER_PORT=29523 RUN=1 bash A2E/scripts/run_interface.sh 6
```

训练并评估：

```text
A2Ec70_gradw_0p1
A2Ec70_gradw_0p2
A2Ec70_gradw_0p8
A2Ec70_refnorm_w1em3
A2Ec70_refnorm_w2em3
A2Ec70_refnorm_w8em3
```

---

## 4. 各组实验具体内容

### 4.1 `0 / smoke`

```text
smoke_gfs_refnorm
```

配置：

```text
SOURCES=gfs
EPOCHS=1
VAL_SAMPLE_PER_MONTH=1
RMSE_EVERY_N_STEPS=10
FUXI_LOSS_MODE=reference_norm
CHANNEL_RMSE_WEIGHT=4e-3
```

评估：

```text
EVAL_SOURCES=gfs
EVAL_DATES=20250101
EVAL_VARIABLES=z500,t2m,t850,ws10,ws850,msl
```

---

### 4.2 `1 / main`

主实验包含四个子实验：

| EXP_NAME | 训练 sources | 说明 |
|---|---|---|
| `A2Ec70_gfs_refnorm` | `gfs` | GFS 单源 A2E-Lite / canonical single-source full |
| `A2Ec70_cma_refnorm` | `cma` | CMA 单源 |
| `A2Ec70_hres_refnorm` | `hres` | HRES 单源 |
| `A2Ec70_gfs_cma_hres_refnorm` | `gfs,cma,hres` | 三源联合 full |

默认训练配置：

```text
EPOCHS=90
FUXI_LOSS_MODE=reference_norm
CHANNEL_RMSE_WEIGHT=4e-3
DEPTH=0,0,1
RES_PER_STAGE=1,1,1
CHANNELS=384,768,1536
EMBED_DIM=384
```

主实验评估会对每个 checkpoint 测：

```text
EVAL_SOURCES=gfs,cma,hres
EVAL_DATES=20250101:20251122
EVAL_VARIABLES=z500,t2m,t850,ws10,ws850,msl
```

---

### 4.3 `2 / loss_ablation`

| EXP_NAME | 说明 |
|---|---|
| `A2Ec70_gfs_refnorm` | Full baseline，复用 main |
| `A2Ec70_ab_wo_fuxi` | L1 + Grad，无 FuXi downstream loss |
| `A2Ec70_ab_wo_grad` | L1 + FuXi，无 Grad loss |
| `A2Ec70_ab_l1_only` | L1 only |

全部基于：

```text
SOURCES=gfs
```

---

### 4.4 `3 / fuxi_loss`

| EXP_NAME | FuXi loss mode | Channel RMSE weight | 说明 |
|---|---|---:|---|
| `A2Ec70_fuxi_rawmean_w5e4` | `raw_mean` | `5e-4` | raw RMSE mean |
| `A2Ec70_gfs_refnorm` | `reference_norm` | `4e-3` | 推荐主方法，复用 main |

---

### 4.5 `4 / embedding`

| EXP_NAME | 训练 sources | Time Emb | Source Emb |
|---|---|---|---|
| `A2Ec70_gfs_cma_hres_refnorm` | `gfs,cma,hres` | 开 | 开 |
| `A2Ec70_ms_wo_time_emb` | `gfs,cma,hres` | 关 | 开 |
| `A2Ec70_ms_wo_source_emb` | `gfs,cma,hres` | 开 | 关 |

---

### 4.6 `5 / scale`

`scale` 包含 width scaling 和 depth scaling，全部基于：

```text
SOURCES=gfs
```

| EXP_NAME | 作用 | EMBED_DIM | CHANNELS | DEPTH | RES_PER_STAGE |
|---|---|---:|---|---|---|
| `A2Ec70_small_refnorm` | width small | 192 | `192,384,768` | `0,0,1` | `1,1,1` |
| `A2Ec70_base_refnorm` | width base | 256 | `256,512,1024` | `0,0,1` | `1,1,1` |
| `A2Ec70_gfs_refnorm` | A2E-Lite，复用 main | 384 | `384,768,1536` | `0,0,1` | `1,1,1` |
| `A2Ec70_deep_refnorm` | A2E-Deep | 384 | `384,768,1536` | `0,1,2` | `1,1,2` |

这个阶段会额外输出每个模型的：

```text
params
FLOPs / MACs
A2E forward latency
peak CUDA memory
```

---

### 4.7 `6 / parameter`

#### Grad loss weight

| EXP_NAME | GRAD_LOSS_WEIGHT |
|---|---:|
| `A2Ec70_gradw_0p1` | 0.1 |
| `A2Ec70_gradw_0p2` | 0.2 |
| `A2Ec70_gfs_refnorm` | 0.4，复用 main |
| `A2Ec70_gradw_0p8` | 0.8 |

#### FuXi reference_norm loss weight

| EXP_NAME | CHANNEL_RMSE_WEIGHT |
|---|---:|
| `A2Ec70_refnorm_w1em3` | 1e-3 |
| `A2Ec70_refnorm_w2em3` | 2e-3 |
| `A2Ec70_gfs_refnorm` | 4e-3，复用 main |
| `A2Ec70_refnorm_w8em3` | 8e-3 |

---

### 4.8 `7 / dual_source`

可选，不包含在 `all` 中。

| EXP_NAME | 训练 sources |
|---|---|
| `A2Ec70_gfs_cma_refnorm` | `gfs,cma` |
| `A2Ec70_gfs_hres_refnorm` | `gfs,hres` |
| `A2Ec70_cma_hres_refnorm` | `cma,hres` |

运行：

```bash
MASTER_PORT=29524 RUN=1 bash A2E/scripts/run_interface.sh 7
```

---

## 5. 单独运行某个训练实验

如果你不想通过编号组运行，也可以直接运行单个实验。

### 5.1 单独训练 GFS 主模型

```bash
MASTER_PORT=29517 RUN=1 bash A2E/scripts/run_one.sh A2Ec70_gfs_refnorm \
  SOURCES=gfs \
  EPOCHS=90 \
  FUXI_LOSS_MODE=reference_norm \
  CHANNEL_RMSE_WEIGHT=4e-3
```

### 5.2 单独训练 A2E-Deep

```bash
MASTER_PORT=29518 RUN=1 bash A2E/scripts/run_one.sh A2Ec70_deep_refnorm \
  SOURCES=gfs \
  EPOCHS=90 \
  EMBED_DIM=384 \
  CHANNELS=384,768,1536 \
  DEPTH=0,1,2 \
  RES_PER_STAGE=1,1,2 \
  FUXI_LOSS_MODE=reference_norm \
  CHANNEL_RMSE_WEIGHT=4e-3
```

### 5.3 单独训练一个 loss 消融

```bash
MASTER_PORT=29519 RUN=1 bash A2E/scripts/run_one.sh A2Ec70_ab_wo_fuxi \
  SOURCES=gfs \
  EPOCHS=90 \
  USE_GRAD_LOSS=true \
  GRAD_LOSS_WEIGHT=0.4 \
  FUXI_LOSS_MODE=reference_norm \
  CHANNEL_RMSE_WEIGHT=0
```

---

## 6. 单独评估某个实验

评估默认使用：

```text
CHECKPOINT_ROOT/EXP_NAME/best.pth
OUTPUT_ROOT/EXP_NAME/metrics/config.json
```

例如单独评估 GFS 主模型：

```bash
RUN=1 bash A2E/scripts/eval_one.sh A2Ec70_gfs_refnorm \
  EVAL_SOURCES=gfs \
  EVAL_DATES=20250101:20251122 \
  EVAL_VARIABLES=z500,t2m,t850,ws10,ws850,msl
```

如果要指定 checkpoint：

```bash
RUN=1 bash A2E/scripts/eval_one.sh A2Ec70_gfs_refnorm \
  CKPT=/path/to/checkpoint_epoch_50.pth \
  EVAL_SOURCES=gfs \
  EVAL_DATES=20250101,20250201,20250301
```

`EVAL_DATES` 支持两种格式：

```text
20250101,20250201,20250301      # 手动列日期
20250101:20251122              # 每天一个 init date
```

---

## 7. 保存结构

默认根目录在 [configs/common.sh](configs/common.sh)：

```bash
PROJECT_ROOT=/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/A2E/Formal
OUTPUT_ROOT=${PROJECT_ROOT}/experiments
CHECKPOINT_ROOT=${PROJECT_ROOT}/checkpoints
TENSORBOARD_ROOT=/home/ximutian/tensorboard_logs
PLOT_ROOT=${PROJECT_ROOT}/channelpics
```

以 `A2Ec70_gfs_refnorm` 为例。

### 7.1 训练输出

```text
${OUTPUT_ROOT}/A2Ec70_gfs_refnorm/
├── train.log
└── metrics/
    ├── launch_config.env
    ├── config.json
    └── epoch_metrics.csv
```

说明：

| 文件 | 内容 |
|---|---|
| `train.log` | 完整训练终端日志 |
| `metrics/launch_config.env` | shell 层启动配置，如 sources、epochs、depth、loss 权重等 |
| `metrics/config.json` | Python 实际训练配置，评估时用来重建模型 |
| `metrics/epoch_metrics.csv` | 每个 epoch 的 train/val loss、lr、耗时等 |

### 7.2 Checkpoint

```text
${CHECKPOINT_ROOT}/A2Ec70_gfs_refnorm/
└── best.pth
```

评估默认使用：

```text
${CHECKPOINT_ROOT}/EXP_NAME/best.pth
```

### 7.3 TensorBoard

```text
${TENSORBOARD_ROOT}/A2Ec70_gfs_refnorm/
```

### 7.4 可视化图片

```text
${PLOT_ROOT}/A2Ec70_gfs_refnorm/
```

### 7.5 评估输出

```text
${OUTPUT_ROOT}/A2Ec70_gfs_refnorm/eval/
├── eval.log
├── fuxi_rollout_metrics_detail.csv
├── fuxi_rollout_metrics_summary.csv
├── a2e_initial_metrics.csv
└── a2e_initial_metrics_summary.csv
```

| 文件 | 内容 |
|---|---|
| `eval.log` | 完整评估日志 |
| `fuxi_rollout_metrics_detail.csv` | 每天、每 source、每变量、每 lead step 的 RMSE/ACC |
| `fuxi_rollout_metrics_summary.csv` | 每 source、每变量、每 lead step 的平均 RMSE/ACC |
| `a2e_initial_metrics.csv` | A2E 初始场 vs ERA5 truth 的 L1/GradLoss/PSNR/SSIM 明细 |
| `a2e_initial_metrics_summary.csv` | 每 source、每变量的平均 PSNR/SSIM/L1/GradLoss |

### 7.6 scale 阶段额外 profile 输出

仅 `5 / scale` 阶段会生成：

```text
${OUTPUT_ROOT}/A2Ec70_gfs_refnorm/profile/
├── model_profile.json
└── model_profile.csv
```

内容包括：

```text
params
params_m
FLOPs / GFLOPs
MACs / GMACs
A2E forward latency mean/p50/p90
peak_cuda_memory_mb
```

---

## 8. 当前数据划分

当前默认：

```text
Train:
2022-01-01 00:00:00 到 2024-06-30 18:00:00

Validation:
2024-07-01 00:00:00 到 2024-12-31 18:00:00

Test / Eval init dates:
2025-01-01 到 2025-11-22，每天一个 init date
```

验证集抽样：

```bash
VAL_SAMPLE_PER_MONTH=7
VAL_SAMPLE_YEAR=2024
```

即约：

```text
6 months * 7 days/month * 4 times/day = 168 samples/source
```

HRES 默认只有 2024-2025，所以 HRES 划分为：

```text
HRES train: 2024-01-01 到 2024-06-30
HRES val:   2024-07-01 到 2024-12-31
HRES test:  2025-01-01 到 2025-11-22
```

---

## 9. 评估指标

### 9.1 FuXi rollout 指标

评估脚本：

```text
A2E/eval/eval_fuxi_rollout.py
```

流程：

```text
source(t0) -> A2E -> ERA5-like initial field
[ERA5(t0-6h), A2E(t0)] -> FuXi rollout
FuXi output(t0+6h ... t0+240h) vs ERA5 truth
```

输出：

```text
RMSE
ACC
```

按以下粒度汇总：

```text
experiment / source / variable / lead_step / lead_hours
```

### 9.2 A2E 初始场图像指标

A2E 直接输出与 ERA5 truth 比较，不经过 FuXi：

```text
A2E(source at t0) vs ERA5(t0)
```

输出：

```text
L1 loss
GradLoss
PSNR
SSIM
```

按以下粒度汇总：

```text
experiment / source / variable
```

### 9.3 默认评估变量

```text
z500,t2m,t850,ws10,ws850,msl
```

其中：

```text
ws10  = sqrt(u10m^2 + v10m^2)
ws850 = sqrt(u850^2 + v850^2)
```

---

## 10. tp 通道处理

当前 dataloader 会把 `tp` 通道置 0：

```text
x_tp = 0
y_tp = 0
```

原因：FuXi 初始场推理时 `tp` 会置 0，所以 A2E 不再学习 `tp` 初始场转换。张量仍然保持 70 通道，只是 `tp` 不贡献有效训练目标。

---

## 11. 常见问题

### Q1：评估是否使用 best checkpoint？

是。默认使用：

```text
CHECKPOINT_ROOT/EXP_NAME/best.pth
```

并使用：

```text
OUTPUT_ROOT/EXP_NAME/metrics/config.json
```

重建模型结构。

### Q2：端口占用怎么办？

如果看到：

```text
EADDRINUSE: address already in use, port 29500
```

换一个端口：

```bash
MASTER_PORT=29517 RUN=1 bash A2E/scripts/run_interface.sh 1
```

### Q3：`all` 包含 dual_source 吗？

不包含。`dual_source` 要单独跑：

```bash
RUN=1 bash A2E/scripts/run_interface.sh 7
```

### Q4：只想预览命令怎么办？

不加 `RUN=1` 即可：

```bash
bash A2E/scripts/run_interface.sh 1
```

### Q5：scale 的 FLOPs 准确吗？

FLOPs 通过 PyTorch profiler 估算：

```text
torch.profiler.profile(with_flops=True)
```

部分算子可能无法统计 FLOPs，因此 `model_profile.json` 中会保存说明。论文中可以写：

```text
FLOPs are estimated using PyTorch profiler for a single A2E forward pass.
```

---

## 12. 核心脚本列表

```text
A2E/configs/common.sh                    # 默认路径、数据划分、训练/评估/profile 配置
A2E/scripts/run_interface.sh             # 编号式一键入口
A2E/scripts/run_recommended_experiments.sh # 分组实验组织逻辑
A2E/scripts/run_one.sh                   # 单个训练实验
A2E/scripts/eval_one.sh                  # 单个实验评估
A2E/main_res_exp.py                      # 参数化训练入口
A2E/data/pairset.py                      # 训练/验证数据集；tp 通道置 0
A2E/eval/eval_fuxi_rollout.py            # FuXi rollout 评估 + A2E 初始场图像指标
A2E/eval/profile_a2e.py                  # scale 阶段模型 profile
A2E/fuxi_rmse_interface_new.py           # FuXi downstream loss 接口
```
