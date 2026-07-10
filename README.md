# A2E-c70 实验调度说明

本文档说明当前 A2E-c70 实验脚本的组织方式、输出目录、推荐运行顺序，以及训练/评测结果会保存到哪里。

---

## 1. 当前新增的核心文件

```text
A2E/
├── configs/
│   └── common.sh                         # 统一实验默认配置
├── scripts/
│   ├── run_one.sh                        # 启动单个训练实验
│   ├── eval_one.sh                       # 评测单个实验 checkpoint
│   └── run_recommended_experiments.sh    # 按推荐顺序组织整套实验
├── eval/
│   └── eval_a2e_fields.py                # 计算场映射 RMSE/ACC
├── trainers/
│   └── fsdptrain_align_metrics.py        # trainer 副本，额外保存 CSV/JSON 指标
├── main_res_exp.py                       # 参数化训练入口，不改原 main_res.py
└── fuxi_rmse_interface_new.py            # 支持 manual_weighted/raw_mean/reference_norm
```

原始文件仍然保留：

```text
main_res.py
trainers/fsdptrain_align.py
```

当前推荐使用：

```text
main_res_exp.py
scripts/run_one.sh
scripts/eval_one.sh
scripts/run_recommended_experiments.sh
```

---

## 2. 实验名 EXP_NAME 是核心索引

每个实验都由一个实验名 `EXP_NAME` 区分，例如：

```text
A2Ec70_cma_refnorm
A2Ec70_ab_wo_fuxi
A2Ec70_fuxi_refnorm_w4e3
A2Ec70_small_refnorm
```

训练和评测结果都会根据这个实验名保存到对应目录。

例如运行：

```bash
bash A2E/scripts/run_one.sh A2Ec70_cma_refnorm SOURCES=cma
```

那么该实验的输出会保存在：

```text
OUTPUT_ROOT/A2Ec70_cma_refnorm/
CHECKPOINT_ROOT/A2Ec70_cma_refnorm/
TENSORBOARD_ROOT/A2Ec70_cma_refnorm/
PLOT_ROOT/A2Ec70_cma_refnorm/
```

注意：这里的 `OUTPUT_ROOT`、`CHECKPOINT_ROOT`、`TENSORBOARD_ROOT`、`PLOT_ROOT` 不是固定等于本地 `E:/myrepo/A2E`，而是由 `configs/common.sh` 定义。

默认配置在 `configs/common.sh` 中：

```bash
OUTPUT_ROOT=${PROJECT_ROOT}/experiments
CHECKPOINT_ROOT=${PROJECT_ROOT}/checkpoints
TENSORBOARD_ROOT=/home/ximutian/tensorboard_logs
PLOT_ROOT=${PROJECT_ROOT}/channelpics
```

如果你想临时改输出位置，可以运行时覆盖：

```bash
OUTPUT_ROOT=/some/path/experiments \
CHECKPOINT_ROOT=/some/path/checkpoints \
bash A2E/scripts/run_one.sh A2Ec70_test SOURCES=cma
```

---

## 3. `common.sh` 是统一默认配置

文件：

```text
A2E/configs/common.sh
```

这里保存所有实验的公共默认配置，包括：

### 数据路径

```bash
ERA5_PATH
GFS_PATH
HRES_PATH
CMA_PATH
FUXI_DIR
CLIM_PATH
```

### 数据划分

```bash
TRAIN_START=2022-01-01 00:00:00
TRAIN_END=2024-12-31 18:00:00
VAL_START=2025-01-01 00:00:00
VAL_END=2025-11-20 18:00:00
```

其中 HRES 默认只有 2024 年训练数据：

```bash
HRES_TRAIN_START=2024-01-01 00:00:00
HRES_TRAIN_END=2024-12-31 18:00:00
```

### 模型配置

```bash
CHANNELS=384,768,1536
EMBED_DIM=384
DEPTH=0,0,1
RES_PER_STAGE=1,1,1
USING_TIME_EMBEDDING=true
USING_SOURCE_EMBEDDING=true
```

### 训练配置

根据当前主实验现象，默认 epoch 改成 90：

```bash
EPOCHS=90
BATCH_SIZE=8
BASE_LR=2e-4
MIN_LR=1e-7
WARMUP_EPOCHS=5
WEIGHT_DECAY=2e-5
```

### Loss 配置

默认主方法为 FuXi-ERA5 reference normalization：

```bash
FUXI_LOSS_MODE=reference_norm
CHANNEL_RMSE_WEIGHT=4e-3
USE_GRAD_LOSS=true
GRAD_LOSS_WEIGHT=0.4
```

### 评测变量

导师提到的 `Q700` 已修正为 `r700`：

```bash
EVAL_VARIABLES=z500,t2m,tp,ws10m,msl,r700
```

其中：

```text
ws10m = sqrt(u10m^2 + v10m^2)
```

---

## 4. `run_one.sh`：启动单个训练实验

文件：

```text
A2E/scripts/run_one.sh
```

用法：

```bash
bash A2E/scripts/run_one.sh EXP_NAME KEY=VALUE KEY=VALUE ...
```

示例：

```bash
bash A2E/scripts/run_one.sh A2Ec70_cma_refnorm \
  SOURCES=cma \
  EPOCHS=90 \
  FUXI_LOSS_MODE=reference_norm \
  CHANNEL_RMSE_WEIGHT=4e-3
```

`run_one.sh` 的执行流程：

```text
1. source configs/common.sh，加载统一默认配置
2. 读取 EXP_NAME
3. 读取 KEY=VALUE 覆盖项
4. 创建输出目录
5. 保存 shell 启动配置 launch_config.env
6. torchrun 启动 main_res_exp.py
7. 保存训练日志 train.log
```

---

## 5. 单个训练实验会保存什么？

假设：

```text
EXP_NAME=A2Ec70_cma_refnorm
```

训练后会保存：

```text
OUTPUT_ROOT/A2Ec70_cma_refnorm/
├── train.log
└── metrics/
    ├── launch_config.env
    ├── config.json
    └── epoch_metrics.csv

CHECKPOINT_ROOT/A2Ec70_cma_refnorm/
└── best.pth

TENSORBOARD_ROOT/A2Ec70_cma_refnorm/
└── events.out.tfevents...

PLOT_ROOT/A2Ec70_cma_refnorm/
└── ...
```

### `metrics/launch_config.env`

由 `run_one.sh` 保存，记录 shell 层面的启动变量，例如：

```text
SOURCES=cma
EPOCHS=90
CHANNELS=384,768,1536
FUXI_LOSS_MODE=reference_norm
CHANNEL_RMSE_WEIGHT=4e-3
```

### `metrics/config.json`

由 `main_res_exp.py` 和 `fsdptrain_align_metrics.py` 保存。

它记录 Python 训练代码实际使用的最终配置，包括：

```text
数据路径
训练/验证时间段
sources
模型结构
训练超参数
loss 配置
FuXi reference RMSE
checkpoint/tensorboard/metrics 路径
```

这个文件用于复现实验。

### `metrics/epoch_metrics.csv`

由 `fsdptrain_align_metrics.py` 每个 epoch 追加一行。

字段包括：

```text
epoch
train_loss
val_loss
lr
seconds
fuxi_loss_mode
channel_rmse_weight
grad_loss_weight
use_grad_loss
sources
channels
depth
epochs
```

这个文件用于汇总训练过程、画收敛曲线、统计训练时间。

### `train.log`

保存完整终端训练输出。

### `best.pth`

保存验证集 loss 最优的 checkpoint。

### TensorBoard

仍然会有 TensorBoard。

默认路径：

```text
TENSORBOARD_ROOT/EXP_NAME/
```

例如：

```text
/home/ximutian/tensorboard_logs/A2Ec70_cma_refnorm/
```

可以用：

```bash
tensorboard --logdir /home/ximutian/tensorboard_logs
```

查看所有实验。

---

## 6. `eval_one.sh`：评测单个实验

文件：

```text
A2E/scripts/eval_one.sh
```

用法：

```bash
bash A2E/scripts/eval_one.sh EXP_NAME KEY=VALUE ...
```

示例：

```bash
bash A2E/scripts/eval_one.sh A2Ec70_cma_refnorm \
  EVAL_SOURCES=cma \
  EVAL_DATES=20250101,20250105 \
  EVAL_VARIABLES=z500,t2m,tp,ws10m,msl,r700
```

默认读取：

```text
CHECKPOINT_ROOT/EXP_NAME/best.pth
OUTPUT_ROOT/EXP_NAME/metrics/config.json
```

如果要指定 checkpoint：

```bash
bash A2E/scripts/eval_one.sh A2Ec70_cma_refnorm \
  CKPT=/path/to/best.pth \
  EVAL_SOURCES=cma
```

评测输出目录：

```text
OUTPUT_ROOT/EXP_NAME/eval/
```

输出文件：

```text
OUTPUT_ROOT/EXP_NAME/eval/
├── eval.log
├── field_metrics_detail.csv
└── field_metrics_summary.csv
```

### `field_metrics_detail.csv`

逐时间、逐 source、逐变量保存：

```text
experiment
source
time
variable
rmse
acc
```

例如：

```text
A2Ec70_cma_refnorm,cma,2025-01-01,z500,xxx,xxx
A2Ec70_cma_refnorm,cma,2025-01-01,t2m,xxx,xxx
A2Ec70_cma_refnorm,cma,2025-01-01,r700,xxx,xxx
```

### `field_metrics_summary.csv`

按 source 和变量汇总平均：

```text
experiment
source
variable
rmse_mean
acc_mean
n_rmse
n_acc
```

---

## 7. `eval_one.sh` 是否和 `run_recommended_experiments.sh` 结合了？

是，已经结合。

文件：

```text
A2E/scripts/run_recommended_experiments.sh
```

里面定义了两个函数：

```bash
run_train() {
  bash A2E/scripts/run_one.sh EXP_NAME ...
}

run_eval() {
  bash A2E/scripts/eval_one.sh EXP_NAME ...
}
```

在推荐实验的每个阶段，训练结束后会调用对应的评测。

例如 main 阶段中会先训练：

```text
A2Ec70_gfs_refnorm
A2Ec70_cma_refnorm
A2Ec70_hres_refnorm
A2Ec70_gfs_cma_hres_refnorm
```

然后会对这些实验调用：

```bash
bash A2E/scripts/eval_one.sh EXP_NAME \
  EVAL_SOURCES=gfs,cma,hres \
  EVAL_DATES=${EVAL_DATES} \
  EVAL_VARIABLES=z500,t2m,tp,ws10m,msl,r700
```

所以：

```text
run_recommended_experiments.sh = 组织总流程
run_one.sh = 单个训练
eval_one.sh = 单个评测
```

---

## 8. `run_recommended_experiments.sh`：推荐实验顺序

文件：

```text
A2E/scripts/run_recommended_experiments.sh
```

默认 dry-run，只打印命令，不执行。

查看所有命令：

```bash
bash A2E/scripts/run_recommended_experiments.sh all
```

真正执行：

```bash
RUN=1 bash A2E/scripts/run_recommended_experiments.sh all
```

推荐不要一开始直接跑 `all`，而是分阶段跑。

---

## 9. 推荐实际执行顺序

### Step 0：smoke test

```bash
RUN=1 bash A2E/scripts/run_recommended_experiments.sh smoke
```

确认：

```text
训练能跑
checkpoint 能保存
config.json 能保存
epoch_metrics.csv 能保存
eval 能跑
field_metrics_summary.csv 能保存
```

---

### Step 1：主实验

```bash
RUN=1 bash A2E/scripts/run_recommended_experiments.sh main
```

包括：

```text
A2Ec70_gfs_refnorm
A2Ec70_cma_refnorm
A2Ec70_hres_refnorm
A2Ec70_gfs_cma_hres_refnorm
```

---

### Step 2：核心 loss 消融

```bash
RUN=1 bash A2E/scripts/run_recommended_experiments.sh loss_ablation
```

包括：

```text
A2Ec70_ab_full_refnorm
A2Ec70_ab_wo_fuxi
A2Ec70_ab_wo_grad
A2Ec70_ab_l1_only
```

---

### Step 3：FuXi loss mode 消融

```bash
RUN=1 bash A2E/scripts/run_recommended_experiments.sh fuxi_loss
```

包括：

```text
A2Ec70_fuxi_manual_w1e3
A2Ec70_fuxi_rawmean_w5e4
A2Ec70_fuxi_refnorm_w4e3
```

三种模式对应：

```text
manual_weighted
raw_mean
reference_norm
```

---

### Step 4：embedding 消融

```bash
RUN=1 bash A2E/scripts/run_recommended_experiments.sh embedding
```

包括：

```text
A2Ec70_ms_full_embed
A2Ec70_ms_wo_time_emb
A2Ec70_ms_wo_source_emb
```

注意：source embedding 消融放在多源训练上更合理。

---

### Step 5：model scaling

```bash
RUN=1 bash A2E/scripts/run_recommended_experiments.sh scaling
```

包括：

```text
A2Ec70_small_refnorm
A2Ec70_base_refnorm
A2Ec70_full_refnorm
```

---

### Step 6：参数敏感性

```bash
RUN=1 bash A2E/scripts/run_recommended_experiments.sh parameter
```

包括：

```text
grad_loss_weight = 0.1, 0.2, 0.4, 0.8
channel_rmse_weight = 1e-3, 2e-3, 4e-3, 8e-3
```

---

## 10. FuXi loss 三种模式

在 `fuxi_rmse_interface_new.py` 中支持：

```text
manual_weighted
raw_mean
reference_norm
```

### `manual_weighted`

当前历史方法：

```text
sum(channel_weight[ch] * rmse_ch)
```

推荐搭配：

```bash
CHANNEL_RMSE_WEIGHT=1e-3
```

### `raw_mean`

直接平均各通道 raw RMSE：

```text
mean(rmse_ch)
```

推荐搭配：

```bash
CHANNEL_RMSE_WEIGHT=5e-4
```

### `reference_norm`

每个通道除以 FuXi-ERA5 reference RMSE 后平均：

```text
mean(rmse_ch / reference_rmse_ch)
```

推荐搭配：

```bash
CHANNEL_RMSE_WEIGHT=4e-3
```

---

## 11. 当前推荐主方法配置

```bash
SOURCES=cma
EPOCHS=90
FUXI_LOSS_MODE=reference_norm
CHANNEL_RMSE_WEIGHT=4e-3
USE_GRAD_LOSS=true
GRAD_LOSS_WEIGHT=0.4
CHANNELS=384,768,1536
EMBED_DIM=384
DEPTH=0,0,1
USING_TIME_EMBEDDING=true
USING_SOURCE_EMBEDDING=true
```

对应命令：

```bash
RUN=1 bash A2E/scripts/run_one.sh A2Ec70_cma_refnorm \
  SOURCES=cma \
  EPOCHS=90 \
  FUXI_LOSS_MODE=reference_norm \
  CHANNEL_RMSE_WEIGHT=4e-3
```

评测：

```bash
RUN=1 bash A2E/scripts/eval_one.sh A2Ec70_cma_refnorm \
  EVAL_SOURCES=cma \
  EVAL_DATES=20250101,20250105 \
  EVAL_VARIABLES=z500,t2m,tp,ws10m,msl,r700
```

---

## 12. 常见问题

### Q1：结果是不是会按实验名保存？

是。

实验名 `EXP_NAME` 是所有输出的索引。

例如：

```text
A2Ec70_cma_refnorm
```

会对应：

```text
OUTPUT_ROOT/A2Ec70_cma_refnorm/
CHECKPOINT_ROOT/A2Ec70_cma_refnorm/
TENSORBOARD_ROOT/A2Ec70_cma_refnorm/
PLOT_ROOT/A2Ec70_cma_refnorm/
```

---

### Q2：会不会还有 TensorBoard？

会。

原 trainer 的 TensorBoard 写入仍然保留。

新增的 CSV/JSON 只是为了方便论文表格和批量汇总，不替代 TensorBoard。

---

### Q3：`run_recommended_experiments.sh` 会自动评测吗？

会。

每个 phase 内部训练后会调用 `eval_one.sh`。

但默认是 dry-run。必须加：

```bash
RUN=1
```

才会真正执行。

---

### Q4：`all` 会跑双源实验吗？

默认不会。

`all` 包括：

```text
smoke
main
loss_ablation
fuxi_loss
embedding
scaling
parameter
```

双源实验要单独跑：

```bash
RUN=1 bash A2E/scripts/run_recommended_experiments.sh dual_source
```

---

### Q5：Raw GFS/CMA/HRES baseline 在哪里？

Checklist 中写 Raw baseline 已完成。

当前脚本主要负责 A2E 训练和 A2E checkpoint 评测。Raw source-vs-ERA5 baseline 最好后续再写一个独立 raw-eval 脚本，输出同样格式：

```text
z500,t2m,tp,ws10m,msl,r700
```

这样可以和 A2E 结果放进同一张表。

---

## 13. 建议最终目录结构

一个完整实验最终大概是：

```text
OUTPUT_ROOT/A2Ec70_cma_refnorm/
├── train.log
├── metrics/
│   ├── launch_config.env
│   ├── config.json
│   └── epoch_metrics.csv
└── eval/
    ├── eval.log
    ├── field_metrics_detail.csv
    └── field_metrics_summary.csv

CHECKPOINT_ROOT/A2Ec70_cma_refnorm/
└── best.pth

TENSORBOARD_ROOT/A2Ec70_cma_refnorm/
└── events.out.tfevents...

PLOT_ROOT/A2Ec70_cma_refnorm/
└── ...
```

---

## 14. 最推荐的第一组命令

```bash
# 1. 先检查所有命令，不执行
bash A2E/scripts/run_recommended_experiments.sh smoke

# 2. 真正跑 smoke test
RUN=1 bash A2E/scripts/run_recommended_experiments.sh smoke

# 3. smoke 成功后，跑主实验
RUN=1 bash A2E/scripts/run_recommended_experiments.sh main

# 4. 主实验稳定后，跑核心消融
RUN=1 bash A2E/scripts/run_recommended_experiments.sh loss_ablation

# 5. 跑 FuXi loss mode 消融
RUN=1 bash A2E/scripts/run_recommended_experiments.sh fuxi_loss
```
