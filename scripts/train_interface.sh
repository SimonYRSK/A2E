#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
#  train-only experiment entry: no-GradLoss branch.
# =============================================================================
#
# This backup assumes the final objective is L1 + FuXi forecast-guided loss.
# GradLoss arguments and experiments are intentionally removed.
#
# List phases:
#   bash /scripts/train_interface.sh list
#
# Preview train commands only:
#   bash /scripts/train_interface.sh 1
#
# Execute train commands only:
#   RUN=1 bash /scripts/train_interface.sh 1
#   RUN=1 bash /scripts/train_interface.sh 1 2 3
#
# After training, run the matching evaluation phase separately:
#   RUN=1 bash /scripts/eval_interface.sh 1
#
# =============================================================================
# Single-experiment commands
# =============================================================================
#
# If you only want to train one minimal sub-experiment, run run_one.sh directly.
# NOTE: run_one.sh executes immediately; it does not use RUN=1 dry-run logic.
#
# Smoke:
#   bash /scripts/run_one.sh smoke_gfs_refnorm \
#     SOURCES=gfs EPOCHS=1 VAL_SAMPLE_PER_MONTH=1 RMSE_EVERY_N_STEPS=1 \
#     FUXI_LOSS_MODE=reference_norm CHANNEL_RMSE_WEIGHT=8e-3
#
# Main single-source / multi-source:
#   bash /scripts/run_one.sh A2Ec70_gfs_refnorm \
#     SOURCES=gfs EPOCHS=90 FUXI_LOSS_MODE=reference_norm CHANNEL_RMSE_WEIGHT=8e-3
#   bash /scripts/run_one.sh A2Ec70_cma_refnorm \
#     SOURCES=cma EPOCHS=90 FUXI_LOSS_MODE=reference_norm CHANNEL_RMSE_WEIGHT=8e-3
#   bash /scripts/run_one.sh A2Ec70_hres_refnorm \
#     SOURCES=hres EPOCHS=90 FUXI_LOSS_MODE=reference_norm CHANNEL_RMSE_WEIGHT=8e-3
#   bash /scripts/run_one.sh A2Ec70_gfs_cma_hres_refnorm \
#     SOURCES=gfs,cma,hres EPOCHS=90 FUXI_LOSS_MODE=reference_norm CHANNEL_RMSE_WEIGHT=8e-3
#
# Core ablation:
#   bash /scripts/run_one.sh A2Ec70_ab_wo_fuxi \
#     SOURCES=gfs EPOCHS=90 FUXI_LOSS_MODE=reference_norm CHANNEL_RMSE_WEIGHT=0
#   bash /scripts/run_one.sh A2Ec70_ab_l1_only \
#     SOURCES=gfs EPOCHS=90 FUXI_LOSS_MODE=reference_norm CHANNEL_RMSE_WEIGHT=0
#   bash /scripts/run_one.sh A2Ec70_ab_wo_source_emb \
#     SOURCES=gfs,cma,hres EPOCHS=90 USING_TIME_EMBEDDING=true USING_SOURCE_EMBEDDING=false \
#     FUXI_LOSS_MODE=reference_norm CHANNEL_RMSE_WEIGHT=8e-3
#
# Data-scale study:
#   bash /scripts/run_one.sh A2Ec70_gfs_data25_refnorm \
#     SOURCES=gfs EPOCHS=90 TRAIN_SAMPLE_RATIO=0.25 FUXI_LOSS_MODE=reference_norm CHANNEL_RMSE_WEIGHT=8e-3
#   bash /scripts/run_one.sh A2Ec70_gfs_data50_refnorm \
#     SOURCES=gfs EPOCHS=90 TRAIN_SAMPLE_RATIO=0.5 FUXI_LOSS_MODE=reference_norm CHANNEL_RMSE_WEIGHT=8e-3
#
# Source-mix study:
#   bash /scripts/run_one.sh A2Ec70_gfs_cma_refnorm \
#     SOURCES=gfs,cma EPOCHS=90 FUXI_LOSS_MODE=reference_norm CHANNEL_RMSE_WEIGHT=8e-3
#   bash /scripts/run_one.sh A2Ec70_gfs_hres_refnorm \
#     SOURCES=gfs,hres EPOCHS=90 FUXI_LOSS_MODE=reference_norm CHANNEL_RMSE_WEIGHT=8e-3
#   bash /scripts/run_one.sh A2Ec70_cma_hres_refnorm \
#     SOURCES=cma,hres EPOCHS=90 FUXI_LOSS_MODE=reference_norm CHANNEL_RMSE_WEIGHT=8e-3
#
# Depth scaling:
#   bash /scripts/run_one.sh A2Ec70_mid_refnorm \
#     SOURCES=gfs EPOCHS=90 EMBED_DIM=384 CHANNELS=384,768,1536 DEPTH=0,0,2 RES_PER_STAGE=2,2,2 \
#     FUXI_LOSS_MODE=reference_norm CHANNEL_RMSE_WEIGHT=8e-3
#   bash /scripts/run_one.sh A2Ec70_deep_refnorm \
#     SOURCES=gfs EPOCHS=90 EMBED_DIM=384 CHANNELS=384,768,1536 DEPTH=0,1,2 RES_PER_STAGE=3,3,3 \
#     FUXI_LOSS_MODE=reference_norm CHANNEL_RMSE_WEIGHT=8e-3
#
# FuXi reference_norm weight sensitivity:
#   bash /scripts/run_one.sh A2Ec70_refnorm_w2em3 \
#     SOURCES=gfs EPOCHS=90 FUXI_LOSS_MODE=reference_norm CHANNEL_RMSE_WEIGHT=2e-3
#   bash /scripts/run_one.sh A2Ec70_refnorm_w4em3 \
#     SOURCES=gfs EPOCHS=90 FUXI_LOSS_MODE=reference_norm CHANNEL_RMSE_WEIGHT=4e-3
#   bash /scripts/run_one.sh A2Ec70_refnorm_w1em2 \
#     SOURCES=gfs EPOCHS=90 FUXI_LOSS_MODE=reference_norm CHANNEL_RMSE_WEIGHT=1e-2
#
# =============================================================================
# Phase table
# =============================================================================
#
# 0 / smoke
#   - smoke_gfs_refnorm
#
# 1 / main
#   - A2Ec70_gfs_refnorm              [GFS-only]
#   - A2Ec70_cma_refnorm              [CMA-only]
#   - A2Ec70_hres_refnorm             [HRES-only]
#   - A2Ec70_gfs_cma_hres_refnorm     [GFS+CMA+HRES]
#
# 2 / loss
#   - A2Ec70_gfs_refnorm              [Final: L1 + FuXi, reused from main]
#   - A2Ec70_ab_wo_fuxi               [L1 only / no FuXi]
#   - A2Ec70_ab_wo_source_emb         [multi-source, no source embedding]
#
# 3 / data_scale
#   - A2Ec70_gfs_data25_refnorm       [TRAIN_SAMPLE_RATIO=0.25]
#   - A2Ec70_gfs_data50_refnorm       [TRAIN_SAMPLE_RATIO=0.5]
#   - A2Ec70_gfs_refnorm              [100%, reused from main]
#
# 4 / source_mix
#   - single-source models from main
#   - A2Ec70_gfs_cma_refnorm
#   - A2Ec70_gfs_hres_refnorm
#   - A2Ec70_cma_hres_refnorm
#   - A2Ec70_gfs_cma_hres_refnorm     [triple-source, reused from main]
#
# 5 / depth
#   - A2Ec70_gfs_refnorm              [A2E-c70-Lite, reused from main]
#   - A2Ec70_mid_refnorm              [A2E-c70-Mid]
#   - A2Ec70_deep_refnorm             [A2E-c70-Deep]
#
# 6 / parameter
#   - A2Ec70_refnorm_w2em3            [CHANNEL_RMSE_WEIGHT=2e-3]
#   - A2Ec70_refnorm_w4em3            [CHANNEL_RMSE_WEIGHT=4e-3]
#   - A2Ec70_gfs_refnorm              [default rmse weight=8e-3, reused from main]
#   - A2Ec70_refnorm_w1em2            [CHANNEL_RMSE_WEIGHT=1e-2]
#
# 8 / paper_min
#   0 -> 1 -> 2 -> 3 -> 5
#
# 9 / paper_full
#   0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6
#
# a / all
#   raw_note + 0 + 1 + 2 + 3 + 4 + 5 + 6
#
# r / raw
#   print Raw baseline note only
#
# =============================================================================

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
A2E_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
RECOMMENDED_SCRIPT="${A2E_ROOT}/scripts/run_recommended_experiments.sh"
RUN=${RUN:-0}

show_list() {
  sed -n '1,190p' "$0"
}

phase_name() {
  case "$1" in
    0|smoke) echo smoke ;;
    1|main) echo main ;;
    2|loss|loss_ablation) echo loss_ablation ;;
    3|data|datascale|data_scale) echo data_scale ;;
    4|mix|source_mix|dual|dual_source) echo source_mix ;;
    5|depth|depth_scaling|scale|scaling) echo depth ;;
    6|param|parameter) echo parameter ;;
    r|raw|raw_note) echo raw_note ;;
    a|all) echo all ;;
    *) echo "" ;;
  esac
}

run_phase() {
  local key=$1
  local phase
  phase=$(phase_name "$key")
  if [[ -z "$phase" ]]; then
    echo "未知实验编号/名称: $key" >&2
    echo "运行 bash /scripts/train_interface.sh list 查看可用编号" >&2
    exit 2
  fi
  echo
  echo "======================================================================"
  echo "训练实验组: $key -> $phase"
  echo "ACTION=train RUN=$RUN bash $RECOMMENDED_SCRIPT $phase"
  echo "======================================================================"
  ACTION=train RUN="$RUN" bash "$RECOMMENDED_SCRIPT" "$phase"
}

run_paper_min() {
  echo "将训练最小论文集合: 0(smoke) -> 1(main) -> 2(loss) -> 3(data_scale) -> 5(depth)"
  run_phase 0
  run_phase 1
  run_phase 2
  run_phase 3
  run_phase 5
}

run_paper_full() {
  echo "将训练完整论文集合: 0(smoke) -> 1(main) -> 2(loss) -> 3(data_scale) -> 4(source_mix) -> 5(depth) -> 6(parameter)"
  run_phase 0
  run_phase 1
  run_phase 2
  run_phase 3
  run_phase 4
  run_phase 5
  run_phase 6
}

if [[ $# -eq 0 || "${1:-}" == "list" || "${1:-}" == "help" || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  show_list
  exit 0
fi

echo "RUN=$RUN"
if [[ "$RUN" != "1" ]]; then
  echo "当前是训练预览模式，不会真正运行。真正训练请加：RUN=1"
else
  echo "当前是真正训练模式。"
fi

echo "请求训练: $*"

for key in "$@"; do
  case "$key" in
    8|paper_min) run_paper_min ;;
    9|paper_full) run_paper_full ;;
    *) run_phase "$key" ;;
  esac
done
