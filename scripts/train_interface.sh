#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# A2E train-only experiment entry: run experiment groups by number/name.
# =============================================================================
#
# List phases:
#   bash A2E/scripts/train_interface.sh list
#
# Preview train commands only:
#   bash A2E/scripts/train_interface.sh 1
#
# Execute train commands only:
#   RUN=1 bash A2E/scripts/train_interface.sh 1
#   RUN=1 bash A2E/scripts/train_interface.sh 1 2 3
#
# After training, run the matching evaluation phase separately:
#   RUN=1 bash A2E/scripts/eval_interface.sh 1
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
#   - A2Ec70_gfs_refnorm              [Full, reused from main]
#   - A2Ec70_ab_wo_fuxi               [L1 + Grad, no FuXi]
#   - A2Ec70_ab_wo_grad               [L1 + FuXi, no Grad]
#   - A2Ec70_ab_l1_only               [L1 only]
#   - A2Ec70_ab_wo_source_emb         [multi-source, no source embedding]
#
# 3 / data_scale
#   GFS-only random data-scale study from the training period:
#   - A2Ec70_gfs_data20_refnorm       [TRAIN_SAMPLE_RATIO=0.2]
#   - A2Ec70_gfs_data40_refnorm       [TRAIN_SAMPLE_RATIO=0.4]
#   - A2Ec70_gfs_data60_refnorm       [TRAIN_SAMPLE_RATIO=0.6]
#   - A2Ec70_gfs_data80_refnorm       [TRAIN_SAMPLE_RATIO=0.8]
#   - A2Ec70_gfs_refnorm              [100%, reused from main]
#
# 4 / source_mix
#   Source combination study:
#   - single-source models from main
#   - A2Ec70_gfs_cma_refnorm
#   - A2Ec70_gfs_hres_refnorm
#   - A2Ec70_cma_hres_refnorm
#   - A2Ec70_gfs_cma_hres_refnorm     [triple-source, reused from main]
#
# 5 / depth
#   Depth scaling only:
#   - A2Ec70_gfs_refnorm              [A2E-Lite, reused from main]
#   - A2Ec70_deep_refnorm             [A2E-Deep]
#
# 6 / parameter
#   Gradient loss weight and FuXi RMSE loss weight sensitivity, default source = GFS:
#   - A2Ec70_gradw_0p1                [GRAD_LOSS_WEIGHT=0.1]
#   - A2Ec70_gradw_0p2                [GRAD_LOSS_WEIGHT=0.2]
#   - A2Ec70_gfs_refnorm              [default grad=0.4, rmse weight=8e-3, reused from main]
#   - A2Ec70_gradw_0p8                [GRAD_LOSS_WEIGHT=0.8]
#   - A2Ec70_refnorm_w1em3            [CHANNEL_RMSE_WEIGHT=1e-3]
#   - A2Ec70_refnorm_w2em3            [CHANNEL_RMSE_WEIGHT=2e-3]
#   - A2Ec70_refnorm_w4em3            [CHANNEL_RMSE_WEIGHT=4e-3]
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
  sed -n '1,95p' "$0"
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
    echo "运行 bash A2E/scripts/train_interface.sh list 查看可用编号" >&2
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
