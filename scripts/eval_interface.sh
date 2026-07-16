#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# A2E eval-only experiment entry: evaluate experiment groups by number/name.
# =============================================================================
#
# List phases:
#   bash A2E/scripts/eval_interface.sh list
#
# Preview eval commands only:
#   bash A2E/scripts/eval_interface.sh 1
#
# Execute eval commands only:
#   RUN=1 bash A2E/scripts/eval_interface.sh 1
#
# This assumes checkpoints already exist under CHECKPOINT_ROOT/EXP_NAME/best.pth.
#
# =============================================================================
# Per-phase eval commands expanded
# =============================================================================
#
# 0 / smoke
#   bash A2E/scripts/eval_one.sh smoke_gfs_refnorm \
#     EVAL_SOURCES=gfs \
#     EVAL_DATES=20250101 \
#     EVAL_VARIABLES=z500,t2m,t850,ws10,ws850,msl
#
# 1 / main
#   bash A2E/scripts/eval_one.sh A2Ec70_gfs_refnorm \
#     EVAL_SOURCES=gfs,cma,hres \
#     EVAL_DATES=${EVAL_DATES} \
#     EVAL_VARIABLES=z500,t2m,t850,ws10,ws850,msl
#   bash A2E/scripts/eval_one.sh A2Ec70_cma_refnorm \
#     EVAL_SOURCES=gfs,cma,hres \
#     EVAL_DATES=${EVAL_DATES} \
#     EVAL_VARIABLES=z500,t2m,t850,ws10,ws850,msl
#   bash A2E/scripts/eval_one.sh A2Ec70_hres_refnorm \
#     EVAL_SOURCES=gfs,cma,hres \
#     EVAL_DATES=${EVAL_DATES} \
#     EVAL_VARIABLES=z500,t2m,t850,ws10,ws850,msl
#   bash A2E/scripts/eval_one.sh A2Ec70_gfs_cma_hres_refnorm \
#     EVAL_SOURCES=gfs,cma,hres \
#     EVAL_DATES=${EVAL_DATES} \
#     EVAL_VARIABLES=z500,t2m,t850,ws10,ws850,msl
#
# 2 / loss_ablation
#   bash A2E/scripts/eval_one.sh A2Ec70_gfs_refnorm \
#     EVAL_SOURCES=gfs \
#     EVAL_DATES=${EVAL_DATES} \
#     EVAL_VARIABLES=z500,t2m,t850,ws10,ws850,msl
#   bash A2E/scripts/eval_one.sh A2Ec70_ab_wo_fuxi \
#     EVAL_SOURCES=gfs \
#     EVAL_DATES=${EVAL_DATES} \
#     EVAL_VARIABLES=z500,t2m,t850,ws10,ws850,msl
#   bash A2E/scripts/eval_one.sh A2Ec70_ab_wo_grad \
#     EVAL_SOURCES=gfs \
#     EVAL_DATES=${EVAL_DATES} \
#     EVAL_VARIABLES=z500,t2m,t850,ws10,ws850,msl
#   bash A2E/scripts/eval_one.sh A2Ec70_ab_l1_only \
#     EVAL_SOURCES=gfs \
#     EVAL_DATES=${EVAL_DATES} \
#     EVAL_VARIABLES=z500,t2m,t850,ws10,ws850,msl
#   bash A2E/scripts/eval_one.sh A2Ec70_ab_wo_source_emb \
#     EVAL_SOURCES=gfs,cma,hres \
#     EVAL_DATES=${EVAL_DATES} \
#     EVAL_VARIABLES=z500,t2m,t850,ws10,ws850,msl
#
# 3 / data_scale
#   bash A2E/scripts/eval_one.sh A2Ec70_gfs_data20_refnorm \
#     EVAL_SOURCES=gfs \
#     EVAL_DATES=${EVAL_DATES} \
#     EVAL_VARIABLES=z500,t2m,t850,ws10,ws850,msl
#   bash A2E/scripts/eval_one.sh A2Ec70_gfs_data40_refnorm \
#     EVAL_SOURCES=gfs \
#     EVAL_DATES=${EVAL_DATES} \
#     EVAL_VARIABLES=z500,t2m,t850,ws10,ws850,msl
#   bash A2E/scripts/eval_one.sh A2Ec70_gfs_data60_refnorm \
#     EVAL_SOURCES=gfs \
#     EVAL_DATES=${EVAL_DATES} \
#     EVAL_VARIABLES=z500,t2m,t850,ws10,ws850,msl
#   bash A2E/scripts/eval_one.sh A2Ec70_gfs_data80_refnorm \
#     EVAL_SOURCES=gfs \
#     EVAL_DATES=${EVAL_DATES} \
#     EVAL_VARIABLES=z500,t2m,t850,ws10,ws850,msl
#   bash A2E/scripts/eval_one.sh A2Ec70_gfs_refnorm \
#     EVAL_SOURCES=gfs \
#     EVAL_DATES=${EVAL_DATES} \
#     EVAL_VARIABLES=z500,t2m,t850,ws10,ws850,msl
#
# 4 / source_mix / dual
#   bash A2E/scripts/eval_one.sh A2Ec70_gfs_refnorm \
#     EVAL_SOURCES=gfs,cma,hres \
#     EVAL_DATES=${EVAL_DATES} \
#     EVAL_VARIABLES=z500,t2m,t850,ws10,ws850,msl
#   bash A2E/scripts/eval_one.sh A2Ec70_cma_refnorm \
#     EVAL_SOURCES=gfs,cma,hres \
#     EVAL_DATES=${EVAL_DATES} \
#     EVAL_VARIABLES=z500,t2m,t850,ws10,ws850,msl
#   bash A2E/scripts/eval_one.sh A2Ec70_hres_refnorm \
#     EVAL_SOURCES=gfs,cma,hres \
#     EVAL_DATES=${EVAL_DATES} \
#     EVAL_VARIABLES=z500,t2m,t850,ws10,ws850,msl
#   bash A2E/scripts/eval_one.sh A2Ec70_gfs_cma_refnorm \
#     EVAL_SOURCES=gfs,cma,hres \
#     EVAL_DATES=${EVAL_DATES} \
#     EVAL_VARIABLES=z500,t2m,t850,ws10,ws850,msl
#   bash A2E/scripts/eval_one.sh A2Ec70_gfs_hres_refnorm \
#     EVAL_SOURCES=gfs,cma,hres \
#     EVAL_DATES=${EVAL_DATES} \
#     EVAL_VARIABLES=z500,t2m,t850,ws10,ws850,msl
#   bash A2E/scripts/eval_one.sh A2Ec70_cma_hres_refnorm \
#     EVAL_SOURCES=gfs,cma,hres \
#     EVAL_DATES=${EVAL_DATES} \
#     EVAL_VARIABLES=z500,t2m,t850,ws10,ws850,msl
#   bash A2E/scripts/eval_one.sh A2Ec70_gfs_cma_hres_refnorm \
#     EVAL_SOURCES=gfs,cma,hres \
#     EVAL_DATES=${EVAL_DATES} \
#     EVAL_VARIABLES=z500,t2m,t850,ws10,ws850,msl
#
# 5 / depth
#   # The depth phase also runs profile_a2e.py for all three experiments.
#   bash A2E/scripts/eval_one.sh A2Ec70_gfs_refnorm \
#     EVAL_SOURCES=gfs \
#     EVAL_DATES=${EVAL_DATES} \
#     EVAL_VARIABLES=z500,t2m,t850,ws10,ws850,msl
#   bash A2E/scripts/eval_one.sh A2Ec70_mid_refnorm \
#     EVAL_SOURCES=gfs \
#     EVAL_DATES=${EVAL_DATES} \
#     EVAL_VARIABLES=z500,t2m,t850,ws10,ws850,msl
#   bash A2E/scripts/eval_one.sh A2Ec70_deep_refnorm \
#     EVAL_SOURCES=gfs \
#     EVAL_DATES=${EVAL_DATES} \
#     EVAL_VARIABLES=z500,t2m,t850,ws10,ws850,msl
#
# 6 / parameter
#   bash A2E/scripts/eval_one.sh A2Ec70_gradw_0p1 \
#     EVAL_SOURCES=gfs \
#     EVAL_DATES=${EVAL_DATES} \
#     EVAL_VARIABLES=z500,t2m,t850,ws10,ws850,msl
#   bash A2E/scripts/eval_one.sh A2Ec70_gradw_0p2 \
#     EVAL_SOURCES=gfs \
#     EVAL_DATES=${EVAL_DATES} \
#     EVAL_VARIABLES=z500,t2m,t850,ws10,ws850,msl
#   bash A2E/scripts/eval_one.sh A2Ec70_gfs_refnorm \
#     EVAL_SOURCES=gfs \
#     EVAL_DATES=${EVAL_DATES} \
#     EVAL_VARIABLES=z500,t2m,t850,ws10,ws850,msl
#   bash A2E/scripts/eval_one.sh A2Ec70_gradw_0p8 \
#     EVAL_SOURCES=gfs \
#     EVAL_DATES=${EVAL_DATES} \
#     EVAL_VARIABLES=z500,t2m,t850,ws10,ws850,msl
#   bash A2E/scripts/eval_one.sh A2Ec70_refnorm_w1em3 \
#     EVAL_SOURCES=gfs \
#     EVAL_DATES=${EVAL_DATES} \
#     EVAL_VARIABLES=z500,t2m,t850,ws10,ws850,msl
#   bash A2E/scripts/eval_one.sh A2Ec70_refnorm_w2em3 \
#     EVAL_SOURCES=gfs \
#     EVAL_DATES=${EVAL_DATES} \
#     EVAL_VARIABLES=z500,t2m,t850,ws10,ws850,msl
#   bash A2E/scripts/eval_one.sh A2Ec70_refnorm_w4em3 \
#     EVAL_SOURCES=gfs \
#     EVAL_DATES=${EVAL_DATES} \
#     EVAL_VARIABLES=z500,t2m,t850,ws10,ws850,msl
#
# =============================================================================

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
A2E_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
RECOMMENDED_SCRIPT="${A2E_ROOT}/scripts/run_recommended_experiments.sh"
RUN=${RUN:-0}

show_list() {
  bash "${A2E_ROOT}/scripts/train_interface.sh" list
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
    echo "运行 bash A2E/scripts/eval_interface.sh list 查看可用编号" >&2
    exit 2
  fi
  echo
  echo "======================================================================"
  echo "评估实验组: $key -> $phase"
  echo "ACTION=eval RUN=$RUN bash $RECOMMENDED_SCRIPT $phase"
  echo "======================================================================"
  ACTION=eval RUN="$RUN" bash "$RECOMMENDED_SCRIPT" "$phase"
}

run_paper_min() {
  echo "将评估最小论文集合: 0(smoke) -> 1(main) -> 2(loss) -> 3(data_scale) -> 5(depth)"
  run_phase 0
  run_phase 1
  run_phase 2
  run_phase 3
  run_phase 5
}

run_paper_full() {
  echo "将评估完整论文集合: 0(smoke) -> 1(main) -> 2(loss) -> 3(data_scale) -> 4(source_mix) -> 5(depth)"
  run_phase 0
  run_phase 1
  run_phase 2
  run_phase 3
  run_phase 4
  run_phase 5
}

if [[ $# -eq 0 || "${1:-}" == "list" || "${1:-}" == "help" || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  show_list
  exit 0
fi

echo "RUN=$RUN"
if [[ "$RUN" != "1" ]]; then
  echo "当前是评估预览模式，不会真正运行。真正评估请加：RUN=1"
else
  echo "当前是真正评估模式。"
fi

echo "请求评估: $*"

for key in "$@"; do
  case "$key" in
    8|paper_min) run_paper_min ;;
    9|paper_full) run_paper_full ;;
    *) run_phase "$key" ;;
  esac
done
