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
