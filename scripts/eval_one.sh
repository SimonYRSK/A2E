#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
A2E_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
source "${A2E_ROOT}/configs/common.sh"

usage() {
  cat <<USAGE
Usage:
  bash A2E/scripts/eval_one.sh EXP_NAME [KEY=VALUE ...]

Examples:
  bash A2E/scripts/eval_one.sh A2Ec70_cma_refnorm EVAL_SOURCES=cma EVAL_DATES=20250101,20250105
  bash A2E/scripts/eval_one.sh A2Ec70_full CKPT=/path/to/best.pth EVAL_SOURCES=gfs,cma,hres

Common KEY=VALUE overrides:
  CKPT=/path/to/checkpoint.pth       default: CHECKPOINT_ROOT/EXP_NAME/best.pth
  CONFIG=/path/to/config.json        default: OUTPUT_ROOT/EXP_NAME/metrics/config.json
  EVAL_SOURCES=cma|gfs,cma,hres      default: SOURCES from common config
  EVAL_DATES=20250101,20250102       default: 20250101:20251122; START:END means daily init dates
  EVAL_VARIABLES=z500,t2m,t850,ws10,ws850,msl
  EVAL_FORECAST_STEPS=40             FuXi rollout steps, 6h per step

Outputs:
  fuxi_rollout_metrics_detail.csv    FuXi(A2E) rollout RMSE/ACC by lead/variable
  fuxi_rollout_metrics_summary.csv   mean FuXi(A2E) rollout RMSE/ACC
  a2e_initial_metrics.csv            A2E-vs-ERA5 L1/GradLoss plus PSNR/SSIM
USAGE
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

EXP_NAME=$1
shift

for kv in "$@"; do
  if [[ "${kv}" != *=* ]]; then
    echo "Invalid override '${kv}'. Expected KEY=VALUE." >&2
    exit 2
  fi
  export "${kv}"
done

CKPT=${CKPT:-${CHECKPOINT_ROOT}/${EXP_NAME}/best.pth}
CONFIG=${CONFIG:-${OUTPUT_ROOT}/${EXP_NAME}/metrics/config.json}
EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR:-${OUTPUT_ROOT}/${EXP_NAME}/eval}
EVAL_SOURCES=${EVAL_SOURCES:-${SOURCES}}

mkdir -p "${EVAL_OUTPUT_DIR}"
LOG_PATH="${EVAL_OUTPUT_DIR}/eval.log"

echo "[eval_one] exp=${EXP_NAME} ckpt=${CKPT} sources=${EVAL_SOURCES} dates=${EVAL_DATES} variables=${EVAL_VARIABLES} forecast_steps=${EVAL_FORECAST_STEPS}"
echo "[eval_one] FuXi rollout output=${EVAL_OUTPUT_DIR}"

cd "${A2E_ROOT}"

"${PYTHON_BIN}" eval/eval_fuxi_rollout.py \
  --exp_name "${EXP_NAME}" \
  --config "${CONFIG}" \
  --ckpt "${CKPT}" \
  --output_dir "${EVAL_OUTPUT_DIR}" \
  --sources "${EVAL_SOURCES}" \
  --dates "${EVAL_DATES}" \
  --variables "${EVAL_VARIABLES}" \
  --forecast_steps "${EVAL_FORECAST_STEPS}" \
  --era5_path "${ERA5_PATH}" \
  --gfs_path "${GFS_PATH}" \
  --hres_path "${HRES_PATH}" \
  --cma_path "${CMA_PATH}" \
  --fuxi_dir "${FUXI_DIR}" \
  --clim_path "${CLIM_PATH}" \
  2>&1 | tee "${LOG_PATH}"
