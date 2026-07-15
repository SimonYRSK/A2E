#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
A2E_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
source "${A2E_ROOT}/configs/common.sh"

usage() {
  cat <<USAGE
Usage:
  bash A2E/scripts/run_one.sh EXP_NAME [KEY=VALUE ...]

Examples:
  bash A2E/scripts/run_one.sh A2Ec70_cma_refnorm SOURCES=cma FUXI_LOSS_MODE=reference_norm CHANNEL_RMSE_WEIGHT=4e-3
  bash A2E/scripts/run_one.sh A2Ec70_small CHANNELS=192,384,768 EMBED_DIM=192
  bash A2E/scripts/run_one.sh A2Ec70_deep_refnorm SOURCES=gfs DEPTH=0,1,2 RES_PER_STAGE=1,1,2

Common KEY=VALUE overrides:
  SOURCES=cma|gfs|hres|gfs,cma,hres
  EPOCHS=90
  FUXI_LOSS_MODE=reference_norm|manual_weighted|raw_mean
  CHANNEL_RMSE_WEIGHT=4e-3
  CHANNELS=384,768,1536
  EMBED_DIM=384
  DEPTH=0,0,1             A2E-Lite; use 0,1,2 for A2E-Deep
  RES_PER_STAGE=1,1,1     A2E-Lite; use 1,1,2 for A2E-Deep
  USE_GRAD_LOSS=true|false
  USING_TIME_EMBEDDING=true|false
  USING_SOURCE_EMBEDDING=true|false
  TRAIN_SAMPLE_RATIO=1.0      data-scale experiments use 0.2/0.4/0.6/0.8
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

mkdir -p "${OUTPUT_ROOT}/${EXP_NAME}/metrics" "${CHECKPOINT_ROOT}/${EXP_NAME}" "${TENSORBOARD_ROOT}/${EXP_NAME}" "${PLOT_ROOT}/${EXP_NAME}"

LOG_PATH="${OUTPUT_ROOT}/${EXP_NAME}/train.log"
CONFIG_PATH="${OUTPUT_ROOT}/${EXP_NAME}/metrics/launch_config.env"

env | sort | grep -E '^(EXP_NAME|SOURCES|EPOCHS|BATCH_SIZE|BASE_LR|MIN_LR|WARMUP_EPOCHS|WEIGHT_DECAY|CHANNELS|EMBED_DIM|DEPTH|RES_PER_STAGE|FUXI_LOSS_MODE|CHANNEL_RMSE_WEIGHT|USE_GRAD_LOSS|GRAD_LOSS_WEIGHT|USING_TIME_EMBEDDING|USING_SOURCE_EMBEDDING|TRAIN_SAMPLE_RATIO|EVAL_VARIABLES|EVAL_FORECAST_STEPS|PROFILE_|TRAIN_|VAL_|GFS_|HRES_|CMA_|ERA5_PATH|GFS_PATH|HRES_PATH|CMA_PATH|FUXI_DIR|CLIM_PATH|OUTPUT_ROOT|CHECKPOINT_ROOT|TENSORBOARD_ROOT|PLOT_ROOT)=' > "${CONFIG_PATH}"

echo "[run_one] exp=${EXP_NAME} sources=${SOURCES} epochs=${EPOCHS} channels=${CHANNELS} depth=${DEPTH} res_per_stage=${RES_PER_STAGE} fuxi_loss=${FUXI_LOSS_MODE} channel_rmse_weight=${CHANNEL_RMSE_WEIGHT}"
echo "[run_one] log=${LOG_PATH}"

cd "${A2E_ROOT}"

torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  main_res_exp.py \
  --exp_name "${EXP_NAME}" \
  --sources "${SOURCES}" \
  --seed "${SEED}" \
  --era5_path "${ERA5_PATH}" \
  --gfs_path "${GFS_PATH}" \
  --hres_path "${HRES_PATH}" \
  --cma_path "${CMA_PATH}" \
  --fuxi_dir "${FUXI_DIR}" \
  --output_root "${OUTPUT_ROOT}" \
  --checkpoint_root "${CHECKPOINT_ROOT}" \
  --tensorboard_root "${TENSORBOARD_ROOT}" \
  --plot_root "${PLOT_ROOT}" \
  --train_start "${TRAIN_START}" \
  --train_end "${TRAIN_END}" \
  --val_start "${VAL_START}" \
  --val_end "${VAL_END}" \
  --gfs_train_start "${GFS_TRAIN_START}" \
  --gfs_train_end "${GFS_TRAIN_END}" \
  --gfs_val_start "${GFS_VAL_START}" \
  --gfs_val_end "${GFS_VAL_END}" \
  --cma_train_start "${CMA_TRAIN_START}" \
  --cma_train_end "${CMA_TRAIN_END}" \
  --cma_val_start "${CMA_VAL_START}" \
  --cma_val_end "${CMA_VAL_END}" \
  --hres_train_start "${HRES_TRAIN_START}" \
  --hres_train_end "${HRES_TRAIN_END}" \
  --hres_val_start "${HRES_VAL_START}" \
  --hres_val_end "${HRES_VAL_END}" \
  --val_sample_per_month "${VAL_SAMPLE_PER_MONTH}" \
  --val_sample_year "${VAL_SAMPLE_YEAR}" \
  --max_samples_per_year "${MAX_SAMPLES_PER_YEAR}" \
  --train_sample_ratio "${TRAIN_SAMPLE_RATIO}" \
  --sample_seed "${SAMPLE_SEED}" \
  --img_size "${IMG_SIZE}" \
  --patch_size "${PATCH_SIZE}" \
  --in_chans "${IN_CHANS}" \
  --out_chans "${OUT_CHANS}" \
  --embed_dim "${EMBED_DIM}" \
  --num_groups "${NUM_GROUPS}" \
  --num_heads "${NUM_HEADS}" \
  --num_stages "${NUM_STAGES}" \
  --window_size "${WINDOW_SIZE}" \
  --depth "${DEPTH}" \
  --res_per_stage "${RES_PER_STAGE}" \
  --channels "${CHANNELS}" \
  --using_time_embedding "${USING_TIME_EMBEDDING}" \
  --using_source_embedding "${USING_SOURCE_EMBEDDING}" \
  --using_kl "${USING_KL}" \
  --dropout_rate "${DROPOUT_RATE}" \
  --use_skip_connections "${USE_SKIP_CONNECTIONS}" \
  --use_residual_blocks "${USE_RESIDUAL_BLOCKS}" \
  --using_dann "${USING_DANN}" \
  --domain_loss_weight "${DOMAIN_LOSS_WEIGHT}" \
  --dann_gamma "${DANN_GAMMA}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --prefetch_factor "${PREFETCH_FACTOR}" \
  --base_lr "${BASE_LR}" \
  --min_lr "${MIN_LR}" \
  --warmup_epochs "${WARMUP_EPOCHS}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --betas "${BETAS}" \
  --save_interval "${SAVE_INTERVAL}" \
  --use_amp "${USE_AMP}" \
  --beta "${BETA}" \
  --kl_anneal "${KL_ANNEAL}" \
  --kl_anneal_epochs "${KL_ANNEAL_EPOCHS}" \
  --recon_loss_type "${RECON_LOSS_TYPE}" \
  --charbonnier_eps "${CHARBONNIER_EPS}" \
  --use_grad_loss "${USE_GRAD_LOSS}" \
  --grad_loss_weight "${GRAD_LOSS_WEIGHT}" \
  --l1_reg_weight "${L1_REG_WEIGHT}" \
  --l2_reg_weight "${L2_REG_WEIGHT}" \
  --fuxi_loss_mode "${FUXI_LOSS_MODE}" \
  --channel_rmse_weight "${CHANNEL_RMSE_WEIGHT}" \
  --rmse_every_n_steps "${RMSE_EVERY_N_STEPS}" \
  --rmse_samples_per_batch "${RMSE_SAMPLES_PER_BATCH}" \
  --fuxi_lead_hours "${FUXI_LEAD_HOURS}" \
  2>&1 | tee "${LOG_PATH}"
