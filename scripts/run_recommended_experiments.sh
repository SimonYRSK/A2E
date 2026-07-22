#!/usr/bin/env bash
set -euo pipefail

# Recommended A2E-c70 no-GradLoss experiment schedule.
#
# This backup schedule assumes the no-GradLoss configuration is the final/main
# configuration. Every training command is forced to:
#   USE_GRAD_LOSS=false, GRAD_LOSS_WEIGHT=0
#
# Use ACTION to decide whether to train, evaluate, or run both:
#   ACTION=train RUN=1 bash A2E_backup/scripts/run_recommended_experiments.sh main
#   ACTION=eval  RUN=1 bash A2E_backup/scripts/run_recommended_experiments.sh main
#   ACTION=all   RUN=1 bash A2E_backup/scripts/run_recommended_experiments.sh main
#
# Evaluation uses eval/eval_fuxi_rollout.py:
#   FuXi(A2E) rollout RMSE/ACC + A2E initial-field L1/PSNR/SSIM.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
A2E_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
source "${A2E_ROOT}/configs/common.sh"

PHASE=${1:-all}
RUN=${RUN:-0}
ACTION=${ACTION:-all}

case "${ACTION}" in
  train|eval|all) ;;
  *)
    echo "Unknown ACTION=${ACTION}. Use train, eval, or all." >&2
    exit 2
    ;;
esac

run_cmd() {
  echo
  echo "================================================================================"
  echo "$*"
  echo "================================================================================"
  if [[ "${RUN}" == "1" ]]; then
    eval "$*"
  else
    echo "[DRY-RUN] Set RUN=1 to execute."
  fi
}

run_train() {
  local exp_name=$1
  shift
  if [[ "${ACTION}" == "train" || "${ACTION}" == "all" ]]; then
    run_cmd "bash '${A2E_ROOT}/scripts/run_one.sh' '${exp_name}' $*"
  else
    echo "[SKIP-TRAIN] ${exp_name} (ACTION=${ACTION})"
  fi
}

run_eval() {
  local exp_name=$1
  shift
  if [[ "${ACTION}" == "eval" || "${ACTION}" == "all" ]]; then
    run_cmd "bash '${A2E_ROOT}/scripts/eval_one.sh' '${exp_name}' $*"
  else
    echo "[SKIP-EVAL] ${exp_name} (ACTION=${ACTION})"
  fi
}

run_profile() {
  local exp_name=$1
  if [[ "${ACTION}" == "eval" || "${ACTION}" == "all" ]]; then
    local config_path="${OUTPUT_ROOT}/${exp_name}/metrics/config.json"
    local profile_dir="${OUTPUT_ROOT}/${exp_name}/profile"
    run_cmd "'${PYTHON_BIN}' '${A2E_ROOT}/eval/profile_a2e.py' --exp_name '${exp_name}' --config '${config_path}' --output_dir '${profile_dir}' --device '${PROFILE_DEVICE}' --batch_size '${PROFILE_BATCH_SIZE}' --warmup '${PROFILE_WARMUP}' --iters '${PROFILE_ITERS}' --compute_flops '${PROFILE_COMPUTE_FLOPS}'"
  else
    echo "[SKIP-PROFILE] ${exp_name} (ACTION=${ACTION})"
  fi
}

# -----------------------------------------------------------------------------
# Phase 0: Smoke test
# -----------------------------------------------------------------------------
phase_smoke() {
  run_train "smoke_gfs_refnorm" \
    "SOURCES=gfs" \
    "EPOCHS=1" \
    "VAL_SAMPLE_PER_MONTH=1" \
    "RMSE_EVERY_N_STEPS=1" \
    "FUXI_LOSS_MODE=reference_norm" \
    "CHANNEL_RMSE_WEIGHT=8e-3"

  run_eval "smoke_gfs_refnorm" \
    "EVAL_SOURCES=gfs" \
    "EVAL_DATES=20250101" \
    "EVAL_VARIABLES=z500,t2m,t850,ws10,ws850,msl"
}

# -----------------------------------------------------------------------------
# Phase 1: Main experiment
# Purpose: no-GradLoss single-source models + full GFS/CMA/HRES model.
# -----------------------------------------------------------------------------
phase_main() {
  run_train "A2Ec70_gfs_refnorm" \
    "SOURCES=gfs" \
    "EPOCHS=90" \
    "FUXI_LOSS_MODE=reference_norm" \
    "CHANNEL_RMSE_WEIGHT=8e-3"

  run_train "A2Ec70_cma_refnorm" \
    "SOURCES=cma" \
    "EPOCHS=90" \
    "FUXI_LOSS_MODE=reference_norm" \
    "CHANNEL_RMSE_WEIGHT=8e-3"

  run_train "A2Ec70_hres_refnorm" \
    "SOURCES=hres" \
    "EPOCHS=90" \
    "FUXI_LOSS_MODE=reference_norm" \
    "CHANNEL_RMSE_WEIGHT=8e-3"

  run_train "A2Ec70_gfs_cma_hres_refnorm" \
    "SOURCES=gfs,cma,hres" \
    "EPOCHS=90" \
    "FUXI_LOSS_MODE=reference_norm" \
    "CHANNEL_RMSE_WEIGHT=8e-3"

  for exp in \
    A2Ec70_gfs_refnorm \
    A2Ec70_cma_refnorm \
    A2Ec70_hres_refnorm \
    A2Ec70_gfs_cma_hres_refnorm
  do
    run_eval "${exp}" \
      "EVAL_SOURCES=gfs,cma,hres" \
      "EVAL_DATES=${EVAL_DATES}" \
      "EVAL_VARIABLES=z500,t2m,t850,ws10,ws850,msl"
  done
}

# -----------------------------------------------------------------------------
# Phase 2: Core loss/source ablation
# Purpose: show whether FuXi downstream loss and source embedding matter.
# Final no-GradLoss baseline is reused from phase_main:
#   A2Ec70_gfs_refnorm = L1 + FuXi reference_norm, w=8e-3.
# -----------------------------------------------------------------------------
phase_loss_ablation() {
  run_train "A2Ec70_ab_wo_fuxi" \
    "SOURCES=gfs" \
    "EPOCHS=90" \
    "FUXI_LOSS_MODE=reference_norm" \
    "CHANNEL_RMSE_WEIGHT=0"

  run_train "A2Ec70_ab_wo_source_emb" \
    "SOURCES=gfs,cma,hres" \
    "EPOCHS=90" \
    "USING_TIME_EMBEDDING=true" \
    "USING_SOURCE_EMBEDDING=false" \
    "FUXI_LOSS_MODE=reference_norm" \
    "CHANNEL_RMSE_WEIGHT=8e-3"

  for exp in \
    A2Ec70_gfs_refnorm \
    A2Ec70_ab_wo_fuxi
  do
    run_eval "${exp}" \
      "EVAL_SOURCES=gfs" \
      "EVAL_DATES=${EVAL_DATES}" \
      "EVAL_VARIABLES=z500,t2m,t850,ws10,ws850,msl"
  done

  run_eval "A2Ec70_ab_wo_source_emb" \
    "EVAL_SOURCES=gfs,cma,hres" \
    "EVAL_DATES=${EVAL_DATES}" \
    "EVAL_VARIABLES=z500,t2m,t850,ws10,ws850,msl"
}

# -----------------------------------------------------------------------------
# Phase 3: Data scale study
# -----------------------------------------------------------------------------
phase_data_scale() {
  run_train "A2Ec70_gfs_data25_refnorm" \
    "SOURCES=gfs" \
    "EPOCHS=90" \
    "TRAIN_SAMPLE_RATIO=0.25" \
    "FUXI_LOSS_MODE=reference_norm" \
    "CHANNEL_RMSE_WEIGHT=8e-3"

  run_train "A2Ec70_gfs_data50_refnorm" \
    "SOURCES=gfs" \
    "EPOCHS=90" \
    "TRAIN_SAMPLE_RATIO=0.5" \
    "FUXI_LOSS_MODE=reference_norm" \
    "CHANNEL_RMSE_WEIGHT=8e-3"

  for exp in \
    A2Ec70_gfs_data25_refnorm \
    A2Ec70_gfs_data50_refnorm \
    A2Ec70_gfs_refnorm
  do
    run_eval "${exp}" \
      "EVAL_SOURCES=gfs" \
      "EVAL_DATES=${EVAL_DATES}" \
      "EVAL_VARIABLES=z500,t2m,t850,ws10,ws850,msl"
  done
}

# -----------------------------------------------------------------------------
# Phase 4: Source mixing study
# -----------------------------------------------------------------------------
phase_source_mix() {
  run_train "A2Ec70_gfs_cma_refnorm" \
    "SOURCES=gfs,cma" \
    "EPOCHS=90" \
    "FUXI_LOSS_MODE=reference_norm" \
    "CHANNEL_RMSE_WEIGHT=8e-3"

  run_train "A2Ec70_gfs_hres_refnorm" \
    "SOURCES=gfs,hres" \
    "EPOCHS=90" \
    "FUXI_LOSS_MODE=reference_norm" \
    "CHANNEL_RMSE_WEIGHT=8e-3"

  run_train "A2Ec70_cma_hres_refnorm" \
    "SOURCES=cma,hres" \
    "EPOCHS=90" \
    "FUXI_LOSS_MODE=reference_norm" \
    "CHANNEL_RMSE_WEIGHT=8e-3"

  for exp in \
    A2Ec70_gfs_refnorm \
    A2Ec70_cma_refnorm \
    A2Ec70_hres_refnorm \
    A2Ec70_gfs_cma_refnorm \
    A2Ec70_gfs_hres_refnorm \
    A2Ec70_cma_hres_refnorm \
    A2Ec70_gfs_cma_hres_refnorm
  do
    run_eval "${exp}" \
      "EVAL_SOURCES=gfs,cma,hres" \
      "EVAL_DATES=${EVAL_DATES}" \
      "EVAL_VARIABLES=z500,t2m,t850,ws10,ws850,msl"
  done
}

# -----------------------------------------------------------------------------
# Phase 5: Depth scaling
# -----------------------------------------------------------------------------
phase_depth_scaling() {
  run_train "A2Ec70_mid_refnorm" \
    "SOURCES=gfs" \
    "EPOCHS=90" \
    "EMBED_DIM=384" \
    "CHANNELS=384,768,1536" \
    "DEPTH=0,0,2" \
    "RES_PER_STAGE=2,2,2" \
    "FUXI_LOSS_MODE=reference_norm" \
    "CHANNEL_RMSE_WEIGHT=8e-3"

  run_train "A2Ec70_deep_refnorm" \
    "SOURCES=gfs" \
    "EPOCHS=90" \
    "EMBED_DIM=384" \
    "CHANNELS=384,768,1536" \
    "DEPTH=0,1,2" \
    "RES_PER_STAGE=3,3,3" \
    "FUXI_LOSS_MODE=reference_norm" \
    "CHANNEL_RMSE_WEIGHT=8e-3"

  for exp in \
    A2Ec70_gfs_refnorm \
    A2Ec70_mid_refnorm \
    A2Ec70_deep_refnorm
  do
    run_profile "${exp}"
    run_eval "${exp}" \
      "EVAL_SOURCES=gfs" \
      "EVAL_DATES=${EVAL_DATES}" \
      "EVAL_VARIABLES=z500,t2m,t850,ws10,ws850,msl"
  done
}

# -----------------------------------------------------------------------------
# Phase 6: FuXi RMSE loss weight sensitivity
# -----------------------------------------------------------------------------
phase_parameter_study() {
  for w in 2e-3 4e-3 1e-2; do
    tag=${w/-/m}
    run_train "A2Ec70_refnorm_w${tag}" \
      "SOURCES=gfs" \
      "EPOCHS=90" \
      "FUXI_LOSS_MODE=reference_norm" \
      "CHANNEL_RMSE_WEIGHT=${w}"
  done

  for exp in \
    A2Ec70_refnorm_w2em3 \
    A2Ec70_refnorm_w4em3 \
    A2Ec70_gfs_refnorm \
    A2Ec70_refnorm_w1em2
  do
    run_eval "${exp}" \
      "EVAL_SOURCES=gfs" \
      "EVAL_DATES=${EVAL_DATES}" \
      "EVAL_VARIABLES=z500,t2m,t850,ws10,ws850,msl"
  done
}

phase_raw_baseline_note() {
  echo
  echo "[NOTE] Raw GFS/CMA/HRES baseline is listed in the checklist as completed."
  echo "[NOTE] Keep it in the same metrics table as A2E+FuXi rollout outputs: z500,t2m,t850,ws10,ws850,msl."
  echo "[NOTE] This script trains/evaluates A2E models; raw source-vs-ERA5 baseline should be exported by a separate raw-eval script if needed."
}

case "${PHASE}" in
  smoke) phase_smoke ;;
  main) phase_main ;;
  loss_ablation|loss) phase_loss_ablation ;;
  data_scale|datascale|data) phase_data_scale ;;
  source_mix|mix|dual_source|dual) phase_source_mix ;;
  depth|depth_scaling|scaling|scale) phase_depth_scaling ;;
  parameter|param) phase_parameter_study ;;
  raw_note|raw) phase_raw_baseline_note ;;
  all)
    phase_raw_baseline_note
    phase_smoke
    phase_main
    phase_loss_ablation
    phase_data_scale
    phase_source_mix
    phase_depth_scaling
    phase_parameter_study
    ;;
  paper_min)
    phase_smoke
    phase_main
    phase_loss_ablation
    phase_data_scale
    phase_depth_scaling
    ;;
  paper_full)
    phase_smoke
    phase_main
    phase_loss_ablation
    phase_data_scale
    phase_source_mix
    phase_depth_scaling
    phase_parameter_study
    ;;
  *)
    echo "Unknown phase: ${PHASE}" >&2
    echo "Available phases: smoke, main, loss_ablation, data_scale, source_mix, depth, parameter, raw_note, all, paper_min, paper_full" >&2
    exit 2
    ;;
esac
