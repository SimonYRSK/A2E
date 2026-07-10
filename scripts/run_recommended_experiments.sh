#!/usr/bin/env bash
set -euo pipefail

# Recommended A2E-c70 experiment schedule.
#
# This script organizes the experiments from:
#   A2E Experiment CheckList.docx
# plus FuXi-loss-mode ablations:
#   raw_mean / reference_norm
#
# De-duplication policy:
#   A2Ec70_gfs_refnorm is the canonical single-source full/reference_norm model.
#   A2Ec70_gfs_cma_hres_refnorm is the canonical multi-source full model.
#   Later phases reuse these results instead of retraining equivalent configs.
#
# Default is DRY RUN: commands are printed but not executed.
# Execute for real with:
#   RUN=1 bash A2E/scripts/run_recommended_experiments.sh all
# Run one phase with:
#   RUN=1 bash A2E/scripts/run_recommended_experiments.sh main
#   RUN=1 bash A2E/scripts/run_recommended_experiments.sh fuxi_loss
#
# Recommended first smoke test:
#   RUN=1 bash A2E/scripts/run_recommended_experiments.sh smoke

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
A2E_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
source "${A2E_ROOT}/configs/common.sh"

PHASE=${1:-all}
RUN=${RUN:-0}

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
  run_cmd "bash '${A2E_ROOT}/scripts/run_one.sh' '${exp_name}' $*"
}

run_eval() {
  local exp_name=$1
  shift
  run_cmd "bash '${A2E_ROOT}/scripts/eval_one.sh' '${exp_name}' $*"
}

# -----------------------------------------------------------------------------
# Phase 0: Smoke test
# -----------------------------------------------------------------------------
phase_smoke() {
  # One-epoch quick check for config saving, training loop, checkpoint, and metrics CSV.
  run_train "smoke_gfs_refnorm" \
    "SOURCES=gfs" \
    "EPOCHS=1" \
    "VAL_SAMPLE_PER_MONTH=1" \
    "RMSE_EVERY_N_STEPS=10" \
    "FUXI_LOSS_MODE=reference_norm" \
    "CHANNEL_RMSE_WEIGHT=4e-3"

  run_eval "smoke_gfs_refnorm" \
    "EVAL_SOURCES=gfs" \
    "EVAL_DATES=20250101" \
    "EVAL_VARIABLES=z500,t2m,tp,ws10m,msl,r700"
}

# -----------------------------------------------------------------------------
# Phase 1: Main experiment
# Purpose: establish single-source and multi-source main results.
# Evaluation variables include r700, not q700.
# -----------------------------------------------------------------------------
phase_main() {
  # Single-source baselines. These are the cleanest comparison against Raw GFS/CMA/HRES.
  run_train "A2Ec70_gfs_refnorm" \
    "SOURCES=gfs" \
    "EPOCHS=90" \
    "FUXI_LOSS_MODE=reference_norm" \
    "CHANNEL_RMSE_WEIGHT=4e-3"

  run_train "A2Ec70_cma_refnorm" \
    "SOURCES=cma" \
    "EPOCHS=90" \
    "FUXI_LOSS_MODE=reference_norm" \
    "CHANNEL_RMSE_WEIGHT=4e-3"

  run_train "A2Ec70_hres_refnorm" \
    "SOURCES=hres" \
    "EPOCHS=90" \
    "FUXI_LOSS_MODE=reference_norm" \
    "CHANNEL_RMSE_WEIGHT=4e-3"

  # Full multi-source model: primary model for paper tables.
  run_train "A2Ec70_gfs_cma_hres_refnorm" \
    "SOURCES=gfs,cma,hres" \
    "EPOCHS=90" \
    "FUXI_LOSS_MODE=reference_norm" \
    "CHANNEL_RMSE_WEIGHT=4e-3"

  # Field-level evaluation for paper variables.
  for exp in \
    A2Ec70_gfs_refnorm \
    A2Ec70_cma_refnorm \
    A2Ec70_hres_refnorm \
    A2Ec70_gfs_cma_hres_refnorm
  do
    run_eval "${exp}" \
      "EVAL_SOURCES=gfs,cma,hres" \
      "EVAL_DATES=${EVAL_DATES}" \
      "EVAL_VARIABLES=z500,t2m,tp,ws10m,msl,r700"
  done
}

# -----------------------------------------------------------------------------
# Phase 1b: Optional dual-source main experiments
# Purpose: show source-combination behavior. Run after full multi-source is stable.
# -----------------------------------------------------------------------------
phase_dual_source() {
  run_train "A2Ec70_gfs_cma_refnorm" \
    "SOURCES=gfs,cma" \
    "EPOCHS=90" \
    "FUXI_LOSS_MODE=reference_norm" \
    "CHANNEL_RMSE_WEIGHT=4e-3"

  run_train "A2Ec70_gfs_hres_refnorm" \
    "SOURCES=gfs,hres" \
    "EPOCHS=90" \
    "FUXI_LOSS_MODE=reference_norm" \
    "CHANNEL_RMSE_WEIGHT=4e-3"

  run_train "A2Ec70_cma_hres_refnorm" \
    "SOURCES=cma,hres" \
    "EPOCHS=90" \
    "FUXI_LOSS_MODE=reference_norm" \
    "CHANNEL_RMSE_WEIGHT=4e-3"

  for exp in \
    A2Ec70_gfs_cma_refnorm \
    A2Ec70_gfs_hres_refnorm \
    A2Ec70_cma_hres_refnorm
  do
    run_eval "${exp}" \
      "EVAL_SOURCES=gfs,cma,hres" \
      "EVAL_DATES=${EVAL_DATES}" \
      "EVAL_VARIABLES=z500,t2m,tp,ws10m,msl,r700"
  done
}

# -----------------------------------------------------------------------------
# Phase 2: Core loss ablation
# Purpose: prove the contribution of Grad Loss and FuXi Loss.
# Use CMA first because it matches current successful experiment behavior.
# -----------------------------------------------------------------------------
phase_loss_ablation() {
  # Full baseline is reused from phase_main:
  #   A2Ec70_gfs_refnorm = L1 + Grad + FuXi reference_norm, w=4e-3.
  # Do not retrain it here.

  # w/o FuXi: L1 + Grad only.
  run_train "A2Ec70_ab_wo_fuxi" \
    "SOURCES=gfs" \
    "EPOCHS=90" \
    "USE_GRAD_LOSS=true" \
    "GRAD_LOSS_WEIGHT=0.4" \
    "FUXI_LOSS_MODE=reference_norm" \
    "CHANNEL_RMSE_WEIGHT=0"

  # w/o Grad: L1 + FuXi only.
  run_train "A2Ec70_ab_wo_grad" \
    "SOURCES=gfs" \
    "EPOCHS=90" \
    "USE_GRAD_LOSS=false" \
    "GRAD_LOSS_WEIGHT=0" \
    "FUXI_LOSS_MODE=reference_norm" \
    "CHANNEL_RMSE_WEIGHT=4e-3"

  # L1 only: no Grad, no FuXi.
  run_train "A2Ec70_ab_l1_only" \
    "SOURCES=gfs" \
    "EPOCHS=90" \
    "USE_GRAD_LOSS=false" \
    "GRAD_LOSS_WEIGHT=0" \
    "FUXI_LOSS_MODE=reference_norm" \
    "CHANNEL_RMSE_WEIGHT=0"

  for exp in \
    A2Ec70_gfs_refnorm \
    A2Ec70_ab_wo_fuxi \
    A2Ec70_ab_wo_grad \
    A2Ec70_ab_l1_only
  do
    run_eval "${exp}" \
      "EVAL_SOURCES=gfs" \
      "EVAL_DATES=${EVAL_DATES}" \
      "EVAL_VARIABLES=z500,t2m,tp,ws10m,msl,r700"
  done
}

# -----------------------------------------------------------------------------
# Phase 3: FuXi loss mode ablation
# Purpose: compare raw_mean / reference_norm fairly.
# Weight choices make effective FuXi-loss contribution close to current behavior.
# -----------------------------------------------------------------------------
phase_fuxi_loss() {
  # Raw RMSE mean: directly average raw channel-wise FuXi downstream RMSE.
  run_train "A2Ec70_fuxi_rawmean_w5e4" \
    "SOURCES=gfs" \
    "EPOCHS=90" \
    "USE_GRAD_LOSS=true" \
    "GRAD_LOSS_WEIGHT=0.4" \
    "FUXI_LOSS_MODE=raw_mean" \
    "CHANNEL_RMSE_WEIGHT=5e-4"

  # Recommended reference_norm baseline is reused from phase_main:
  #   A2Ec70_gfs_refnorm = reference_norm, w=4e-3.
  # Do not retrain it here.

  for exp in \
    A2Ec70_fuxi_rawmean_w5e4 \
    A2Ec70_gfs_refnorm
  do
    run_eval "${exp}" \
      "EVAL_SOURCES=gfs" \
      "EVAL_DATES=${EVAL_DATES}" \
      "EVAL_VARIABLES=z500,t2m,tp,ws10m,msl,r700"
  done
}

# -----------------------------------------------------------------------------
# Phase 4: Embedding ablation
# Purpose: verify time/source conditioning. Source embedding is most meaningful
# in multi-source training, so this phase uses GFS+CMA+HRES.
# -----------------------------------------------------------------------------
phase_embedding_ablation() {
  # Full multi-source embedding baseline is reused from phase_main:
  #   A2Ec70_gfs_cma_hres_refnorm = time emb + source emb.
  # Do not retrain it here.

  run_train "A2Ec70_ms_wo_time_emb" \
    "SOURCES=gfs,cma,hres" \
    "EPOCHS=90" \
    "USING_TIME_EMBEDDING=false" \
    "USING_SOURCE_EMBEDDING=true" \
    "FUXI_LOSS_MODE=reference_norm" \
    "CHANNEL_RMSE_WEIGHT=4e-3"

  run_train "A2Ec70_ms_wo_source_emb" \
    "SOURCES=gfs,cma,hres" \
    "EPOCHS=90" \
    "USING_TIME_EMBEDDING=true" \
    "USING_SOURCE_EMBEDDING=false" \
    "FUXI_LOSS_MODE=reference_norm" \
    "CHANNEL_RMSE_WEIGHT=4e-3"

  for exp in \
    A2Ec70_gfs_cma_hres_refnorm \
    A2Ec70_ms_wo_time_emb \
    A2Ec70_ms_wo_source_emb
  do
    run_eval "${exp}" \
      "EVAL_SOURCES=gfs,cma,hres" \
      "EVAL_DATES=${EVAL_DATES}" \
      "EVAL_VARIABLES=z500,t2m,tp,ws10m,msl,r700"
  done
}

# -----------------------------------------------------------------------------
# Phase 5: Model scaling
# Purpose: replace broad CNN/ResUNet/SwinUNet structure comparison with
# same-family parameter-size comparison.
# -----------------------------------------------------------------------------
phase_model_scaling() {
  run_train "A2Ec70_small_refnorm" \
    "SOURCES=gfs" \
    "EPOCHS=90" \
    "EMBED_DIM=192" \
    "CHANNELS=192,384,768" \
    "DEPTH=0,0,1" \
    "FUXI_LOSS_MODE=reference_norm" \
    "CHANNEL_RMSE_WEIGHT=4e-3"

  run_train "A2Ec70_base_refnorm" \
    "SOURCES=gfs" \
    "EPOCHS=90" \
    "EMBED_DIM=256" \
    "CHANNELS=256,512,1024" \
    "DEPTH=0,0,1" \
    "FUXI_LOSS_MODE=reference_norm" \
    "CHANNEL_RMSE_WEIGHT=4e-3"

  # Full scaling baseline is reused from phase_main:
  #   A2Ec70_gfs_refnorm = embed_dim=384, channels=384,768,1536.
  # Do not retrain it here.

  for exp in \
    A2Ec70_small_refnorm \
    A2Ec70_base_refnorm \
    A2Ec70_gfs_refnorm
  do
    run_eval "${exp}" \
      "EVAL_SOURCES=gfs" \
      "EVAL_DATES=${EVAL_DATES}" \
      "EVAL_VARIABLES=z500,t2m,tp,ws10m,msl,r700"
  done
}

# -----------------------------------------------------------------------------
# Phase 6: Parameter sensitivity
# Purpose: supplementary robustness checks. Run after main and ablations.
# -----------------------------------------------------------------------------
phase_parameter_study() {
  # Gradient-loss weight sensitivity.
  # w=0.0 is covered by A2Ec70_ab_wo_grad;
  # w=0.4 is reused from A2Ec70_gfs_refnorm.
  # Do not retrain duplicate default points here.
  for w in 0.1 0.2 0.8; do
    tag=${w/./p}
    run_train "A2Ec70_gradw_${tag}" \
      "SOURCES=gfs" \
      "EPOCHS=90" \
      "USE_GRAD_LOSS=true" \
      "GRAD_LOSS_WEIGHT=${w}" \
      "FUXI_LOSS_MODE=reference_norm" \
      "CHANNEL_RMSE_WEIGHT=4e-3"
  done

  # Reference-normalized FuXi loss weight sensitivity.
  # w=4e-3 is reused from A2Ec70_gfs_refnorm.
  # Do not retrain duplicate default point here.
  for w in 1e-3 2e-3 8e-3; do
    tag=${w/-/m}
    run_train "A2Ec70_refnorm_w${tag}" \
      "SOURCES=gfs" \
      "EPOCHS=90" \
      "USE_GRAD_LOSS=true" \
      "GRAD_LOSS_WEIGHT=0.4" \
      "FUXI_LOSS_MODE=reference_norm" \
      "CHANNEL_RMSE_WEIGHT=${w}"
  done
}

# -----------------------------------------------------------------------------
# Raw baseline placeholder
# -----------------------------------------------------------------------------
phase_raw_baseline_note() {
  echo
  echo "[NOTE] Raw GFS/CMA/HRES baseline is listed in the checklist as completed."
  echo "[NOTE] Keep it in the same metrics table as A2E outputs: z500,t2m,tp,ws10m,msl,r700."
  echo "[NOTE] This script trains/evaluates A2E models; raw source-vs-ERA5 baseline should be exported by a separate raw-eval script if needed."
}

case "${PHASE}" in
  smoke)
    phase_smoke
    ;;
  main)
    phase_main
    ;;
  dual_source)
    phase_dual_source
    ;;
  loss_ablation)
    phase_loss_ablation
    ;;
  fuxi_loss)
    phase_fuxi_loss
    ;;
  embedding)
    phase_embedding_ablation
    ;;
  scaling)
    phase_model_scaling
    ;;
  parameter)
    phase_parameter_study
    ;;
  raw_note)
    phase_raw_baseline_note
    ;;
  all)
    phase_raw_baseline_note
    phase_smoke
    phase_main
    phase_loss_ablation
    phase_fuxi_loss
    phase_embedding_ablation
    phase_model_scaling
    phase_parameter_study
    echo
    echo "[INFO] Optional dual-source phase is not included in all by default. Run:"
    echo "       RUN=1 bash ${A2E_ROOT}/scripts/run_recommended_experiments.sh dual_source"
    ;;
  *)
    echo "Unknown phase: ${PHASE}" >&2
    echo "Available phases: smoke, main, dual_source, loss_ablation, fuxi_loss, embedding, scaling, parameter, raw_note, all" >&2
    exit 2
    ;;
esac
