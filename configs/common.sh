#!/usr/bin/env bash
# Common configuration for A2E-c70 experiments.
# Source this file from run/eval scripts, then override only the variables that differ.

# -------------------------
# Runtime
# -------------------------
export NPROC_PER_NODE=${NPROC_PER_NODE:-4}
export MASTER_PORT=${MASTER_PORT:-29500}
export PYTHON_BIN=${PYTHON_BIN:-python}

# -------------------------
# Paths
# -------------------------
export A2E_ROOT=${A2E_ROOT:-E:/myrepo/A2E}
export PROJECT_ROOT=${PROJECT_ROOT:-/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/A2E}
export DATA_ROOT=${DATA_ROOT:-/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/data}

export ERA5_PATH=${ERA5_PATH:-${DATA_ROOT}/era5.2020_2025_norm.zarr}
export GFS_PATH=${GFS_PATH:-${DATA_ROOT}/gfs_2020_2025_c226_0p25_norm.zarr}
export HRES_PATH=${HRES_PATH:-${DATA_ROOT}/hres_2024_2025_c226_0p25_norm.zarr}
export CMA_PATH=${CMA_PATH:-${DATA_ROOT}/cma_gfs_2020_2026.c226.norm.zarr}
export FUXI_DIR=${FUXI_DIR:-/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/fuxi_inference/main/fuxi}
export CLIM_PATH=${CLIM_PATH:-/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/fanjiang/eval/era5/clim.daily}

export OUTPUT_ROOT=${OUTPUT_ROOT:-${PROJECT_ROOT}/experiments}
export CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${PROJECT_ROOT}/checkpoints}
export TENSORBOARD_ROOT=${TENSORBOARD_ROOT:-/home/ximutian/tensorboard_logs}
export PLOT_ROOT=${PLOT_ROOT:-${PROJECT_ROOT}/channelpics}
export INFERENCE_ROOT=${INFERENCE_ROOT:-${PROJECT_ROOT}/inference_results}

# -------------------------
# Data split
# -------------------------
export SOURCES=${SOURCES:-cma}
export TRAIN_START=${TRAIN_START:-2022-01-01 00:00:00}
export TRAIN_END=${TRAIN_END:-2024-12-31 18:00:00}
export VAL_START=${VAL_START:-2025-01-01 00:00:00}
export VAL_END=${VAL_END:-2025-11-20 18:00:00}
export VAL_SAMPLE_PER_MONTH=${VAL_SAMPLE_PER_MONTH:-4}
export VAL_SAMPLE_YEAR=${VAL_SAMPLE_YEAR:-2025}
export MAX_SAMPLES_PER_YEAR=${MAX_SAMPLES_PER_YEAR:-none}
export SAMPLE_SEED=${SAMPLE_SEED:-43}

# HRES has only 2024-2025 in current data layout.
export GFS_TRAIN_START=${GFS_TRAIN_START:-${TRAIN_START}}
export GFS_TRAIN_END=${GFS_TRAIN_END:-${TRAIN_END}}
export GFS_VAL_START=${GFS_VAL_START:-${VAL_START}}
export GFS_VAL_END=${GFS_VAL_END:-${VAL_END}}
export CMA_TRAIN_START=${CMA_TRAIN_START:-${TRAIN_START}}
export CMA_TRAIN_END=${CMA_TRAIN_END:-${TRAIN_END}}
export CMA_VAL_START=${CMA_VAL_START:-${VAL_START}}
export CMA_VAL_END=${CMA_VAL_END:-${VAL_END}}
export HRES_TRAIN_START=${HRES_TRAIN_START:-2024-01-01 00:00:00}
export HRES_TRAIN_END=${HRES_TRAIN_END:-2024-12-31 18:00:00}
export HRES_VAL_START=${HRES_VAL_START:-${VAL_START}}
export HRES_VAL_END=${HRES_VAL_END:-${VAL_END}}

# Paper/evaluation variables. Q700 was corrected to r700.
export EVAL_VARIABLES=${EVAL_VARIABLES:-z500,t2m,tp,ws10m,msl,r700}

# -------------------------
# Model config
# -------------------------
export MODEL_NAME=${MODEL_NAME:-A2E-c70-full}
export IMG_SIZE=${IMG_SIZE:-721,1440}
export PATCH_SIZE=${PATCH_SIZE:-4,4}
export IN_CHANS=${IN_CHANS:-70}
export OUT_CHANS=${OUT_CHANS:-70}
export EMBED_DIM=${EMBED_DIM:-384}
export CHANNELS=${CHANNELS:-384,768,1536}
export NUM_GROUPS=${NUM_GROUPS:-32}
export NUM_HEADS=${NUM_HEADS:-8}
export NUM_STAGES=${NUM_STAGES:-3}
export WINDOW_SIZE=${WINDOW_SIZE:-9}
export DEPTH=${DEPTH:-0,0,1}
export RES_PER_STAGE=${RES_PER_STAGE:-1,1,1}
export USING_TIME_EMBEDDING=${USING_TIME_EMBEDDING:-true}
export USING_SOURCE_EMBEDDING=${USING_SOURCE_EMBEDDING:-true}
export USING_KL=${USING_KL:-false}
export DROPOUT_RATE=${DROPOUT_RATE:-0.1}
export USE_SKIP_CONNECTIONS=${USE_SKIP_CONNECTIONS:-true}
export USE_RESIDUAL_BLOCKS=${USE_RESIDUAL_BLOCKS:-true}
export USING_DANN=${USING_DANN:-false}
export DOMAIN_LOSS_WEIGHT=${DOMAIN_LOSS_WEIGHT:-1e-3}
export DANN_GAMMA=${DANN_GAMMA:-10.0}

# -------------------------
# Training config
# Main experiments show 90 epochs is usually enough.
# -------------------------
export SEED=${SEED:-42}
export EPOCHS=${EPOCHS:-90}
export BATCH_SIZE=${BATCH_SIZE:-8}
export NUM_WORKERS=${NUM_WORKERS:-4}
export PREFETCH_FACTOR=${PREFETCH_FACTOR:-1}
export BASE_LR=${BASE_LR:-2e-4}
export MIN_LR=${MIN_LR:-1e-7}
export WARMUP_EPOCHS=${WARMUP_EPOCHS:-5}
export WEIGHT_DECAY=${WEIGHT_DECAY:-2e-5}
export BETAS=${BETAS:-0.9,0.999}
export SAVE_INTERVAL=${SAVE_INTERVAL:-1}
export USE_AMP=${USE_AMP:-false}
export GRAD_CLIP_NORM=${GRAD_CLIP_NORM:-5.0}

# -------------------------
# Loss config
# -------------------------
export RECON_LOSS_TYPE=${RECON_LOSS_TYPE:-l1}
export CHARBONNIER_EPS=${CHARBONNIER_EPS:-1e-3}
export USE_GRAD_LOSS=${USE_GRAD_LOSS:-true}
export GRAD_LOSS_WEIGHT=${GRAD_LOSS_WEIGHT:-0.4}
export BETA=${BETA:-1e-4}
export KL_ANNEAL=${KL_ANNEAL:-false}
export KL_ANNEAL_EPOCHS=${KL_ANNEAL_EPOCHS:-7}
export L1_REG_WEIGHT=${L1_REG_WEIGHT:-0.0}
export L2_REG_WEIGHT=${L2_REG_WEIGHT:-0.0}

# FuXi downstream loss. reference_norm should normally pair with 4e-3.
export FUXI_LOSS_MODE=${FUXI_LOSS_MODE:-reference_norm}
export CHANNEL_RMSE_WEIGHT=${CHANNEL_RMSE_WEIGHT:-4e-3}
export RMSE_EVERY_N_STEPS=${RMSE_EVERY_N_STEPS:-1}
export RMSE_SAMPLES_PER_BATCH=${RMSE_SAMPLES_PER_BATCH:-1}
export FUXI_LEAD_HOURS=${FUXI_LEAD_HOURS:-6}

# -------------------------
# Evaluation config
# -------------------------
export EVAL_DATES=${EVAL_DATES:-20250101}
export EVAL_SOURCES=${EVAL_SOURCES:-${SOURCES}}
export EVAL_FORECAST_STEPS=${EVAL_FORECAST_STEPS:-40}
