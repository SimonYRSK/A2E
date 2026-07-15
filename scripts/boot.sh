#!/usr/bin/env bash



# MASTER_PORT=29517 RUN=1 bash A2E/scripts/run_one.sh A2Ec70_gfs_refnorm \
#   SOURCES=gfs \
#   EPOCHS=90 \
#   FUXI_LOSS_MODE=reference_norm \
#   CHANNEL_RMSE_WEIGHT=4e-3

MASTER_PORT=29517 RUN=1 bash A2E/scripts/run_interface.sh 0



# MASTER_PORT=29517 RUN=1 bash A2E/scripts/run_one.sh A2Ec70_gfs_refnorm \
#   SOURCES=gfs \
#   EPOCHS=90 \
#   FUXI_LOSS_MODE=reference_norm \
#   CHANNEL_RMSE_WEIGHT=4e-3


#MASTER_PORT=29523 RUN=1 bash A2E/scripts/run_interface.sh 6