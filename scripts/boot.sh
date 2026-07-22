#!/usr/bin/env bash
bash A2E/scripts/run_one.sh A2Ec70_gfs_refnorm \
  SOURCES=gfs EPOCHS=90 FUXI_LOSS_MODE=reference_norm CHANNEL_RMSE_WEIGHT=8e-3




bash A2E/scripts/run_one.sh A2Ec70_refnorm_w4em3 \
  SOURCES=gfs EPOCHS=90 FUXI_LOSS_MODE=reference_norm CHANNEL_RMSE_WEIGHT=4e-3

bash A2E/scripts/run_one.sh A2Ec70_refnorm_w1em2 \
  SOURCES=gfs EPOCHS=90 FUXI_LOSS_MODE=reference_norm CHANNEL_RMSE_WEIGHT=1e-2

bash A2E/scripts/run_one.sh A2Ec70_refnorm_w2em3 \
  SOURCES=gfs EPOCHS=90 FUXI_LOSS_MODE=reference_norm CHANNEL_RMSE_WEIGHT=2e-3


torchrun --nproc_per_node=4 /home/ximutian/A2E/occ.py

# tensorboard --logdir /cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/A2E/Formal/tensorboard_logs