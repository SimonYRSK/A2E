#!/usr/bin/env bash
#RUN=1 bash A2E/scripts/train_interface.sh r

MASTER_PORT=29517 RUN=1 bash A2E/scripts/train_interface.sh 0






# RUN=1 bash /home/ximutian/A2E/scripts/eval_one.sh A2Ec70_gfs_refnorm \
#   EVAL_SOURCES=gfs \
#   EVAL_DATES=${EVAL_DATES} \
#   EVAL_VARIABLES=z500,t2m,t850,ws10,ws850,msl


RUN=1 bash A2E/scripts/train_interface.sh 6


torchrun --nproc_per_node=4 /home/ximutian/A2E/occ.py

# tensorboard --logdir /cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/A2E/Formal/tensorboard_logs