#!/bin/bash

# Collect arguments or set defaults
LR=${1:-0}
JLDM=${2:-0}
CP=${3:-0}

# Export them as environment variables so they can be accessed within the SLURM script
export LR
export JLDM
export CP

test -d train_masks || mkdir train_masks
cp -f train_pic_2_pic.slurm train_masks/
cd train_masks

# Submit the SLURM job
sbatch train_pic_2_pic.slurm