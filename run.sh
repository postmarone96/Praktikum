#!/bin/bash

# Collect arguments or set defaults
MODEL=${1:-""}
SIZE=${2:-""}
LR=${3:-0}
JVAE=${4:-0}
JLDM=${5:-0}
JMASK=${6:-0}
JCN=${7:-0}
CP=${8:-0}

# Export them as environment variables so they can be accessed within the SLURM script
export MODEL
export SIZE
export LR
export JVAE
export JLDM
export JMASK
export JCN
export CP

test -d train_${SIZE} || mkdir train_${SIZE}
cp -f train.slurm train_${SIZE}/
cd train_${SIZE}

# Submit the SLURM job
sbatch train.slurm
