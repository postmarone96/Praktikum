#!/bin/bash

# Collect arguments or set defaults
ARG1=${1:-""}
ARG2=${2:-0}
ARG3=${3:-0}
ARG4=${4:-0}
ARG5=${5:-0}
ARG6=${6:-0}

# Export them as environment variables so they can be accessed within the SLURM script
export ARG1
export ARG2
export ARG3
export ARG4
export ARG5
export ARG6

test -d train_${ARG2} || mkdir train_${ARG2}

cp -f train_2_dim_anno.slurm train_${ARG2}/

cd train_${ARG2}

# Submit the SLURM job
sbatch train_2_dim_anno.slurm
