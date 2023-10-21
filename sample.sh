#!/bin/bash

# Collect arguments or set defaults
ARG1=${1:-0}
ARG2=${1:-0}

# Export them as environment variables so they can be accessed within the SLURM script
export ARG1
export ARG2

cp -f sample.slurm train_${ARG1}/

cd train_${ARG1}

# Submit the SLURM job
sbatch sample.slurm
