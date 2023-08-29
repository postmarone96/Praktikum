#!/bin/bash

# Collect arguments or set defaults
ARG1=${1:-""}
ARG2=${2:-""}
# Export them as environment variables so they can be accessed within the SLURM script
export ARG1
export ARG2
# Submit the SLURM job
sbatch train.slurm
