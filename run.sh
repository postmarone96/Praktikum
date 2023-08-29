#!/bin/bash

# Collect arguments or set defaults
ARG1=${1:-""}

# Export them as environment variables so they can be accessed within the SLURM script
export ARG1

# Submit the SLURM job
sbatch job.slurm
