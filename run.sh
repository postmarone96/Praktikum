#!/bin/bash

# Collect arguments or set defaults
ARG1=${1:-""}
ARG2=${2:-""}
ARG3=${3:-20}

# Export them as environment variables so they can be accessed within the SLURM script
export ARG1
export ARG2
export ARG3
 
test -d train_${ARG2} || mkdir train_${ARG2}

cp -f train.slurm train_${ARG2}/

cd train_${ARG2}

# Submit the SLURM job
sbatch train.slurm
