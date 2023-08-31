#!/bin/bash

# Collect arguments or set defaults
ARG1=${1:-""}
ARG2=${2:-""}

# Export them as environment variables so they can be accessed within the SLURM script
export ARG1
export ARG2

working_dir = $HOME/Praktikum/train_${ARG2}

mkdir -p $WORKING_DIR

cp -n $HOME/Praktikum/train.slurm $WORKING_DIR/

cd $WORKING_DIR

# Submit the SLURM job
sbatch train.slurm
