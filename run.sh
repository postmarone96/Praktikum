#!/bin/bash

# Collect arguments or set defaults
ARG1=${1:-""}
ARG2=${2:-""}

# Export them as environment variables so they can be accessed within the SLURM script
export ARG1
export ARG2
 
mkdir -p $HOME/Praktikum/train_${ARG2}

cp -n $HOME/Praktikum/train.slurm $HOME/Praktikum/train_${ARG2}/

cd $HOME/Praktikum/train_${ARG2}

# Submit the SLURM job
sbatch train.slurm
