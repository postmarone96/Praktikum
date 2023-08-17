#!/bin/bash

#SBATCH -J train_job
#SBATCH --output=train_output_%j.txt
#SBATCH --error=train_error_%j.txt
#SBATCH -p gpu_p
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --nice=1000
#SBATCH --mail-user=Marouane.Hajri@helmholtz-munich.de
#SBATCH --qos=gpu_normal
#SBATCH --mail-type=ALL

echo "HELLo"
module load python

source activate iterm_sd_pr

echo "env activated"


echo "runnning train.py"

srun python train.py
