#!/bin/bash

#SBATCH -c 16
#SBATCH -J train_job
#SBATCH --output=train_output_%j.txt
#SBATCH --error=train_error_%j.txt
#SBATCH -p gpu_p
#SBATCH --gres=gpu:1
#SBATCH --mem=150G
#SBATCH --time=2-00:00:00
#SBATCH --nice=1000
#SBATCH --mail-user=marouanehajri@hotmail.com
#SBATCH --qos=gpu_normal
#SBATCH --mail-type=ALL


echo "Hello world!"
eval "$(conda shell.bash hook)"
conda activate myenv
echo "env activated"


echo "runnning train.py"

python -u train.py
