#!/bin/bash

#SBATCH -c 16
#SBATCH -J train_job
#SBATCH --output=o_data_%j.txt
#SBATCH --error=e_data_%j.txt
#SBATCH -p gpu_p
#SBATCH --gres=gpu:2
#SBATCH --mem=160G
#SBATCH --time=2-00:00:00
#SBATCH --nice=1000
#SBATCH --qos=gpu_normal

eval "$(conda shell.bash hook)"
conda activate myenv
echo "env activated"

cp $HOME/Praktikum/create_dataset.py $SCRATCH_DIR/
python -u create_dataset.py