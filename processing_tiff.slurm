#!/bin/bash

#SBATCH -c 16
#SBATCH -J train_job
#SBATCH --output=o_%j.txt
#SBATCH --error=e_%j.txt
#SBATCH -p gpu_p
#SBATCH --gres=gpu:2
#SBATCH --mem=160G
#SBATCH --time=2-00:00:00
#SBATCH --nice=1000
#SBATCH --qos=gpu_normal
##SBATCH -w supergpu02

P_DIR=/home/viro/marouane.hajri/Praktikum
SCRATCH_DIR=/localscratch/marouane.hajri/job_$SLURM_JOB_ID

eval "$(conda shell.bash hook)"
conda activate myenv
echo "env activated"

cp $P_DIR/preprocessing_tiff.py $SCRATCH_DIR/
cd $SCRATCH_DIR

python -u preprocessing_tiff.py