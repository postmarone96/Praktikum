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

SCRATCH_DIR="/localscratch/$USER/job_$SLURM_JOB_ID"
mkdir -p "$SCRATCH_DIR"

cp $HOME/Praktikum/preprocessing.py $SCRATCH_DIR/

cd $SCRATCH_DIR

eval "$(conda shell.bash hook)"
conda activate myenv
echo "env activated"


##python -u create_dataset.py

python -u preprocessing.py --data_path $SLURM_SUBMIT_DIR/train_xs/data --output_file $SLURM_SUBMIT_DIR/train_xs/data/datasetxy.hdf5 --data_size xs

rm -rf "$SCRATCH_DIR"

