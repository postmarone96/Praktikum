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

echo "Define the scratch directory"
SCRATCH_DIR=/localscratch/$USER/job_$SLURM_JOB_ID

echo "Create the directory"
mkdir -p $SCRATCH_DIR

echo "Copy data to scratch"
cp -r /lustre/groups/iterm/Rami/HFD_neurons/HFD_210320_UCHL1_755_HFD_DORSAL_l_1x_35o_4x8_GRB12-7-12_17-00-17/C00/ $SCRATCH_DIR/
cp $HOME/Praktikum/train.py $SCRATCH_DIR/

echo "change directory"
cd $SCRATCH_DIR

eval "$(conda shell.bash hook)"
conda activate myenv
echo "env activated"

echo "runnning train.py"
python -u train.py --data_path $SCRATCH_DIR/C00 --batch_size 64
 
echo "Clean up"
rm -rf $SCRATCH_DIR

