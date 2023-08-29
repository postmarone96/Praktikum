#!/bin/bash

#SBATCH -c 16
#SBATCH -J train_job
#SBATCH --output=train_pre/train_output_%j.txt
#SBATCH --error=train_pre/train_error_%j.txt
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
cp $SLURM_SUBMIT_DIR/train_pre/job_13214025 $SCRATCH_DIR/
cp $HOME/Praktikum/train_vae.py $SCRATCH_DIR/

echo "change directory"
cd $SCRATCH_DIR

eval "$(conda shell.bash hook)"
conda activate myenv
echo "env activated"

echo "runnning train.py"
python -u train_vae.py --output_file $SCRATCH_DIR/dataset.hdf5

echo "Moving files (excluding .py and .nii.gz) to the job submission directory"
TARGET_DIR=$SLURM_SUBMIT_DIR/train_pre/job_$SLURM_JOB_ID
mkdir -p $TARGET_DIR
find $SCRATCH_DIR -type f ! \( -name "*.py" -o -name "*.nii.gz" \) -print0 | xargs -0 mv -t $TARGET_DIR
 
echo "Clean up"
rm -rf $SCRATCH_DIR

