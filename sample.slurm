#!/bin/bash

#SBATCH -c 16
#SBATCH -J train_job
#SBATCH --output=o_sample_%j.txt
#SBATCH --error=e_sample_%j.txt
#SBATCH -p gpu_p
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=2-00:00:00
#SBATCH --nice=1000
#SBATCH --qos=gpu_normal

# Directory Definitions
echo "Directory definition"
SCRATCH_DIR="/localscratch/$USER/job_$SLURM_JOB_ID"
TARGET_DIR="$SLURM_SUBMIT_DIR/sample_$SLURM_JOB_ID"


# Directory Setup
echo "Creating the directories"
mkdir -p "$SCRATCH_DIR"
mkdir -p "$TARGET_DIR"

# Copy files to scratch directory
cp -r "$SLURM_SUBMIT_DIR/job_$ARG2/vae_model*.pth" "$SCRATCH_DIR/"
cp -r "$SLURM_SUBMIT_DIR/job_$ARG2/ldm_model*.pth" "$SCRATCH_DIR/"
cp -r "$HOME/Praktikum/get_diff_samples.py" $SCRATCH_DIR/

# Move to the local scratch directory
echo "Changing to local scratch directory"
cd "$SCRATCH_DIR"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate myenv
echo "Environment activated"

#start the python script
python -u get_diff_samples.py

cp -r "$SCRATCH_DIR/*.svg" "$TARGET_DIR/"
cp -r "$SCRATCH_DIR/diff_raw" "$TARGET_DIR/"
cp -r "$SCRATCH_DIR/diff_bg" "$TARGET_DIR/"

rm -rf "$SCRATCH_DIR"

