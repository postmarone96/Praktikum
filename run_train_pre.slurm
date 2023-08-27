#!/bin/bash

#SBATCH -c 16
#SBATCH -J train_job
#SBATCH --output=train_pre_output_%j.txt
#SBATCH --error=train_pre_error_%j.txt
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

cleanup() {
    echo "Job was terminated. Cleaning up..."
    rm -rf $SCRATCH_DIR
    # Add any other cleanup operations you need here.
}
trap 'cleanup' SIGTERM

echo "Copy data to scratch"
cp -r /lustre/groups/iterm/Rami/HFD_neurons/HFD_210320_UCHL1_755_HFD_DORSAL_l_1x_35o_4x8_GRB12-7-12_17-00-17/C00/* $SCRATCH_DIR/
cp $HOME/Praktikum/train_pre.py $SCRATCH_DIR/
cp $HOME/Praktikum/preprocessing.py $SCRATCH_DIR/

echo "getting size of raw data"
find $SCRATCH_DIR -type f -name "*.nii.gz" -print0 | xargs -0 du -ch | grep total

echo "change directory"
cd $SCRATCH_DIR

eval "$(conda shell.bash hook)"
conda activate myenv
echo "env activated"

echo "run preprocessing"
OUTPUTFILE=$SCRATCH_DIR/dataset.hdf5
python -u preprocessing.py --datapath $SCRATCH_DIR

echo "get size of preprocessed dataset"
du -h $OUTPUTFILE

echo "runnning train.py"
python -u train_pre.py --output_file $OUTPUTFILE

echo "Moving files (excluding .py and .nii.gz) to the job submission directory"
TARGET_DIR=$SLURM_SUBMIT_DIR/job_$SLURM_JOB_ID
mkdir -p $TARGET_DIR
find $SCRATCH_DIR -type f ! \( -name "*.py" -o -name "*.nii.gz" \) -print0 | xargs -0 mv -t $TARGET_DIR
 
echo "Clean up"
rm -rf $SCRATCH_DIR
