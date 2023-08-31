#!/bin/bash

#SBATCH -c 16
#SBATCH -J train_job
#SBATCH --output=o_%j.txt
#SBATCH --error=e_%j.txt
#SBATCH -p gpu_p
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=2-00:00:00
#SBATCH --nice=1000
##SBATCH --mail-user=Marouane.Hajri@helmholtz-munich.de
#SBATCH --qos=gpu_normal
##SBATCH --mail-type=ALL

echo "directory definition"
SCRATCH_DIR=/localscratch/$USER/job_$SLURM_JOB_ID
OUTPUTFILE=$SCRATCH_DIR/dataset.hdf5
TARGET_DIR=$SLURM_SUBMIT_DIR/job_$SLURM_JOB_ID
DATASET=$HOME/Praktikum/train_$ARG2/job_$ARG1/dataset.hdf5
CHECKPOINT=$HOME/Praktikum/train_$ARG2/job_$ARG4/vae_best_checkpoint.pth

echo "Create the directory"
mkdir -p $SCRATCH_DIR

cleanup() {
    echo "Job was terminated. Cleaning up..."
    mkdir -p $TARGET_DIR
    find $SCRATCH_DIR -type f ! \( -name "*.py" -o -name "*.nii.gz" \) -print0 | xargs -0 mv -t $TARGET_DIR
    rm -rf $SCRATCH_DIR
}
trap 'cleanup' SIGTERM

echo "change to local scratch directory"
cd $SCRATCH_DIR

eval "$(conda shell.bash hook)"
conda activate myenv
echo "env activated"

if [ ! -e "$DATASET" ]; then
    find /lustre/groups/iterm/Rami/HFD_neurons/HFD_210320_UCHL1_755_HFD_DORSAL_l_1x_35o_4x8_GRB12-7-12_17-00-17/C00/ -type f | shuf -n $ARG2 | xargs -I {} cp {} $SCRATCH_DIR/
    cp $HOME/Praktikum/preprocessing.py $SCRATCH_DIR/
    python -u preprocessing.py --data_path $SCRATCH_DIR --output_file $OUTPUTFILE
    echo "preprocessing completed"

elif  [ -e "$DATASET" ]; then
    cp $HOME/Praktikum/train_$ARG2/job_$ARG4/dataset.hdf5 $SCRATCH_DIR/
    echo "dataset.hdf5 copied successfully."
fi

if [ -e "$CHECKPOINT" ]; then
    cp $CHECKPOINT $SCRATCH_DIR/
    echo "checkpoint copied successfully."
fi 

if [ "$ARG1" = "vae" ]; then
    echo "training vae"
    cp $HOME/Praktikum/train_vae.py $SCRATCH_DIR/
    python -u train_vae.py --output_file $OUTPUTFILE --batch_size $ARG3
elif [ "$ARG1" = "ldm" ]; then
    echo "training ldm"
    cp $HOME/Praktikum/train_ldm.py $SCRATCH_DIR/
    python -u train_ldm.py --output_file $OUTPUTFILE --batch_size $ARG3
fi
    
echo "Moving files (excluding .py and .nii.gz) to the job submission directory"
mkdir -p $TARGET_DIR
find $SCRATCH_DIR -type f ! \( -name "*.py" -o -name "*.nii.gz" \) -print0 | xargs -0 mv -t $TARGET_DIR
 
echo "Clean up"
rm -rf $SCRATCH_DIR
