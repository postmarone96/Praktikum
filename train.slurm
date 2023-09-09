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
#SBATCH --qos=gpu_normal

echo "directory definition"
SCRATCH_DIR=/localscratch/$USER/job_$SLURM_JOB_ID
OUTPUTFILE=$SCRATCH_DIR/dataset.hdf5
TARGET_DIR=$SLURM_SUBMIT_DIR/job_$SLURM_JOB_ID
DATASET=$HOME/Praktikum/train_$ARG2/job_$ARG4/dataset.hdf5
AE_CHECKPOINT=$HOME/Praktikum/train_$ARG2/job_$ARG4/vae_best_checkpoint.pth
LDM_CHECKPOINT=$HOME/Praktikum/train_$ARG2/job_$ARG5/ldm_best_checkpoint.pth
AE=$HOME/Praktikum/train_$ARG2/job_$ARG4/vae_model_*.pth
FILENAME=$(date "+%Y%m%d.txt")
FILEPATH="$TARGET_DIR/$FILENAME"

echo "Create the directories"
mkdir -p $SCRATCH_DIR
mkdir -p $TARGET_DIR
touch $FILEPATH

log_and_copy() {
    local src=$1
    local dest=$2
    echo "$(basename "$src")" >> $FILEPATH
    cp "$src" "$dest"
}
periodic_backup() {
    while true; do
        sleep 9m
        find $SCRATCH_DIR -type f ! \( -name "*.py" -o -name "*.nii.gz" \) | while read -r file; do
            basefile=$(basename "$file")
            if [[ "$basefile" == *.png ]]; then
                cp "$file" "$TARGET_DIR/$basefile"
            else
                if ! grep -q "$basefile" "$FILEPATH"; then
                    log_and_copy "$file" "$TARGET_DIR"
                fi
            fi
        done
    done
}

cleanup() {
    echo "Final backup and cleanup..."
    find $SCRATCH_DIR -type f ! \( -name "*.py" -o -name "*.nii.gz" \) | while read -r file; do
        basefile=$(basename "$file")
        if [[ "$basefile" == *.png ]]; then
            cp "$file" "$TARGET_DIR/$basefile"
        else
            if ! grep -q "$basefile" "$FILEPATH"; then
                log_and_copy "$file" "$TARGET_DIR"
            fi
        fi
    done
    kill $BACKUP_PID
    rm -rf $SCRATCH_DIR
    echo "Cleanup complete."
}

trap 'cleanup' SIGTERM

echo "change to local scratch directory"
cd $SCRATCH_DIR

eval "$(conda shell.bash hook)"
conda activate myenv
echo "env activated"

periodic_backup &
BACKUP_PID=$!

if [ ! -e "$DATASET" ]; then
    find /lustre/groups/iterm/Rami/HFD_neurons/HFD_210320_UCHL1_755_HFD_DORSAL_l_1x_35o_4x8_GRB12-7-12_17-00-17/C00/ -type f | shuf -n $ARG2 | xargs -I {} cp {} $SCRATCH_DIR/
    cp $HOME/Praktikum/preprocessing.py $SCRATCH_DIR/
    python -u preprocessing.py --data_path $SCRATCH_DIR --output_file $OUTPUTFILE 
    echo "preprocessing completed"
elif  [ -e "$DATASET" ]; then
    cp $DATASET $SCRATCH_DIR/
    echo "dataset.hdf5 copied successfully."
fi

if [ "$ARG1" = "vae" ]; then
    echo "training vae"
    if [ -e "$AE_CHECKPOINT" ]; then
        cp $AE_CHECKPOINT $SCRATCH_DIR/
        echo "ae_checkpoint copied successfully."
    fi 
    cp $HOME/Praktikum/train_vae.py $SCRATCH_DIR/
    python -u train_vae.py --output_file $OUTPUTFILE --lr $ARG3
elif [ "$ARG1" = "ldm" ]; then
    echo "training ldm"
    if [ -e "$LDM_CHECKPOINT" ]; then
        cp $LDM_CHECKPOINT $SCRATCH_DIR/
        echo "ldm_checkpoint copied successfully."
    fi 
    cp $AE $SCRATCH_DIR/
    cp $HOME/Praktikum/train_ldm.py $SCRATCH_DIR/
    python -u train_ldm.py --output_file $OUTPUTFILE --lr $ARG3
fi

cleanup
