#!/bin/bash

#SBATCH -c 16
#SBATCH -J train_job
#SBATCH --output=o_2_%j.txt
#SBATCH --error=e_2_%j.txt
#SBATCH -p gpu_p
#SBATCH --gres=gpu:2
#SBATCH --mem=160G
#SBATCH --time=2-00:00:00
#SBATCH --nice=1000
#SBATCH --qos=gpu_normal

echo "directory definition"
SCRATCH_DIR=/localscratch/$USER/job_$SLURM_JOB_ID
OUTPUTFILE=$SCRATCH_DIR/dataset.hdf5
TARGET_DIR=$SLURM_SUBMIT_DIR/job_$SLURM_JOB_ID
DATASET=$HOME/Praktikum/train_$ARG2/job_$ARG4/dataset.hdf5
AE_CHECKPOINT=$HOME/Praktikum/train_$ARG2/job_$ARG4/vae_checkpoint_epoch_$ARG6.pth
LDM_CHECKPOINT=$HOME/Praktikum/train_$ARG2/job_$ARG5/ldm_checkpoint_epoch_$ARG6.pth
AE=$HOME/Praktikum/train_$ARG2/job_$ARG4/vae_model_*.pth
FILENAME=$(date "+%Y%m%d.txt")
FILEPATH="$TARGET_DIR/$FILENAME"
IDS_FILE="$HOME/Praktikum/ids.txt"

echo "Create the directories"
mkdir -p $SCRATCH_DIR
mkdir -p $SCRATCH_DIR/bg
mkdir -p $SCRATCH_DIR/raw
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
        sleep 4m
        find $SCRATCH_DIR -type f ! \( -name "*.py" -o -name "*.nii.gz" \) ! -path "$SCRATCH_DIR/raw/*" ! -path "$SCRATCH_DIR/bg/*" | while read -r file; do
            basefile=$(basename "$file")
            if [[ "$basefile" == *.png ]] || [[ "$basefile" == *.hdf5 ]]; then
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
    find $SCRATCH_DIR -type f ! \( -name "*.py" -o -name "*.nii.gz" \) ! -path "$SCRATCH_DIR/bg/*" ! -path "$SCRATCH_DIR/raw/*" | while read -r file; do
        basefile=$(basename "$file")
        if [[ "$basefile" == *.png ]] || [[ "$basefile" == *.hdf5 ]]; then
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
    if [ "$ARG2" = 'xl' ]; then
        while IFS= read -r id; do
            find "/lustre/groups/iterm/Rami/HFD_neurons/HFD_210320_UCHL1_755_HFD_DORSAL_l_1x_35o_4x8_GRB12-7-12_17-00-17/C00/" -type f -name "*${id}.nii.gz" -exec cp {} "$SCRATCH_DIR/bg/" \;
        done < "$IDS_FILE"

        while IFS= read -r id; do
            find "/lustre/groups/iterm/Rami/HFD_neurons/HFD_210320_UCHL1_755_HFD_DORSAL_l_1x_35o_4x8_GRB12-7-12_17-00-17/C02/" -type f -name "*${id}.nii.gz" -exec cp {} "$SCRATCH_DIR/raw/" \;
        done < "$IDS_FILE"

    elif [ "$ARG2" = 'xs' ]; then
        SRC_DIR="/lustre/groups/iterm/Annotated_Datasets/Annotated Datasets/UCHL1 HFD - Peripheral Nervous System instance"
        cp -r "$SRC_DIR/bg" "$SCRATCH_DIR/"
        cp -r "$SRC_DIR/raw" "$SCRATCH_DIR/"
        cp -r "$SRC_DIR/gt" "$SCRATCH_DIR/"
    fi

    cp $HOME/Praktikum/preprocessing_2_dim.py $SCRATCH_DIR/
    python -u preprocessing_2_dim.py --data_path $SCRATCH_DIR --output_file $OUTPUTFILE 
    echo "preprocessing completed"
elif  [ -e "$DATASET" ]; then
    cp $DATASET $SCRATCH_DIR/
    echo "dataset.hdf5 copied successfully."
fi

if [ "$ARG1" = "vae" ]; then
    echo "training vae"
    if [ -e "$AE_CHECKPOINT" ]; then
        cp $AE_CHECKPOINT $SCRATCH_DIR/
        echo "vae_checkpoint copied successfully."
    fi 
    cp $HOME/Praktikum/train_vae_2_dim.py $SCRATCH_DIR/
    python -u train_vae_2_dim.py --output_file $OUTPUTFILE --lr $ARG3
elif [ "$ARG1" = "ldm" ]; then
    echo "training ldm"
    if [ -e "$LDM_CHECKPOINT" ]; then
        cp $LDM_CHECKPOINT $SCRATCH_DIR/
        echo "ldm_checkpoint copied successfully."
    fi 
    cp $AE $SCRATCH_DIR/
    cp $HOME/Praktikum/train_ldm_2_dim.py $SCRATCH_DIR/
    python -u train_ldm_2_dim.py --output_file $OUTPUTFILE --lr $ARG3
fi

cleanup
