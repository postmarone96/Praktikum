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
#SBATCH -w supergpu04

echo "directory definition"
SCRATCH_DIR=/localscratch/$USER/job_$SLURM_JOB_ID
OUTPUTFILE=$SCRATCH_DIR/dataset.hdf5
TARGET_DIR=$SLURM_SUBMIT_DIR/job_$SLURM_JOB_ID
DATASET=$SLURM_SUBMIT_DIR/job_$JVAE/dataset.hdf5
AE_CHECKPOINT=$SLURM_SUBMIT_DIR/job_$JVAE/vae_checkpoint_epoch_$CP.pth
LDM_CHECKPOINT=$SLURM_SUBMIT_DIR/job_$JLDM/ldm_checkpoint_epoch_$CP.pth
CN_CHECKPOINT="$SLURM_SUBMIT_DIR/job_$JCN/cn_checkpoint_epoch_$CP.pth"
AE=$SLURM_SUBMIT_DIR/job_$JVAE/vae_model_*.pth
LDM="$SLURM_SUBMIT_DIR/job_$JLDM/ldm_model_*.pth"
FILENAME=$(date "+%Y%m%d.txt")
FILEPATH="$TARGET_DIR/$FILENAME"
IDS_FILE="$HOME/Praktikum/ids.txt"

echo "Create the directories"
mkdir -p $SCRATCH_DIR
mkdir -p $SCRATCH_DIR/bg
mkdir -p $SCRATCH_DIR/raw
mkdir -p $SCRATCH_DIR/gt
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
        find $SCRATCH_DIR -type f ! \( -name "*.py" -o -name "*.nii.gz" \) ! -path "$SCRATCH_DIR/raw/*" ! -path "$SCRATCH_DIR/bg/*" ! -path "$SCRATCH_DIR/gt/*" ! -path "$SCRATCH_DIR/pkl_dir/*" | while read -r file; do
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
    find $SCRATCH_DIR -type f ! \( -name "*.py" -o -name "*.nii.gz" \) ! -path "$SCRATCH_DIR/bg/*" ! -path "$SCRATCH_DIR/raw/*" ! -path "$SCRATCH_DIR/gt/*" ! -path "$SCRATCH_DIR/pkl_dir/*" | while read -r file; do
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
    if [ "$SIZE" = 'xl' ]; then
            cp -r $SLURM_SUBMIT_DIR/data/bg "$SCRATCH_DIR"
            cp -r $SLURM_SUBMIT_DIR/data/raw "$SCRATCH_DIR"

    elif [ "$SIZE" = 'xs' ]; then
        SRC_DIR="/lustre/groups/iterm/Annotated_Datasets/Annotated Datasets/UCHL1 HFD - Peripheral Nervous System instance"
        cp -r "$SRC_DIR/bg" "$SCRATCH_DIR"
        cp -r "$SRC_DIR/raw" "$SCRATCH_DIR"
        cp -r "$SRC_DIR/gt" "$SCRATCH_DIR"
    fi

    cp $HOME/Praktikum/preprocessing.py $SCRATCH_DIR/
    python -u preprocessing.py --data_path $SCRATCH_DIR --output_file $OUTPUTFILE --data_size $SIZE
    echo "preprocessing completed"
elif  [ -e "$DATASET" ]; then
    cp $DATASET $SCRATCH_DIR/
    echo "dataset.hdf5 copied successfully."
fi

if [ "$MODEL" = "vae" ]; then
    echo "training vae"
    if [ -e "$AE_CHECKPOINT" ]; then
        cp $AE_CHECKPOINT $SCRATCH_DIR/
        echo "vae_checkpoint copied successfully."
    fi 
    cp $HOME/Praktikum/train_vae.py $SCRATCH_DIR/
    python -u train_vae.py --output_file $OUTPUTFILE --lr $LR --data_size $SIZE 

elif [ "$MODEL" = "ldm" ]; then
    echo "training ldm"
    if [ -e "$LDM_CHECKPOINT" ]; then
        cp $LDM_CHECKPOINT $SCRATCH_DIR/
        echo "ldm_checkpoint copied successfully."
    fi 
    cp $AE $SCRATCH_DIR/
    cp $HOME/Praktikum/train_ldm.py $SCRATCH_DIR/
    python -u train_ldm.py --output_file $OUTPUTFILE --lr $LR --data_size $SIZE --job $SLURM_JOB_ID

elif [ "$MODEL" = "cn" ]; then
    echo "training cn"
    if [ -e "$CN_CHECKPOINT" ]; then
        cp $CN_CHECKPOINT $SCRATCH_DIR/
        echo "cn_checkpoint copied successfully."
    fi
    cp $LDM $SCRATCH_DIR/
    cp $HOME/Praktikum/train_cn.py $SCRATCH_DIR/
    python -u train_cn.py --output_file $OUTPUTFILE --lr $LR
fi

cp $SCRATCH_DIR/samples_*.zip $TARGET_DIR

cleanup
