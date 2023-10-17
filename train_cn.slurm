#!/bin/bash

#SBATCH -c 16
#SBATCH -J train_job
#SBATCH --output=o_cn_%j.txt
#SBATCH --error=e_cn_%j.txt
#SBATCH -p gpu_p
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=2-00:00:00
#SBATCH --nice=1000
#SBATCH --qos=gpu_normal

echo "directory definition"
SCRATCH_DIR="/localscratch/$USER/job_$SLURM_JOB_ID"
OUTPUTFILE="$SCRATCH_DIR/dataset.hdf5"
TARGET_DIR="$SLURM_SUBMIT_DIR/job_$SLURM_JOB_ID"
DATASET="$HOME/Praktikum/train_$ARG2/job_$ARG4/dataset.hdf5"
CN_CHECKPOINT="$HOME/Praktikum/train_$ARG2/job_$ARG4/cn_checkpoint_epoch_$ARG5.pth"
LDM="$HOME/Praktikum/train_$ARG2/job_$ARG4/ldm_model_*.pth"
FILENAME="$(date "+%Y%m%d.txt")"
FILEPATH="$TARGET_DIR/$FILENAME"

echo "Create the directories"
mkdir -p $SCRATCH_DIR
mkdir -p $TARGET_DIR
touch "$FILEPATH"

log_and_copy() {
    local src=$1
    local dest=$2
    echo "$(basename "$src")" >> $FILEPATH
    cp "$src" "$dest"
}
periodic_backup() {
    while true; do
        sleep 4m
        find $SCRATCH_DIR -type f ! \( -name "*.py" -o -name "*.nii.gz" \) ! -path "$SCRATCH_DIR/bg/*" ! -path "$SCRATCH_DIR/raw/*" ! -path "$SCRATCH_DIR/gt/*" | while read -r file; do
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
    find $SCRATCH_DIR -type f ! \( -name "*.py" -o -name "*.nii.gz" \) ! -path "$SCRATCH_DIR/bg/*" ! -path "$SCRATCH_DIR/raw/*" ! -path "$SCRATCH_DIR/gt/*" | while read -r file; do
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

# Copy files to scratch directory
SRC_DIR="/lustre/groups/iterm/Annotated_Datasets/Annotated Datasets/UCHL1 HFD - Peripheral Nervous System instance"
cp -r "$SRC_DIR/bg" "$SCRATCH_DIR/"
cp -r "$SRC_DIR/raw" "$SCRATCH_DIR/"
cp -r "$SRC_DIR/gt" "$SCRATCH_DIR/"

cp $HOME/Praktikum/preprocessing_cn.py $SCRATCH_DIR/
python -u preprocessing_cn.py --data_path $SCRATCH_DIR --output_file $OUTPUTFILE 
echo "preprocessing completed"

echo "training ControlNet"
if [ -e "$CN_CHECKPOINT" ]; then
    cp $CN_CHECKPOINT $SCRATCH_DIR/
    echo "cn_checkpoint copied successfully."
fi 

cp $AE $SCRATCH_DIR/
cp $HOME/Praktikum/train_cn.py $SCRATCH_DIR/
python -u train_cn.py --output_file $OUTPUTFILE --lr $ARG3

cleanup
