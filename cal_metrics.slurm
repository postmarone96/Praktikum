#!/bin/bash

#SBATCH -c 16
#SBATCH -J train_job
#SBATCH --output=o_%j.txt
#SBATCH --error=e_%j.txt
#SBATCH -p gpu_p
#SBATCH --gres=gpu:2
#SBATCH --mem=160G
#SBATCH --time=2-00:00:00
#SBATCH --nice=1000
#SBATCH --qos=gpu_normal
##SBATCH -w supergpu02

## Initiate directories
SCRATCH_DIR=/localscratch/$USER/job_$SLURM_JOB_ID
DATASET=$P_DIR/train_xs/data/test/test.hdf5
VAE_LDM_TXT=$P_DIR/vae_ldm.txt
CN_TXT=$P_DIR/cn.txt
TARGET_DIR=$SLURM_SUBMIT_DIR/${MODEL}
RADNET=$P_DIR/RadImageNet-ResNet50_notop.pth

## Create directories
mkdir $TARGET_DIR
mkdir $SCRATCH_DIR

FILENAME=$(date "+%Y%m%d.txt")
FILEPATH=$TARGET_DIR/$FILENAME
touch $FILEPATH
log_and_copy() {
    local src=$1
    local dest=$2
    echo "$(basename "$src")" >> "$FILEPATH"
    cp "$src" "$dest"
}
cleanup() {
    echo "Final backup and cleanup..."
    find $SCRATCH_DIR -type f ! \( -name "*.py" -o -name "*.nii.gz" -o -name "*.pyc" \) ! -path "$SCRATCH_DIR/bg/*" ! -path "$SCRATCH_DIR/raw/*" ! -path "$SCRATCH_DIR/gt/*" ! -path "$SCRATCH_DIR/pkl_dir/*" | while read -r file; do
        basefile=$(basename $file)
        if [[ $basefile == *.png ]] || [[ $basefile == *.hdf5 ]]; then
            cp $file $TARGET_DIR/$basefile
        else
            if ! grep -q $basefile $FILEPATH; then
                log_and_copy $file $TARGET_DIR
            fi
        fi
    done
    ## remove the output and error because they are already stored in the job file
    #rm "$SLURM_SUBMIT_DIR/o_$SLURM_JOB_ID.txt"
    #rm "$SLURM_SUBMIT_DIR/e_$SLURM_JOB_ID.txt"

    kill $BACKUP_PID
    rm -rf $SCRATCH_DIR
    echo "Cleanup complete."
}
trap 'cleanup' SIGTERM
periodic_backup() {
    while true; do
        echo "Performing periodic backup..."
        sleep 4m
        find "$SCRATCH_DIR" -type f ! \( -name "*.py" -o -name "*.nii.gz" -o -name "*.pyc" \) ! -path "$SCRATCH_DIR/raw/*" ! -path "$SCRATCH_DIR/bg/*" ! -path "$SCRATCH_DIR/gt/*" ! -path "$SCRATCH_DIR/pkl_dir/*" | while read -r file; do
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
echo "Starting periodic backup..."
periodic_backup &
BACKUP_PID=$!

## Save the param.json file to the target directory and scratch directory
cp -f $P_DIR/params.json $TARGET_DIR/
cp -f $P_DIR/params.json $SCRATCH_DIR/

## Copy all the necessary data
cp -f "$P_DIR/helper_functions.py" "$SCRATCH_DIR"
cp -f "$P_DIR/calc_metrics.py" "$SCRATCH_DIR"
cp -f $DATASET $SCRATCH_DIR
cp -f $VAE_LDM_TXT $SCRATCH_DIR
cp -f $CN_TXT $SCRATCH_DIR
cp -f $RADNET $SCRATCH_DIR

cd $SCRATCH_DIR

eval "$(conda shell.bash hook)"
conda activate myenv
echo "env activated"

python -u calc_metrics.py
cleanup


