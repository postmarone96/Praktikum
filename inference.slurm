#!/bin/bash

#SBATCH -c 16
#SBATCH -J Inference
#SBATCH --output=o_%j.txt
#SBATCH --error=e_%j.txt
#SBATCH -p gpu_p
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=2-00:00:00
#SBATCH --nice=1000
#SBATCH --qos=gpu_normal

## Initiate directories
SCRATCH_DIR=/localscratch/$USER/job_$SLURM_JOB_ID
DATASET=$SCRATCH_DIR/dataset.hdf5
TARGET_DIR=$SLURM_SUBMIT_DIR/job_$SLURM_JOB_ID
bg_dir="/home/viro/marouane.hajri/Praktikum/train_xs/data/test/bg"
raw_dir="/home/viro/marouane.hajri/Praktikum/train_xs/data/test/raw"
gt_dir="/home/viro/marouane.hajri/Praktikum/train_xs/data/test/gt"
scratch_bg_dir="$SCRATCH_DIR/bg"
scratch_raw_dir="$SCRATCH_DIR/raw"
scratch_gt_dir="$SCRATCH_DIR/gt"
AE=$P_DIR/$VAE_DIR/job_$JVAE/vae_model_*.pth
LDM=$P_DIR/$LDM_DIR/job_$JLDM/ldm_model_*.pth
CN=$P_DIR/$CN_DIR/job_$JCN/cn_model_*.pth

## Create directories
mkdir $TARGET_DIR
mkdir $SCRATCH_DIR
mkdir -p "$scratch_bg_dir"
mkdir -p "$scratch_raw_dir"
mkdir -p "$scratch_gt_dir"

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
    find $SCRATCH_DIR -type f ! \( -name "*.py" -o -name "*.pyc" \) ! -path "$SCRATCH_DIR/bg/*" ! -path "$SCRATCH_DIR/raw/*" ! -path "$SCRATCH_DIR/gt/*" ! -path "$SCRATCH_DIR/pkl_dir/*" | while read -r file; do
        basefile=$(basename $file)
        if [[ $basefile == *.png ]] || [[ $basefile == *.hdf5 ]]; then
            cp $file $TARGET_DIR/$basefile
        else
            if ! grep -q $basefile $FILEPATH; then
                log_and_copy $file $TARGET_DIR
            fi
        fi
    done
    rm -rf $SCRATCH_DIR
    echo "Cleanup complete."
}
trap 'cleanup' SIGTERM

## Save the param.json file to the target directory and scratch directory
cp -f $P_DIR/params.json $TARGET_DIR/
cp -f $P_DIR/params.json $SCRATCH_DIR/


## Copy all the necessary data
cp -f "$P_DIR/helper_functions.py" "$SCRATCH_DIR"
cp -f "$P_DIR/calc_metrics.py" "$SCRATCH_DIR"
cp $AE $SCRATCH_DIR/
cp $LDM $SCRATCH_DIR/
cp $CN $SCRATCH_DIR/
cp $P_DIR/preprocessing.py $SCRATCH_DIR/
cp $P_DIR/predict.py $SCRATCH_DIR/

cd $SCRATCH_DIR

source /home/viro/marouane.hajri/miniconda3/bin/activate myenv
eval "$(conda shell.bash hook)"
conda activate myenv
echo "env activated"

# Loop over each file in the bg directory
for file in "$bg_dir"/*; do
    filename=$(basename "$file")
    cp -r "$file" "$scratch_bg_dir/"
    cp -r "$raw_dir/$filename" "$scratch_raw_dir/"
    cp -r "$gt_dir/$filename" "$scratch_gt_dir/"
    find "$SCRATCH_DIR"
    python -u preprocessing.py --data_path $SCRATCH_DIR --output_file $DATASET
    python -u predict.py --dataset_file $DATASET
    rm -rf $scratch_raw_dir/$filename
    rm -rf $scratch_bg_dir/$filename
    rm -rf $scratch_gt_dir/$filename
    rm -rf $DATASET
done

cleanup