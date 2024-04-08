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
##SBATCH -w supergpu02

## directory and file definition
SCRATCH_DIR=/localscratch/$USER/job_$SLURM_JOB_ID
PREPROCESSING_OUTPUT=$SCRATCH_DIR/dataset.hdf5
TARGET_DIR=$SLURM_SUBMIT_DIR/job_$SLURM_JOB_ID

DATASET=$SLURM_SUBMIT_DIR/data/dataset.hdf5

if ["$CP_VAE" -ne 0] then
    AE_CHECKPOINT="$P_DIR/$VAE_DIR/job_$JVAE/vae_checkpoint_epoch_$CP_VAE.pth"
elif ["$CP_VAE" -eq 0] then
    AE="$P_DIR/$VAE_DIR/job_$JVAE/vae_model_*.pth"
fi

if ["$CP_LDM" -ne 0] then
    LDM_CHECKPOINT="$P_DIR/$LDM_DIR/job_$JLDM/ldm_checkpoint_epoch_$CP_LDM.pth"
elif ["$CP_LDM" -eq 0] then
    LDM="$P_DIR/$LDM_DIR/job_$JLDM/ldm_model_*.pth"
fi

CN_CHECKPOINT="$P_DIR/$CN_DIR/job_$JCN/cn_checkpoint_epoch_$CP_CN.pth"


FILENAME=$(date "+%Y%m%d.txt")
FILEPATH="$TARGET_DIR/$FILENAME"

## Create the directories
mkdir -p $SCRATCH_DIR
mkdir -p $SCRATCH_DIR/bg
mkdir -p $SCRATCH_DIR/raw
mkdir -p $SCRATCH_DIR/gt
mkdir -p $TARGET_DIR
touch $FILEPATH

## Save the param.json file to the target directory
cp -f $P_DIR/params.json $TARGET_DIR

## Logging functions definition
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

# Change to local scratch directory
cd $SCRATCH_DIR

eval "$(conda shell.bash hook)"
conda activate myenv
echo "env activated"

periodic_backup &
BACKUP_PID=$!

# Data Preprocessing
if [[ ! -e "$DATASET" ]] || [["$BROKEN_DS" -eq 1]]; then
    if [ "$SIZE" = 'xl' ]; then
        cp $P_DIR/cp_xl_data.py $SCRATCH_DIR/
        python -u cp_xl_data.py --target_dir $SCRATCH_DIR --bg $BG_XL --raw $RAW_XL --ids $IDS_FILE

    elif [ "$SIZE" = 'xs' ]; then
        cp -r "$BG_XS" "$SCRATCH_DIR"
        cp -r "$RAW_XS" "$SCRATCH_DIR"
        cp -r "$GT_XS" "$SCRATCH_DIR"
    fi

    cp $P_DIR/preprocessing.py $SCRATCH_DIR/
    python -u preprocessing.py --data_path $SCRATCH_DIR --output_file $PREPROCESSING_OUTPUT --data_size $SIZE --gt_th $GT_TH
    echo "preprocessing completed"

elif  [[ -e "$DATASET" ]] && [["$BROKEN_DS" -eq 0]] ; then
    cp $DATASET $SCRATCH_DIR/
    echo "dataset.hdf5 copied successfully."
fi

# Train VAE
if [ "$MODEL" = "vae" ]; then
    echo "training vae"
    if [ -e "$AE_CHECKPOINT" ]; then
        cp $AE_CHECKPOINT $SCRATCH_DIR/
        echo "vae_checkpoint copied successfully."
    fi
    cp $P_DIR/train_vae.py $SCRATCH_DIR/
    python -u train_vae.py --dataset_file $PREPROCESSING_OUTPUT

# Train LDM
elif [ "$MODEL" = "ldm" ]; then
    echo "training ldm"
    if [ -e "$LDM_CHECKPOINT" ]; then
        cp $LDM_CHECKPOINT $SCRATCH_DIR/
        echo "ldm_checkpoint copied successfully."
    fi 
    cp $AE $SCRATCH_DIR/
    cp $P_DIR/train_ldm.py $SCRATCH_DIR/
    python -u train_ldm.py --output_file $OUTPUTFILE --lr $LR --data_size $SIZE --job $SLURM_JOB_ID

# Train $CN
elif [ "$MODEL" = "cn" ]; then
    echo "training cn"
    if [ -e "$CN_CHECKPOINT" ]; then
        cp $CN_CHECKPOINT $SCRATCH_DIR/
        echo "cn_checkpoint copied successfully."
    fi
    cp $AE $SCRATCH_DIR/
    cp $LDM $SCRATCH_DIR/
    cp $P_DIR/train_cn.py $SCRATCH_DIR/
    python -u train_cn.py --output_file $OUTPUTFILE --lr $LR
fi

cp $SCRATCH_DIR/samples_*.zip $TARGET_DIR

cleanup
