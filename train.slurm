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

## Start timer
start_time=$(date +%s)

## Directory and file definition
SCRATCH_DIR=/localscratch/$USER/job_$SLURM_JOB_ID
PREPROCESSING_OUTPUT=$SCRATCH_DIR/dataset.hdf5
TARGET_DIR=$SLURM_SUBMIT_DIR/job_$SLURM_JOB_ID
DATASET=$SLURM_SUBMIT_DIR/data/dataset.hdf5 ## check this out !!!!

if [ "$CP_VAE" -ne 0 ]; then
    AE_CHECKPOINT="$P_DIR/$VAE_DIR/job_$JVAE/vae_checkpoint_epoch_$CP_VAE.pth"
else
    AE="$P_DIR/$VAE_DIR/job_$JVAE/vae_model_*.pth"
fi

if [ "$CP_LDM" -ne 0 ]; then
    LDM_CHECKPOINT="$P_DIR/$LDM_DIR/job_$JLDM/ldm_checkpoint_epoch_$CP_LDM.pth"
else
    LDM="$P_DIR/$LDM_DIR/job_$JLDM/ldm_model_*.pth"
fi

CN_CHECKPOINT="$P_DIR/$CN_DIR/job_$JCN/cn_checkpoint_epoch_$CP_CN.pth"

FILENAME=$(date "+%Y%m%d.txt")
FILEPATH="$TARGET_DIR/$FILENAME"

## Create the directories
echo "Creating directories..."
mkdir -p "$SCRATCH_DIR/bg" "$SCRATCH_DIR/raw" "$SCRATCH_DIR/gt" "$TARGET_DIR"
test -d "$SLURM_SUBMIT_DIR/data" || mkdir "$SLURM_SUBMIT_DIR/data"
touch "$FILEPATH"

## Link the Job logfiles to the logfiles in the job directory
echo "Linking job logfiles..."
ln -f "o_$SLURM_JOB_ID.txt" "$TARGET_DIR/output.txt"
ln -f "e_$SLURM_JOB_ID.txt" "$TARGET_DIR/error.txt"

## Save the param.json file to the target directory and scratch directory
echo "Copying param.json..."
cp -f "$P_DIR/params.json" "$TARGET_DIR"
cp -f "$P_DIR/params.json" "$SCRATCH_DIR"

## Copy helper functions
echo "Copying helper functions..."
cp -f "$P_DIR/helper_functions.py" "$SCRATCH_DIR"

## Logging functions definition
log_and_copy() {
    local src=$1
    local dest=$2
    echo "$(basename "$src")" >> "$FILEPATH"
    cp "$src" "$dest"
}

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

cleanup() {
    echo "Final backup and cleanup..."
    find "$SCRATCH_DIR" -type f ! \( -name "*.py" -o -name "*.nii.gz" -o -name "*.pyc" \) ! -path "$SCRATCH_DIR/bg/*" ! -path "$SCRATCH_DIR/raw/*" ! -path "$SCRATCH_DIR/gt/*" ! -path "$SCRATCH_DIR/pkl_dir/*" | while read -r file; do
        basefile=$(basename "$file")
        if [[ "$basefile" == *.png ]] || [[ "$basefile" == *.hdf5 ]]; then
            cp "$file" "$TARGET_DIR/$basefile"
        else
            if ! grep -q "$basefile" "$FILEPATH"; then
                log_and_copy "$file" "$TARGET_DIR"
            fi
        fi
    done
    ## remove the output and error because they are already stored in the job file
    #rm "$SLURM_SUBMIT_DIR/o_$SLURM_JOB_ID.txt"
    #rm "$SLURM_SUBMIT_DIR/e_$SLURM_JOB_ID.txt"

    kill "$BACKUP_PID"
    rm -rf "$SCRATCH_DIR"
    echo "Cleanup complete."
}

trap 'cleanup' SIGTERM

# Function to check remaining time and perform cleanup if necessary
check_remaining_time_and_cleanup() {
    while true; do
        # Get the total job time (2 days in seconds)
        total_time=$((2 * 24 * 3600))
        
        # Get the current time in seconds since the epoch
        current_time=$(date +%s)
        echo "current_time : $current_time"
        # Calculate the elapsed time since the job started
        elapsed_seconds=$((current_time - start_time))
        echo "job_start_time: $start_time"
        # Calculate the remaining time
        remaining_time=$((total_time - elapsed_seconds))
        
        echo "Elapsed time: $elapsed_seconds seconds"
        echo "Remaining time: $remaining_time seconds"
        # If remaining time is less than 15 minutes, perform cleanup
        if [ "$remaining_time" -lt 900 ]; then
            echo "Remaining time is less than 15 minutes. Performing cleanup..."
            # Send SIGTERM signal to the Python process group
            kill -s SIGTERM -- -$$
            break
        fi
        
        # Sleep for a specified interval before checking again (e.g., every 10 minutes)
        sleep 600
    done
}

# Run the check_remaining_time_and_cleanup function in the background
echo "Starting time check..."
check_remaining_time_and_cleanup &

# Change to local scratch directory
echo "Changing to scratch directory..."
cd "$SCRATCH_DIR"

eval "$(conda shell.bash hook)"
conda activate myenv
echo "env activated"

echo "Starting periodic backup..."
periodic_backup &
BACKUP_PID=$!

# Data Preprocessing
echo "Starting data preprocessing..."
if [[ ! -e "$DATASET" ]] || [[ "$BROKEN_DS" -eq 1 ]]; then
    if [ "$SIZE" = 'xl' ]; then
        cp "$P_DIR/cp_xl_data.py" "$SCRATCH_DIR/"
        # cp "$IDS_FILE" "$TARGET_DIR/"
        python -u cp_xl_data.py --target_dir "$SCRATCH_DIR" --bg "$BG_XL" --raw "$RAW_XL" --ids "$IDS_FILE" --num_patches "$NUM_PATCH"
        echo "print directory items"
        ls $SCRATCH_DIR/raw
        ls $SCRATCH_DIR/gt
        ls $SCRATCH_DIR/bg
        cp "$SCRATCH_DIR/ids.json" "$SLURM_SUBMIT_DIR/data/"
    elif [ "$SIZE" = 'xs' ]; then
        cp -r "$BG_XS" "$SCRATCH_DIR"
        cp -r "$RAW_XS" "$SCRATCH_DIR"
        cp -r "$GT_XS" "$SCRATCH_DIR"
    fi

    cp "$P_DIR/preprocessing.py" "$SCRATCH_DIR/"
    python -u preprocessing.py --data_path "$SCRATCH_DIR" --output_file "$PREPROCESSING_OUTPUT"
    cp "$PREPROCESSING_OUTPUT" "$SLURM_SUBMIT_DIR/data/"
    echo "preprocessing completed"

elif [[ -e "$DATASET" ]] && [[ "$BROKEN_DS" -eq 0 ]]; then
    cp "$DATASET" "$SCRATCH_DIR/"
    echo "dataset.hdf5 copied successfully."
fi

# Train VAE
if [ "$MODEL" = "vae" ]; then
    echo "training vae"
    if [ -e "$AE_CHECKPOINT" ]; then
        cp "$AE_CHECKPOINT" "$SCRATCH_DIR/"
        echo "vae_checkpoint copied successfully."
    fi
    cp "$P_DIR/train_vae.py" "$SCRATCH_DIR/"
    python -u train_vae.py --dataset_file "$PREPROCESSING_OUTPUT"

# Train LDM
elif [ "$MODEL" = "ldm" ]; then
    echo "training ldm"
    if [ -e "$LDM_CHECKPOINT" ]; then
        cp "$LDM_CHECKPOINT" "$SCRATCH_DIR/"
        echo "ldm_checkpoint copied successfully."
    fi 
    cp "$AE" "$SCRATCH_DIR/"
    cp "$P_DIR/train_ldm.py" "$SCRATCH_DIR/"
    python -u train_ldm.py --dataset_file "$PREPROCESSING_OUTPUT" --job "$SLURM_JOB_ID"
    cp "$SCRATCH_DIR/samples_*.zip" "$TARGET_DIR"

# Train CN
elif [ "$MODEL" = "cn" ]; then
    echo "training cn"
    if [ -e "$CN_CHECKPOINT" ]; then
        cp "$CN_CHECKPOINT" "$SCRATCH_DIR/"
        echo "cn_checkpoint copied successfully."
    fi
    cp "$AE" "$SCRATCH_DIR/"
    cp "$LDM" "$SCRATCH_DIR/"
    cp "$P_DIR/train_cn.py" "$SCRATCH_DIR/"
    python -u train_cn.py --dataset_file "$PREPROCESSING_OUTPUT"
fi

cleanup
