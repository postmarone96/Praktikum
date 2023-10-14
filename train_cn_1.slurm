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

# Directory Definitions
echo "Directory definition"
SCRATCH_DIR="/localscratch/$USER/job_$SLURM_JOB_ID"
TARGET_DIR="$SLURM_SUBMIT_DIR/job_$SLURM_JOB_ID"
FILENAME="$(date "+%Y%m%d.txt")"
FILEPATH="$TARGET_DIR/$FILENAME"

# Directory Setup
echo "Creating the directories"
mkdir -p "$SCRATCH_DIR"
mkdir -p "$TARGET_DIR"
touch "$FILEPATH"

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
    rm -rf "$SCRATCH_DIR"
    echo "Cleanup complete."
}

trap 'cleanup' SIGTERM

# Move to the local scratch directory
echo "Changing to local scratch directory"
cd "$SCRATCH_DIR"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate myenv
echo "Environment activated"

# Copy files to scratch directory
SRC_DIR="/lustre/groups/iterm/Annotated_Datasets/Annotated Datasets/UCHL1 HFD - Peripheral Nervous System instance"
cp -r "$SRC_DIR/bg" "$SCRATCH_DIR/"
cp -r "$SRC_DIR/raw" "$SCRATCH_DIR/"
cp -r "$SRC_DIR/gt" "$SCRATCH_DIR/"

# Print the contents of bg, raw, and gt directories
echo "Contents of the bg directory:"
ls "$SCRATCH_DIR/bg"
echo "Contents of the raw directory:"
ls "$SCRATCH_DIR/raw"
echo "Contents of the gt directory:"
ls "$SCRATCH_DIR/gt"

cleanup