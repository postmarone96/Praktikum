#!/bin/bash

# Collect arguments or set defaults
SIZE=$(jq -r '.size' params.json)
MODEL=$(jq -r '.model' params.json)
LR=$(jq -r '.learning_rate' params.json)
CP=$(jq -r '.checkpoint' params.json)
# Directory
P_DIR=$(jq -r '.project_dir' params.json)
BG_XS=$(jq -r '.bg_xs' params.json)
RAW_XS=$(jq -r '.raw' params.json)
GT_XS=$(jq -r '.gt' params.json)
# Visual Auto Encoder
JVAE=$(jq -r '.jobs.vae.id' params.json)
VAE_DIR=$(jq -r '.jobs.vae.directory' params.json)
# Latent Diffusion 
JLDM=$(jq -r '.jobs.ldm.id' params.json)
LDM_DIR=$(jq -r '.jobs.ldm.directory' params.json)
# Control Net
JCN=$(jq -r '.jobs.cn.id' params.json)
CN_DIR=$(jq -r '.jobs.cn.directory' params.json)

# Export them as environment variables so they can be accessed within the SLURM script
export SIZE
export MODEL
export LR
export CP
export P_DIR
export BG
export RAW
export GT
export JVAE
export VAE_DIR
export JLDM
export LDM_DIR
export JCN
export CN_DIR



test -d train_${SIZE} || mkdir train_${SIZE}
cp -f train.slurm train_${SIZE}/
cd train_${SIZE}

# Submit the SLURM job
sbatch train.slurm
