#!/bin/bash

# Collect arguments
BROKEN_DS=$(jq -r '.broken_ds' params.json)
SIZE=$(jq -r '.size' params.json)
GT_TH=$(jq -r '.gt_th' params.json)
MODEL=$(jq -r '.model' params.json)
LR=$(jq -r '.learning_rate' params.json)
IDS_FILE=$(jq -r '.ids_file' params.json)
export BROKEN_DS
export SIZE
export GT_TH
export MODEL
export LR
export IDS_FILE

# Project Directory
P_DIR=$(jq -r '.project_dir' params.json)
export P_DIR

# XS
BG_XS=$(jq -r '.data.xs.bg' params.json)
RAW_XS=$(jq -r '.data.xs.raw' params.json)
GT_XS=$(jq -r '.data.xs.gt' params.json)
export BG_XS
export RAW_XS
export GT_XS

# XL
BG_XL=$(jq -r '.data.xl.bg' params.json)
RAW_XL=$(jq -r '.data.xl.raw' params.json)
export BG_XL
export RAW_XL

# Visual Auto Encoder
CP_VAE=$(jq -r '.jobs.vae.cp' params.json)
JVAE=$(jq -r '.jobs.vae.id' params.json)
VAE_DIR=$(jq -r '.jobs.vae.directory' params.json)
export CP_VAE
export JVAE
export VAE_DIR

# Latent Diffusion
CP_LDM=$(jq -r '.jobs.ldm.cp' params.json)
JLDM=$(jq -r '.jobs.ldm.id' params.json)
LDM_DIR=$(jq -r '.jobs.ldm.directory' params.json)
export CP_LDM
export JLDM
export LDM_DIR

# Control Net
CP_CN=$(jq -r '.jobs.cn.cp' params.json)
JCN=$(jq -r '.jobs.cn.id' params.json)
CN_DIR=$(jq -r '.jobs.cn.directory' params.json)
export CP_CN
export JCN
export CN_DIR

test -d train_${SIZE} || mkdir train_${SIZE}
cp -f train.slurm train_${SIZE}/
cd train_${SIZE}

# Submit the SLURM job
sbatch train.slurm
