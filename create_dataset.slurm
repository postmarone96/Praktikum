#!/bin/bash

#SBATCH -c 16
#SBATCH -J train_job
#SBATCH --output=o_data_%j.txt
#SBATCH --error=e_data_%j.txt
#SBATCH -p gpu_p
#SBATCH --gres=gpu:2
#SBATCH --mem=160G
#SBATCH --time=2-00:00:00
#SBATCH --nice=1000
#SBATCH --qos=gpu_normal

IDS_FILE="$HOME/Praktikum/ids.txt"

mkdir -p $HOME/Praktikum/train_xl/data/bg/
mkdir -p $HOME/Praktikum/train_xl/data/raw/

while IFS= read -r id; do
    cp -r "/lustre/groups/iterm/Rami/HFD/HFD_neurons/HFD_210320_UCHL1_755_HFD_DORSAL_l_1x_35o_4x8_GRB12-7-12_17-00-17/C00/${id}" "$HOME/Praktikum/train_xl/data/bg/"
done < "$IDS_FILE"

while IFS= read -r id; do
    cp -r "/lustre/groups/iterm/Rami/HFD/HFD_neurons/HFD/HFD_210320_UCHL1_755_HFD_DORSAL_l_1x_35o_4x8_GRB12-7-12_17-00-17/C02/${id}" "$HOME/Praktikum/train_xl/data/raw/"
done < "$IDS_FILE"