#!/bin/bash 
#SBATCH --qos=gpu_normal
#SBATCH -J gpu_test 
#SBATCH --output=gpu_test_output_%j.txt 
#SBATCH --error=gpu_test_error_%j.txt 
#SBATCH -p gpu_p
#SBATCH -q gpu_normal
#SBATCH -w supergpu08
#SBATCH --gres=gpu:1 
#SBATCH --nice=1000
#SBATCH -c 16
#SBATCH --time=00:10:00
#SBATCH --mem=10G 
nvidia-smi
