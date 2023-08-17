#!/bin/bash 
#SBATCH -J cpu_test 
#SBATCH --output=cpu_test_output_%j.txt 
#SBATCH --error=cpu_test_error_%j.txt 
#SBATCH -p normal_q 
#SBATCH -c 1 
#SBATCH --time=00:10:00 


lscpu
