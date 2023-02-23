#!/bin/bash
#
# Simple "Hello World" submit script for Slurm.
#
#SBATCH --account=edu         # Replace ACCOUNT with your group account name
#SBATCH --job-name=HW2-C1     # The job name.
#SBATCH -c 3                      # The number of cpu cores to use
#SBATCH -t 0-00:30                 # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=10gb        # The memory the job will use per cpu core

module load anaconda

conda init bash

source ~/.bashrc

conda activate hw2

python3 lab2.py C2