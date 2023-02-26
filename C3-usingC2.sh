#!/bin/bash
#
#
#SBATCH --account=edu               # Replace ACCOUNT with your group account name
#SBATCH --job-name=HW2-C3-1         # The job name.
#SBATCH -c 3                        # The number of cpu cores to use
#SBATCH -t 0-00:30                  # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=10gb          # The memory the job will use per cpu core
#SBATCH --mail-user=yva2002@columbia.edu

module load anaconda

conda init bash

source ~/.bashrc

conda activate hw2

python3 lab2.py C2 --dataloader_workers 0 --outfile C3-0worker.txt
python3 lab2.py C2 --dataloader_workers 4 --outfile C3-4workers.txt
python3 lab2.py C2 --dataloader_workers 8 --outfile C3-8workers.txt
python3 lab2.py C2 --dataloader_workers 12 --outfile C3-16workers.txt