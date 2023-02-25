#!/bin/bash
#
# Simple "Hello World" submit script for Slurm.
#
#SBATCH --account=edu           # Replace ACCOUNT with your group account name
#SBATCH --job-name=HW2-C6       # The job name.
#SBATCH -c 3                    # The number of cpu cores to use
#SBATCH -t 0-07:00              # Runtime in D-HH:MM
#SBATCH --gres=gpu
#SBATCH --constraint=k80
#SBATCH --mem-per-cpu=10gb        # The memory the job will use per cpu core
#SBATCH --mail-user=yva2002@columbia.edu

module load anaconda

conda init bash

source ~/.bashrc

rm -rf ~/pytorch-cifar-benchmarking/data

conda activate hw2

python3 lab2.py C6 --cuda --optimizer SGD --outfile C6-SGD.txt
python3 lab2.py C6 --cuda --optimizer SGD_Nesterov --outfile C6-SGD-Nesterov.txt
python3 lab2.py C6 --cuda --optimizer Adam --outfile C6-Adam.txt
python3 lab2.py C6 --cuda --optimizer Adagrad --outfile C6-Adagrad.txt
python3 lab2.py C6 --cuda --optimizer Adadelta --outfile C6-Adadelta.txt