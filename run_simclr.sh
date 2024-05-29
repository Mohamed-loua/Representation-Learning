#!/bin/bash

#SBATCH --job-name=simCLR
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=72:00:00
#SBATCH --partition=long
#SBATCH --error=/home/mila/c/chris.emezue/representation-learning-assignment/slurmerror_simCLR_use_best.txt
#SBATCH --output=/home/mila/c/chris.emezue/representation-learning-assignment/slurmoutput_simCLR_use_best.txt


cd /home/mila/c/chris.emezue/representation-learning-assignment/assignment3
source /home/mila/c/chris.emezue/gsl-env/bin/activate
module load python/3.7
python chris_20225597_simclr_use_best.py
