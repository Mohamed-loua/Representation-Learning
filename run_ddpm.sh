#!/bin/bash

#SBATCH --job-name=ddpm
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=72:00:00
#SBATCH --partition=long
#SBATCH --error=/home/mila/c/chris.emezue/representation-learning-assignment/slurmerror_ddpm.txt
#SBATCH --output=/home/mila/c/chris.emezue/representation-learning-assignment/slurmoutput_ddpm.txt


cd /home/mila/c/chris.emezue/representation-learning-assignment/assignment3
source /home/mila/c/chris.emezue/gsl-env/bin/activate
module load python/3.7
python chris_20225597_ddpm.py