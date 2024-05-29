#!/bin/bash

#SBATCH --job-name=mlp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=24
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --partition=long
#SBATCH --error=/home/mila/c/chris.emezue/representation-learning-assignment/slurmerror_mlp.txt
#SBATCH --output=/home/mila/c/chris.emezue/representation-learning-assignment/slurmoutput_mlp.txt


cd /home/mila/c/chris.emezue/representation-learning-assignment
source /home/mila/c/chris.emezue/gsl-env/bin/activate
module load python/3.7
python chris-assignment-mlp.py $1