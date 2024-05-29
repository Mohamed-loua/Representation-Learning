#!/bin/bash

#SBATCH --job-name=yelp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=24
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --partition=long
#SBATCH --error=/home/mila/c/chris.emezue/representation-learning-assignment/assignment2/slurms/slurmerror_perturb.txt
#SBATCH --output=/home/mila/c/chris.emezue/representation-learning-assignment/assignment2/slurms/slurmoutput_perturb.txt


cd /home/mila/c/chris.emezue/representation-learning-assignment/assignment2
source /home/mila/c/chris.emezue/gsl-env/bin/activate
module load python/3.7
#python main.py $1
python main_perturb.py $1 $2